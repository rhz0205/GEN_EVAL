from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from gen_eval.dataset import load_manifest
from gen_eval.evaluator import (
    Evaluator,
    VIDEO_INTEGRITY_METRIC,
    apply_video_integrity_gate,
    build_gated_metric_result,
    build_video_integrity_gate,
    iter_enabled_metrics,
    is_quality_metric,
    normalize_metric_result,
    run_metric,
)
from gen_eval.metrics import register_builtin_metrics

def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    runtime = config.get("runtime") or {}
    backend = runtime.get("backend", "local")
    if not isinstance(backend, str):
        raise ValueError("runtime.backend must be a string when provided.")

    backend_name = backend.lower()
    if backend_name == "local":
        return _run_local(config)
    if backend_name == "ray":
        return _run_ray(config)
    raise ValueError(f"Unsupported runtime backend: {backend}")

def _run_local(config: dict[str, Any]) -> dict[str, Any]:
    evaluator = Evaluator(config)
    return evaluator.run()

def _run_ray(config: dict[str, Any]) -> dict[str, Any]:
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError(
            "Ray backend requested, but ray is not installed in the local environment."
        ) from exc

    runtime = config.get("runtime") or {}
    ray_address = runtime.get("ray_address", "auto")
    shard_size = _coerce_positive_int(runtime.get("shard_size")) or 1
    max_in_flight = _coerce_positive_int(runtime.get("max_in_flight"))
    num_gpus_per_task = _coerce_non_negative_float(
        runtime.get("num_gpus_per_task"),
        default=0.0,
    )
    num_cpus_per_task = _coerce_positive_int(runtime.get("num_cpus_per_task")) or 1

    if not ray.is_initialized():
        ray.init(address=ray_address)

    register_builtin_metrics()
    manifest_path = config["manifest_path"]
    samples = load_manifest(manifest_path)
    sample_payloads = [sample.to_dict() for sample in samples]

    def evaluate_shard(
        metric_name: str,
        metric_config: dict[str, Any],
        shard_payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        from gen_eval.metrics import register_builtin_metrics
        from gen_eval.schemas import GenerationSample

        register_builtin_metrics()
        shard_samples = [
            GenerationSample.from_dict(payload) for payload in shard_payloads
        ]
        return run_metric(metric_name, metric_config, shard_samples)

    evaluate_shard_remote = ray.remote(
        num_gpus=num_gpus_per_task,
        num_cpus=num_cpus_per_task,
    )(evaluate_shard)

    results = []
    initial_shard_count = max(1, len(_split_into_shards(sample_payloads, shard_size)))
    effective_max_in_flight = max_in_flight or initial_shard_count
    enabled_metrics = iter_enabled_metrics(config.get("metrics", {}))
    video_integrity_gate: dict[str, Any] | None = None

    for metric_name, metric_config in enabled_metrics:
        metric_payloads = sample_payloads
        if video_integrity_gate is not None and is_quality_metric(metric_name):
            valid_sample_ids = set(video_integrity_gate["valid_sample_ids"])
            metric_payloads = [
                payload
                for payload in sample_payloads
                if str(payload.get("sample_id") or "unknown") in valid_sample_ids
            ]

        sample_shards = _split_into_shards(metric_payloads, shard_size) if metric_payloads else []

        if not sample_shards:
            if video_integrity_gate is not None and is_quality_metric(metric_name):
                results.append(
                    build_gated_metric_result(
                        metric_name,
                        total_samples=len(sample_payloads),
                        invalid_sample_ids=list(video_integrity_gate["invalid_sample_ids"]),
                    )
                )
                continue
            results.append(run_metric(metric_name, metric_config, []))
            continue

        pending = [
            evaluate_shard_remote.remote(metric_name, metric_config, shard)
            for shard in sample_shards
        ]
        partials = _ray_gather_limited(ray, pending, effective_max_in_flight)
        merged_result = _merge_metric_results(metric_name, partials, len(metric_payloads))
        if metric_name == VIDEO_INTEGRITY_METRIC:
            sample_id_payloads = [
                _SamplePayloadProxy(payload) for payload in sample_payloads
            ]
            video_integrity_gate = build_video_integrity_gate(
                sample_id_payloads,
                merged_result,
            )
            results.append(merged_result)
            continue

        if video_integrity_gate is not None and is_quality_metric(metric_name):
            merged_result = apply_video_integrity_gate(
                metric_name,
                merged_result,
                total_samples=len(sample_payloads),
                invalid_sample_ids=list(video_integrity_gate["invalid_sample_ids"]),
            )
        results.append(
            merged_result
        )

    return {
        "manifest_path": str(manifest_path),
        "num_samples": len(sample_payloads),
        "results": results,
    }

def _split_into_shards(items: list[Any], shard_size: int) -> list[list[Any]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be a positive integer.")
    return [items[i : i + shard_size] for i in range(0, len(items), shard_size)]

def _coerce_non_negative_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)) and value >= 0:
        return float(value)
    return default

def _ray_gather_limited(
    ray_module: Any, refs: list[Any], max_in_flight: int
) -> list[dict[str, Any]]:
    if max_in_flight <= 0 or len(refs) <= max_in_flight:
        return list(ray_module.get(refs))

    pending = list(refs)
    ready_results: list[dict[str, Any]] = []
    active = pending[:max_in_flight]
    remaining = pending[max_in_flight:]

    while active:
        ready, active = ray_module.wait(active, num_returns=1)
        ready_results.extend(ray_module.get(ready))
        if remaining:
            active.append(remaining.pop(0))
    return ready_results

def _merge_metric_results(
    metric_name: str,
    partial_results: list[dict[str, Any]],
    total_samples: int,
) -> dict[str, Any]:
    scoreless_result = _merge_scoreless_metric_results(
        metric_name,
        partial_results,
        total_samples,
    )
    if scoreless_result is not None:
        return scoreless_result

    valid_scores: list[float] = []
    valid_weights: list[int] = []
    failed = 0
    reasons: list[str] = []
    merged_details: dict[str, list[Any]] = defaultdict(list)
    averaged_detail_fields: dict[str, list[tuple[dict[str, Any], int]]] = defaultdict(
        list
    )
    scalar_detail_fields: dict[str, Any] = {}

    for partial in partial_results:
        partial = normalize_metric_result(metric_name, partial)

        status = str(partial.get("status", "unknown"))
        score = partial.get("score")
        num_samples = partial.get("num_samples")
        weight = (
            int(num_samples) if isinstance(num_samples, int) and num_samples > 0 else 0
        )

        if (
            status == "success"
            and isinstance(score, (int, float))
            and math.isfinite(float(score))
        ):
            valid_scores.append(float(score))
            valid_weights.append(weight or 1)
        elif status == "failed":
            failed += 1

        reason = partial.get("reason")
        if isinstance(reason, str) and reason:
            reasons.append(reason)

        details = partial.get("details")
        if isinstance(details, dict):
            for key, value in details.items():
                if isinstance(value, list):
                    merged_details[key].extend(value)
                elif isinstance(value, dict) and _is_numeric_mapping(value):
                    averaged_detail_fields[key].append((value, weight or 1))
                elif value is not None:
                    scalar_detail_fields.setdefault(key, value)

    if valid_scores:
        weighted_sum = sum(
            score * weight for score, weight in zip(valid_scores, valid_weights)
        )
        total_weight = sum(valid_weights)
        final_score = (
            weighted_sum / total_weight
            if total_weight > 0
            else sum(valid_scores) / len(valid_scores)
        )
        status = "success"
        reason = None
        num_valid_samples = total_weight if total_weight > 0 else len(valid_scores)
    else:
        final_score = None
        status = "failed" if failed else "skipped"
        reason = (
            reasons[0]
            if reasons
            else f"No sample produced a valid {metric_name} score."
        )
        num_valid_samples = 0

    detail_payload: dict[str, Any] = dict(merged_details)
    for key, value in averaged_detail_fields.items():
        detail_payload[key] = _merge_numeric_mappings(value)
    for key, value in scalar_detail_fields.items():
        detail_payload.setdefault(key, value)

    result: dict[str, Any] = {
        "metric": metric_name,
        "score": final_score,
        "num_samples": num_valid_samples,
        "details": detail_payload,
        "status": status,
    }
    if reason:
        result["reason"] = reason

    if "evaluated_samples" not in result["details"]:
        result["details"]["evaluated_samples"] = []
    if "skipped_samples" not in result["details"]:
        result["details"]["skipped_samples"] = []
    if "failed_samples" not in result["details"]:
        result["details"]["failed_samples"] = []
    result["details"]["total_samples_seen"] = total_samples
    return result

def _merge_scoreless_metric_results(
    metric_name: str,
    partial_results: list[dict[str, Any]],
    total_samples: int,
) -> dict[str, Any] | None:
    if metric_name == "video_integrity":
        return _merge_video_integrity_results(partial_results, total_samples)
    if is_quality_metric(metric_name):
        return _merge_named_score_metric_results(
            metric_name,
            partial_results,
            total_samples,
        )
    return None

def _merge_video_integrity_results(
    partial_results: list[dict[str, Any]],
    total_samples: int,
) -> dict[str, Any]:
    normalized_partials = [
        normalize_metric_result("video_integrity", partial) for partial in partial_results
    ]
    details = _merge_detail_lists(normalized_partials, total_samples)
    valid_sample_count = sum(
        int(partial.get("valid_sample_count") or 0) for partial in normalized_partials
    )
    invalid_sample_count = sum(
        int(partial.get("invalid_sample_count") or 0) for partial in normalized_partials
    )
    evaluated_count = valid_sample_count + invalid_sample_count
    failed = sum(1 for partial in normalized_partials if partial.get("status") == "failed")
    reasons = [
        partial.get("reason")
        for partial in normalized_partials
        if isinstance(partial.get("reason"), str) and partial.get("reason")
    ]

    if evaluated_count > 0:
        status = "success"
        reason = None
    else:
        status = "failed" if failed else "skipped"
        reason = (
            reasons[0]
            if reasons
            else "No sample could be evaluated for video integrity."
        )

    result: dict[str, Any] = {
        "metric": "video_integrity",
        "status": status,
        "num_samples": total_samples,
        "valid_sample_count": valid_sample_count,
        "invalid_sample_count": invalid_sample_count,
        "pass_rate": (
            valid_sample_count / evaluated_count if evaluated_count > 0 else None
        ),
        "details": details,
    }
    if reason:
        result["reason"] = reason
    return result

def _merge_named_score_metric_results(
    metric_name: str,
    partial_results: list[dict[str, Any]],
    total_samples: int,
) -> dict[str, Any]:
    from gen_eval.evaluator import (
        get_quality_metric_mean_field,
        get_quality_metric_mean_source_field,
    )

    normalized_partials = [
        normalize_metric_result(metric_name, partial) for partial in partial_results
    ]
    details = _merge_detail_lists(normalized_partials, total_samples)
    failed = sum(1 for partial in normalized_partials if partial.get("status") == "failed")
    reasons = [
        partial.get("reason")
        for partial in normalized_partials
        if isinstance(partial.get("reason"), str) and partial.get("reason")
    ]

    mean_field = get_quality_metric_mean_field(metric_name)
    mean_source_field = get_quality_metric_mean_source_field(metric_name)
    if mean_field is None or mean_source_field is None:
        raise KeyError(f"Unsupported scoreless metric merge: {metric_name}")

    weighted_sum = 0.0
    valid_evaluated_count = 0
    for partial in normalized_partials:
        if partial.get("status") != "success":
            continue
        score = partial.get(mean_source_field)
        count = partial.get("valid_sample_count")
        if not isinstance(count, int):
            count = partial.get("valid_evaluated_count")
        if (
            isinstance(score, (int, float))
            and math.isfinite(float(score))
            and isinstance(count, int)
            and count > 0
        ):
            weighted_sum += float(score) * count
            valid_evaluated_count += count

    if valid_evaluated_count > 0:
        status = "success"
        reason = None
        mean_score = weighted_sum / valid_evaluated_count
    else:
        status = "failed" if failed else "skipped"
        reason = (
            reasons[0]
            if reasons
            else f"No sample produced a valid {mean_field}."
        )
        mean_score = None

    result: dict[str, Any] = {
        "metric": metric_name,
        "status": status,
        "num_samples": total_samples,
        "valid_sample_count": valid_evaluated_count,
        mean_field: mean_score,
        "details": details,
    }
    if metric_name == "view_consistency":
        result["view_consistency_score"] = mean_score
        result["valid_evaluated_count"] = valid_evaluated_count
    if reason:
        result["reason"] = reason
    return result

def _merge_detail_lists(
    partial_results: list[dict[str, Any]],
    total_samples: int,
) -> dict[str, Any]:
    merged_details: dict[str, list[Any]] = defaultdict(list)
    for partial in partial_results:
        details = partial.get("details")
        if not isinstance(details, dict):
            continue
        for key in ("evaluated_samples", "skipped_samples", "failed_samples"):
            value = details.get(key)
            if isinstance(value, list):
                merged_details[key].extend(value)

    detail_payload: dict[str, Any] = dict(merged_details)
    if "evaluated_samples" not in detail_payload:
        detail_payload["evaluated_samples"] = []
    if "skipped_samples" not in detail_payload:
        detail_payload["skipped_samples"] = []
    if "failed_samples" not in detail_payload:
        detail_payload["failed_samples"] = []
    detail_payload["total_samples_seen"] = total_samples
    return detail_payload

class _SamplePayloadProxy:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.sample_id = payload.get("sample_id") or "unknown"

def _coerce_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None

def _is_numeric_mapping(value: dict[str, Any]) -> bool:
    if not value:
        return False
    return all(isinstance(item, (int, float)) for item in value.values())

def _merge_numeric_mappings(
    weighted_values: list[tuple[dict[str, Any], int]],
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    weights: dict[str, int] = defaultdict(int)

    for mapping, weight in weighted_values:
        for key, value in mapping.items():
            totals[key] += float(value) * weight
            weights[key] += weight

    merged: dict[str, float] = {}
    for key, total in totals.items():
        merged[key] = total / weights[key] if weights[key] > 0 else total
    return merged
