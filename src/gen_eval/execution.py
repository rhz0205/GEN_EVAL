from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from gen_eval.dataset import load_manifest
from gen_eval.evaluator import (
    Evaluator,
    iter_enabled_metrics,
    normalize_metric_result,
    run_metric,
)
from gen_eval.metrics import register_builtin_metrics


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """根据运行配置执行评估，支持本地和Ray分布式两种后端"""
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


# Ray 的执行流程（例如 1000 个样本评估 6 个指标）：
# 1. 加载 manifest，并将样本转换为可序列化的 payload。
# 2. 根据 shard_size 将样本切分为多个 shard。
# 3. 遍历每个指标，为每个 shard 提交一个 Ray task。
#    例如：metric_a + shard_001 -> task, metric_a + shard_002 -> task。
# 4. 每个 task 内部对一批样本运行同一个指标，避免一个样本一个 task 造成过高调度开销。
# 5. 合并所有 shard-level partial results，得到该指标的最终结果。
def _run_ray(config: dict[str, Any]) -> dict[str, Any]:
    """使用 Ray 分布式执行评估，按指标与样本分片并行计算并合并结果。"""
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
    sample_shards = _split_into_shards(sample_payloads, shard_size)

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
    effective_max_in_flight = max_in_flight or max(1, len(sample_shards))

    for metric_name, metric_config in iter_enabled_metrics(config.get("metrics", {})):
        if not sample_shards:
            results.append(run_metric(metric_name, metric_config, []))
            continue

        pending = [
            evaluate_shard_remote.remote(metric_name, metric_config, shard)
            for shard in sample_shards
        ]
        partials = _ray_gather_limited(ray, pending, effective_max_in_flight)
        results.append(
            _merge_metric_results(metric_name, partials, len(sample_payloads))
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
            status == "ok"
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
        status = "ok"
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
