from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
import traceback
from typing import Any


def evaluate_local_multi_gpu(
    *,
    samples: list[Any],
    metrics_config: dict[str, Any],
    devices: list[int],
) -> dict[str, Any]:
    if not devices:
        raise ValueError("runtime.devices must be a non-empty list for local_multi_gpu backend.")

    enabled_metrics = [
        (module_name, dict(module_config))
        for module_name, module_config in metrics_config.items()
        if isinstance(module_config, dict) and module_config.get("enabled", False)
    ]
    if not enabled_metrics:
        return {}

    aggregated: dict[str, Any] = {}
    for module_name, module_config in enabled_metrics:
        aggregated[module_name] = _evaluate_single_module(
            module_name=module_name,
            module_config=module_config,
            samples=samples,
            devices=devices,
        )
    return aggregated


def _evaluate_single_module(
    *,
    module_name: str,
    module_config: dict[str, Any],
    samples: list[Any],
    devices: list[int],
) -> dict[str, Any]:
    shards = split_round_robin(samples, len(devices))
    non_empty_shards = [(index, shard) for index, shard in enumerate(shards) if shard]
    if not non_empty_shards:
        return {
            "metric": module_name,
            "status": "skipped",
            "num_samples": len(samples),
            "valid_sample_count": 0,
            "skipped_sample_count": 0,
            "failed_sample_count": 0,
            "details": {
                "evaluated_samples": [],
                "skipped_samples": [],
                "failed_samples": [],
            },
            "wall_time_seconds": 0.0,
            "aggregated_compute_seconds": 0.0,
            "duration_seconds": 0.0,
            "reason": f"No samples assigned to module {module_name}.",
        }

    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    processes: list[Any] = []
    wall_started_at = time.perf_counter()

    for worker_index, shard in non_empty_shards:
        gpu_id = devices[worker_index % len(devices)]
        process = ctx.Process(
            target=_worker_entry,
            args=(worker_index, gpu_id, module_name, module_config, shard, queue),
        )
        process.start()
        processes.append(process)

    worker_payloads: dict[int, dict[str, Any]] = {}
    for _ in processes:
        payload = queue.get()
        worker_payloads[int(payload.get("worker_index", -1))] = payload

    for process in processes:
        process.join()

    wall_time_seconds = round(time.perf_counter() - wall_started_at, 6)
    worker_failures = [
        payload
        for payload in worker_payloads.values()
        if payload.get("status") != "success"
    ]
    shard_results = [
        payload["result"]
        for payload in worker_payloads.values()
        if payload.get("status") == "success" and isinstance(payload.get("result"), dict)
    ]
    return aggregate_module_results(
        module_name=module_name,
        shard_results=shard_results,
        worker_failures=worker_failures,
        num_samples=len(samples),
        wall_time_seconds=wall_time_seconds,
    )


def _worker_entry(
    worker_index: int,
    gpu_id: int,
    module_name: str,
    module_config: dict[str, Any],
    shard_samples: list[Any],
    queue: Any,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        from modules import build_module

        started_at = time.perf_counter()
        module = build_module(module_name, dict(module_config))
        module_result = module.evaluate(shard_samples)
        compute_seconds = round(time.perf_counter() - started_at, 6)
        if isinstance(module_result, dict):
            module_result["aggregated_compute_seconds"] = compute_seconds
            module_result["duration_seconds"] = compute_seconds

        queue.put(
            {
                "worker_index": worker_index,
                "status": "success",
                "gpu_id": gpu_id,
                "sample_ids": [getattr(sample, "sample_id", "unknown") for sample in shard_samples],
                "result": module_result,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "worker_index": worker_index,
                "status": "failed",
                "gpu_id": gpu_id,
                "sample_ids": [getattr(sample, "sample_id", "unknown") for sample in shard_samples],
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def aggregate_module_results(
    *,
    module_name: str,
    shard_results: list[dict[str, Any]],
    worker_failures: list[dict[str, Any]],
    num_samples: int,
    wall_time_seconds: float,
) -> dict[str, Any]:
    metric_name = module_name
    evaluated_samples: list[dict[str, Any]] = []
    skipped_samples: list[dict[str, Any]] = []
    failed_samples: list[dict[str, Any]] = []
    aggregated_compute_seconds = 0.0
    valid_sample_count = 0
    invalid_sample_count = 0

    for shard_result in shard_results:
        metric_name = str(shard_result.get("metric", metric_name))
        aggregated_compute_seconds += float(shard_result.get("aggregated_compute_seconds", shard_result.get("duration_seconds", 0.0)) or 0.0)
        valid_sample_count += int(shard_result.get("valid_sample_count", 0) or 0)
        invalid_sample_count += int(shard_result.get("invalid_sample_count", 0) or 0)
        details = shard_result.get("details")
        if not isinstance(details, dict):
            continue
        evaluated_samples.extend(as_list_of_dicts(details.get("evaluated_samples")))
        skipped_samples.extend(as_list_of_dicts(details.get("skipped_samples")))
        failed_samples.extend(as_list_of_dicts(details.get("failed_samples")))

    for failure in worker_failures:
        reason = str(failure.get("reason", "worker failed"))
        for sample_id in failure.get("sample_ids", []):
            failed_samples.append({"sample_id": sample_id, "reason": reason})

    result: dict[str, Any] = {
        "metric": metric_name,
        "num_samples": num_samples,
        "valid_sample_count": valid_sample_count,
        "skipped_sample_count": len(skipped_samples),
        "failed_sample_count": len(failed_samples),
        "details": {
            "evaluated_samples": evaluated_samples,
            "skipped_samples": skipped_samples,
            "failed_samples": failed_samples,
        },
        "wall_time_seconds": round(wall_time_seconds, 6),
        "aggregated_compute_seconds": round(aggregated_compute_seconds, 6),
        "duration_seconds": round(wall_time_seconds, 6),
    }

    if invalid_sample_count > 0:
        result["invalid_sample_count"] = invalid_sample_count

    score_keys = collect_numeric_score_keys(evaluated_samples)
    if score_keys:
        for score_key in score_keys:
            values = collect_numeric_values(evaluated_samples, score_key)
            mean_value = mean_or_none(values)
            if mean_value is not None:
                result[f"mean_{score_key}"] = mean_value
                if has_top_level_key(shard_results, score_key):
                    result[score_key] = mean_value
        result["status"] = "success" if valid_sample_count > 0 else ("failed" if failed_samples else "skipped")
        if valid_sample_count == 0:
            result["reason"] = f"No sample produced a valid {score_keys[0]}."
        return result

    if any("video_integrity_passed" in sample for sample in evaluated_samples):
        pass_rate = safe_div(valid_sample_count, len(evaluated_samples)) if evaluated_samples else None
        result["pass_rate"] = pass_rate
        result["status"] = "success" if evaluated_samples else ("failed" if failed_samples else "skipped")
        if not evaluated_samples:
            result["reason"] = "No sample could be evaluated for video integrity."
        return result

    result["status"] = "success" if evaluated_samples else ("failed" if failed_samples else "skipped")
    if not evaluated_samples and failed_samples:
        result["reason"] = f"No sample produced a valid result for {metric_name}."
    return result


def split_round_robin(samples: list[Any], num_shards: int) -> list[list[Any]]:
    shards: list[list[Any]] = [[] for _ in range(max(1, num_shards))]
    for index, sample in enumerate(samples):
        shards[index % len(shards)].append(sample)
    return shards


def as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def collect_numeric_score_keys(samples: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for sample in samples:
        for key, value in sample.items():
            if key == "sample_id":
                continue
            if key.endswith("_score") and is_finite_number(value):
                keys.add(str(key))
    return sorted(keys)


def collect_numeric_values(samples: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = sample.get(key)
        if is_finite_number(value):
            values.append(float(value))
    return values


def has_top_level_key(shard_results: list[dict[str, Any]], key: str) -> bool:
    for shard_result in shard_results:
        if key in shard_result:
            return True
    return False


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a) / float(b)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
