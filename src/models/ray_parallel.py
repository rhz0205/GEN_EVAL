from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from models.local_parallel import aggregate_module_results
from reference.preparer import (
    build_enabled_generator_configs,
    build_reference_generator,
    extract_samples,
    infer_run_context,
    load_json,
    merge_metadata,
    set_samples,
    write_json,
)


def evaluate_ray(
    *,
    samples: list[Any],
    metrics_config: dict[str, Any],
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    ray = _init_ray(runtime_config)
    if _evaluate_module_shard is None:
        raise RuntimeError("Ray remote evaluation entry is unavailable. Ensure ray is installed in the runtime environment.")
    shard_size = max(1, int(runtime_config.get("shard_size", 1) or 1))
    num_cpus = float(runtime_config.get("num_cpus_per_task", 1) or 1)
    num_gpus = float(runtime_config.get("num_gpus_per_task", 1) or 1)

    enabled_metrics = [
        (module_name, dict(module_config))
        for module_name, module_config in metrics_config.items()
        if isinstance(module_config, dict) and module_config.get("enabled", False)
    ]
    aggregated: dict[str, Any] = {}
    for module_name, module_config in enabled_metrics:
        shards = split_into_shards(samples, shard_size)
        wall_started_at = _perf_counter()
        tasks = [
            _evaluate_module_shard.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(
                module_name=module_name,
                module_config=module_config,
                shard_samples=shard,
            )
            for shard in shards
        ]
        payloads = ray.get(tasks)
        wall_time_seconds = round(_perf_counter() - wall_started_at, 6)
        worker_failures = [payload for payload in payloads if payload.get("status") != "success"]
        shard_results = [
            payload["result"]
            for payload in payloads
            if payload.get("status") == "success" and isinstance(payload.get("result"), dict)
        ]
        aggregated[module_name] = aggregate_module_results(
            module_name=module_name,
            shard_results=shard_results,
            worker_failures=worker_failures,
            num_samples=len(samples),
            wall_time_seconds=wall_time_seconds,
        )
    return aggregated


def prepare_reference_ray(
    *,
    reference_config: dict[str, Any],
    data_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    output_dir: str | Path,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    ray = _init_ray(runtime_config)
    if _prepare_reference_shard is None:
        raise RuntimeError("Ray remote reference entry is unavailable. Ensure ray is installed in the runtime environment.")
    shard_size = max(1, int(runtime_config.get("shard_size", 1) or 1))
    num_cpus = float(runtime_config.get("num_cpus_per_task", 1) or 1)
    num_gpus = float(runtime_config.get("num_gpus_per_task", 1) or 1)

    raw_payload = load_json(data_path)
    samples = extract_samples(raw_payload)
    reference_root = Path(output_dir)
    reference_root.mkdir(parents=True, exist_ok=True)
    run_context = infer_run_context(data_path)
    continue_on_error = bool((reference_config.get("reference") or {}).get("continue_on_error", False))
    generator_configs = build_enabled_generator_configs(reference_config)

    wall_started_at = _perf_counter()
    tasks = [
        _prepare_reference_shard.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(
            shard_samples=shard,
            generator_configs=generator_configs,
            output_dir=str(reference_root),
            continue_on_error=continue_on_error,
        )
        for shard in split_into_shards(samples, shard_size)
    ]
    shard_payloads = ray.get(tasks)
    wall_time_seconds = round(_perf_counter() - wall_started_at, 6)

    enriched_samples: list[dict[str, Any]] = []
    failed_samples: list[dict[str, Any]] = []
    generator_summary: dict[str, dict[str, int]] = {
        name: {"prepared": 0, "failed": 0} for name, _ in generator_configs
    }

    for shard_payload in shard_payloads:
        enriched_samples.extend(shard_payload.get("samples", []))
        failed_samples.extend(shard_payload.get("failed_samples", []))
        shard_summary = shard_payload.get("generator_summary")
        if isinstance(shard_summary, dict):
            for generator_name, counts in shard_summary.items():
                if generator_name not in generator_summary:
                    generator_summary[generator_name] = {"prepared": 0, "failed": 0}
                generator_summary[generator_name]["prepared"] += int(counts.get("prepared", 0) or 0)
                generator_summary[generator_name]["failed"] += int(counts.get("failed", 0) or 0)

    enriched_payload = set_samples(raw_payload, enriched_samples)
    write_json(output_path, enriched_payload)
    summary = {
        "status": "success",
        "dataset_name": run_context["dataset_name"],
        "data_count": run_context["data_count"],
        "timestamp": run_context["timestamp"],
        "data_file": str(Path(data_path)),
        "output_dir": str(Path(output_dir)),
        "num_samples": len(samples),
        "num_generators": len(generator_configs),
        "enriched_data_path": str(Path(output_path)),
        "reference_output_dir": str(reference_root),
        "generator_summary": generator_summary,
        "failed_samples": failed_samples,
        "continue_on_error": continue_on_error,
        "wall_time_seconds": wall_time_seconds,
    }
    write_json(summary_path, summary)
    return summary


def split_into_shards(samples: list[Any], shard_size: int) -> list[list[Any]]:
    return [samples[index : index + shard_size] for index in range(0, len(samples), shard_size)]


def _init_ray(runtime_config: dict[str, Any]) -> Any:
    import ray

    if not ray.is_initialized():
        ray.init(
            address=runtime_config.get("ray_address", "auto"),
            ignore_reinit_error=True,
            log_to_driver=True,
        )
    return ray


def _perf_counter() -> float:
    import time

    return time.perf_counter()


def _build_generators(generator_configs: list[tuple[str, dict[str, Any]]]) -> list[Any]:
    return [build_reference_generator(name, dict(config)) for name, config in generator_configs]


def _prepare_sample(
    *,
    sample: dict[str, Any],
    generators: list[Any],
    output_dir: str,
    continue_on_error: bool,
    generator_summary: dict[str, dict[str, int]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    enriched_sample = deepcopy(sample)
    raw_metadata = enriched_sample.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    if raw_metadata is not None and not isinstance(raw_metadata, dict):
        sample_id = str(enriched_sample.get("sample_id") or "unknown")
        raise ValueError(f"sample {sample_id} metadata must be an object.")
    enriched_sample["metadata"] = metadata
    sample_id = str(enriched_sample.get("sample_id") or "unknown")
    sample_failures: list[dict[str, str]] = []

    for generator in generators:
        try:
            patch = generator.prepare_sample(enriched_sample, Path(output_dir))
            metadata = merge_metadata(metadata, patch)
            enriched_sample["metadata"] = metadata
            generator_summary[generator.name]["prepared"] += 1
        except Exception as exc:
            generator_summary[generator.name]["failed"] += 1
            failure = {
                "generator": generator.name,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            sample_failures.append(failure)
            if not continue_on_error:
                raise RuntimeError(
                    f"Reference generation failed for sample={sample_id}, generator={generator.name}: {exc}"
                ) from exc

    if sample_failures:
        return enriched_sample, {"sample_id": sample_id, "errors": sample_failures}
    return enriched_sample, None


try:
    import ray

    @ray.remote
    def _evaluate_module_shard(
        *,
        module_name: str,
        module_config: dict[str, Any],
        shard_samples: list[Any],
    ) -> dict[str, Any]:
        from models.local_parallel import _worker_entry

        class QueueProxy:
            def __init__(self) -> None:
                self.payload: dict[str, Any] | None = None

            def put(self, payload: dict[str, Any]) -> None:
                self.payload = payload

        queue = QueueProxy()
        _worker_entry(
            worker_index=0,
            gpu_id=0,
            module_name=module_name,
            module_config=module_config,
            shard_samples=shard_samples,
            queue=queue,
        )
        if queue.payload is None:
            return {
                "worker_index": 0,
                "status": "failed",
                "gpu_id": 0,
                "sample_ids": [getattr(sample, "sample_id", "unknown") for sample in shard_samples],
                "reason": "Ray task produced no payload.",
            }
        return queue.payload


    @ray.remote
    def _prepare_reference_shard(
        *,
        shard_samples: list[dict[str, Any]],
        generator_configs: list[tuple[str, dict[str, Any]]],
        output_dir: str,
        continue_on_error: bool,
    ) -> dict[str, Any]:
        generators = _build_generators(generator_configs)
        generator_summary: dict[str, dict[str, int]] = {
            generator.name: {"prepared": 0, "failed": 0} for generator in generators
        }
        enriched_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []

        for sample in shard_samples:
            enriched_sample, failure = _prepare_sample(
                sample=sample,
                generators=generators,
                output_dir=output_dir,
                continue_on_error=continue_on_error,
                generator_summary=generator_summary,
            )
            enriched_samples.append(enriched_sample)
            if failure is not None:
                failed_samples.append(failure)

        return {
            "samples": enriched_samples,
            "failed_samples": failed_samples,
            "generator_summary": generator_summary,
        }
except Exception:
    _evaluate_module_shard = None
    _prepare_reference_shard = None
