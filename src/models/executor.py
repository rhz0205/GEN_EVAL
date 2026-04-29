from __future__ import annotations

from typing import Any


def run_prepare_reference_stage(
    *,
    reference_config: dict[str, Any],
    data_path: str,
    output_path: str,
    summary_path: str,
    output_dir: str,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    backend = str(runtime_config.get("backend", "serial"))
    if backend == "ray":
        from models.ray_parallel import prepare_reference_ray

        return prepare_reference_ray(
            reference_config=reference_config,
            data_path=data_path,
            output_path=output_path,
            summary_path=summary_path,
            output_dir=output_dir,
            runtime_config=runtime_config,
        )
    if backend != "serial" and backend != "local_multi_gpu":
        raise ValueError(f"Unsupported prepare_reference backend: {backend}")
    from reference import ReferencePreparer

    preparer = ReferencePreparer(reference_config)
    return preparer.prepare(
        data_path=data_path,
        output_path=output_path,
        summary_path=summary_path,
        output_dir=output_dir,
    )


def run_evaluate_stage(
    *,
    samples: list[Any],
    metrics_config: dict[str, Any],
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    backend = str(runtime_config.get("backend", "serial"))
    if backend == "local_multi_gpu":
        from models.local_parallel import evaluate_local_multi_gpu

        devices = normalize_runtime_devices(runtime_config)
        return evaluate_local_multi_gpu(
            samples=samples,
            metrics_config=metrics_config,
            devices=devices,
        )
    if backend == "ray":
        from models.ray_parallel import evaluate_ray

        return evaluate_ray(
            samples=samples,
            metrics_config=metrics_config,
            runtime_config=runtime_config,
        )
    if backend != "serial":
        raise ValueError(f"Unsupported evaluate backend: {backend}")
    raise RuntimeError("serial backend should be handled directly by GenEval.")


def normalize_runtime_devices(runtime_config: dict[str, Any]) -> list[int]:
    devices = runtime_config.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ValueError("runtime.devices must be a non-empty list when backend=local_multi_gpu.")
    normalized: list[int] = []
    for value in devices:
        if not isinstance(value, int):
            raise ValueError("runtime.devices must contain only integers.")
        normalized.append(int(value))
    return normalized
