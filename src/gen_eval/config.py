from __future__ import annotations

from pathlib import Path
from typing import Any

from gen_eval.result_writer import default_output_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read YAML configs. Install pyyaml in the evaluation environment."
        ) from exc

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object: {config_path}")
    return data


def resolve_dataset_config(dataset: str) -> tuple[Path, dict[str, Any]]:
    dataset_config_path = PROJECT_ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    return dataset_config_path, load_yaml(dataset_config_path)


def resolve_metric_config() -> tuple[Path, dict[str, Any]]:
    metric_config_path = PROJECT_ROOT / "configs" / "metrics.yaml"
    return metric_config_path, load_yaml(metric_config_path)


def normalize_runtime_config(runtime: dict[str, Any] | None) -> dict[str, Any]:
    runtime_config = dict(runtime or {})
    backend = runtime_config.get("backend", "local")
    if not isinstance(backend, str):
        raise ValueError("Run config field 'runtime.backend' must be a string when provided.")

    backend_name = backend.lower()
    runtime_config["backend"] = backend_name

    if backend_name == "ray":
        runtime_config.setdefault("ray_address", "auto")
        runtime_config.setdefault("num_workers", 0)
    else:
        runtime_config.setdefault("num_workers", 0)

    return runtime_config


def resolve_run_config(config_path: str | Path) -> dict[str, Any]:
    run_config_path = Path(config_path).resolve()
    run_config = load_yaml(run_config_path)
    run_name = run_config_path.stem

    dataset = run_config.get("dataset")
    if not dataset or not isinstance(dataset, str):
        raise ValueError("Run config must define a string 'dataset'.")

    selected_metrics = run_config.get("metrics")
    if not isinstance(selected_metrics, list) or not selected_metrics:
        raise ValueError("Run config must define a non-empty 'metrics' list.")

    dataset_config_path, dataset_config = resolve_dataset_config(dataset)
    metric_config_path, metric_file = resolve_metric_config()

    raw_runtime = run_config.get("runtime") or {}
    if not isinstance(raw_runtime, dict):
        raise ValueError("Run config field 'runtime' must be an object.")
    runtime = normalize_runtime_config(raw_runtime)

    metrics: dict[str, dict[str, Any]] = {}
    for metric_name in selected_metrics:
        if not isinstance(metric_name, str):
            raise ValueError("Run config field 'metrics' must contain only strings.")
        if metric_name not in metric_file:
            raise ValueError(
                f"Selected metric '{metric_name}' not found in metric config: "
                f"{metric_config_path}"
            )

        raw_metric_config = metric_file.get(metric_name)
        metric_config = dict(raw_metric_config or {})
        for key, value in runtime.items():
            metric_config.setdefault(key, value)
        metrics[metric_name] = metric_config

    dataset_name = dataset_config.get("dataset_name") or dataset
    output_dir = default_output_dir(dataset, run_name)

    return {
        "run_name": run_name,
        "dataset": dataset,
        "dataset_name": dataset_name,
        "manifest_path": dataset_config.get("manifest_path"),
        "output_dir": output_dir,
        "runtime": runtime,
        "metrics": metrics,
        "save_details": True,
        "config_path": str(run_config_path),
        "dataset_config_path": str(dataset_config_path),
        "metric_config_path": str(metric_config_path),
        "selected_metrics": list(selected_metrics),
    }
