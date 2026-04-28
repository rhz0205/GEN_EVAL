from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gen_eval.config import load_yaml
from gen_eval.dataset import load_manifest
from gen_eval.metrics import register_builtin_metrics
from gen_eval.registry import registry
from gen_eval.result_writer import write_results


class Evaluator:
    """Orchestrate manifest loading and metric execution."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        register_builtin_metrics()

    @classmethod
    def from_config_path(cls, path: str | Path) -> "Evaluator":
        config_path = Path(path)
        return cls(_load_config(config_path))

    def run(self) -> dict[str, Any]:
        """Run all enabled metrics against the configured manifest."""
        manifest_path = self.config["manifest_path"]
        samples = load_manifest(manifest_path)

        results = []
        for metric_name, metric_config in iter_enabled_metrics(self.config.get("metrics", {})):
            results.append(run_metric(metric_name, metric_config, samples))

        payload = {
            "manifest_path": str(manifest_path),
            "num_samples": len(samples),
            "results": results,
        }

        output_path = self.config.get("output_path")
        if output_path:
            write_results(output_path, payload)
        return payload


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return load_yaml(path)


def iter_enabled_metrics(metrics_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return enabled metric names and normalized metric configs."""
    enabled = []
    for metric_name, raw_config in metrics_config.items():
        metric_config = raw_config or {}
        if metric_config.get("enabled", False):
            enabled.append((metric_name, metric_config))
    return enabled


def run_metric(
    metric_name: str,
    metric_config: dict[str, Any],
    samples: list[Any],
) -> dict[str, Any]:
    metric_class = registry.get(metric_name)
    metric = metric_class(metric_config)
    result = metric.evaluate(samples)
    return normalize_metric_result(metric_name, result)


def normalize_metric_result(metric_name: str, result: Any) -> dict[str, Any]:
    """Normalize metric outputs to the shared evaluator result contract."""
    if not isinstance(result, dict):
        raise TypeError(f"Metric '{metric_name}' must return a dict, got {type(result).__name__}.")

    normalized = dict(result)
    normalized.setdefault("metric", metric_name)
    normalized.setdefault("score", None)
    normalized.setdefault("num_samples", 0)

    details = normalized.get("details")
    normalized["details"] = details if isinstance(details, dict) else {}

    status = normalized.get("status")
    normalized["status"] = str(status) if status is not None else "failed"
    return normalized
