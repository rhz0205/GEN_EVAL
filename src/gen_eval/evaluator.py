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
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        register_builtin_metrics()

    @classmethod
    def from_config_path(cls, path: str | Path) -> "Evaluator":
        config_path = Path(path)
        return cls(_load_config(config_path))

    def run(self) -> dict[str, Any]:
        manifest_path = self.config["manifest_path"]
        samples = load_manifest(manifest_path)

        results = []
        for metric_name, metric_config in _iter_enabled_metrics(self.config.get("metrics", {})):
            metric_class = registry.get(metric_name)
            metric = metric_class(metric_config)
            results.append(metric.evaluate(samples))

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


def _iter_enabled_metrics(metrics_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    enabled = []
    for metric_name, raw_config in metrics_config.items():
        metric_config = raw_config or {}
        if metric_config.get("enabled", False):
            enabled.append((metric_name, metric_config))
    return enabled
