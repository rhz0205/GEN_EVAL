from __future__ import annotations

from typing import Any

from gen_eval.registry import registry

class PlaceholderMetric:
    name = "placeholder_metric"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        if not samples:
            return {
                "metric": self.name,
                "score": None,
                "num_samples": 0,
                "details": {},
                "status": "skipped",
                "reason": "No samples were provided.",
            }
        return {
            "metric": self.name,
            "score": None,
            "num_samples": len(samples),
            "details": {
                "implemented": False,
                "config": self.config,
            },
            "status": "skipped",
            "reason": "Placeholder metric. Implementation requires model/runtime integration.",
        }

def register_builtin_metrics() -> None:
    from .appearance_consistency import AppearanceConsistencyMetric
    from .depth_consistency import DepthConsistencyMetric
    from .instance_consistency import InstanceConsistencyMetric
    from .semantic_consistency import SemanticConsistencyMetric
    from .temporal_consistency import TemporalConsistencyMetric
    from .video_integrity import VideoIntegrityMetric
    from .view_consistency import ViewConsistencyMetric

    metric_classes = [
        VideoIntegrityMetric,
        ViewConsistencyMetric,
        TemporalConsistencyMetric,
        AppearanceConsistencyMetric,
        SemanticConsistencyMetric,
        DepthConsistencyMetric,
        InstanceConsistencyMetric,
    ]

    for metric_class in metric_classes:
        registry.register(metric_class)

__all__ = [
    "PlaceholderMetric",
    "register_builtin_metrics",
]
