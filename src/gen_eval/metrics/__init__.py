from __future__ import annotations

from typing import Any

from gen_eval.registry import registry


class PlaceholderMetric:
    name = "placeholder_metric"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

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
    from .depth_consistency import DepthConsistency
    from .instance_consistency import InstanceConsistency
    from .semantic_consistency import SemanticConsistency
    from .temporal_consistency import TemporalConsistency
    from .view_consistency import ViewConsistency
    from .appearance_consistency import AppearanceConsistency

    fvd_metric = None
    try:
        from .fvd import FVDMetric

        fvd_metric = FVDMetric
    except ModuleNotFoundError:
        fvd_metric = None

    metric_classes = [
        ViewConsistency,
        TemporalConsistency,
        AppearanceConsistency,
        SemanticConsistency,
        DepthConsistency,
        InstanceConsistency,
    ]
    if fvd_metric is not None:
        metric_classes.insert(0, fvd_metric)

    for metric_class in metric_classes:
        registry.register(metric_class)


__all__ = [
    "PlaceholderMetric",
    "register_builtin_metrics",
]
