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


from .cross_view_consistency import CrossViewConsistency
from .depth_consistency import DepthConsistency
from .fvd import FVDMetric
from .object_coherence import ObjectCoherence
from .subject_consistency import SubjectConsistency
from .temporal_consistency import TemporalConsistency
from .temporal_semantic_consistency import TemporalSemanticConsistency


def register_builtin_metrics() -> None:
    for metric_class in (
        FVDMetric,
        TemporalConsistency,
        SubjectConsistency,
        TemporalSemanticConsistency,
        DepthConsistency,
        ObjectCoherence,
        CrossViewConsistency,
    ):
        registry.register(metric_class)


__all__ = [
    "PlaceholderMetric",
    "register_builtin_metrics",
    "FVDMetric",
    "TemporalConsistency",
    "SubjectConsistency",
    "TemporalSemanticConsistency",
    "DepthConsistency",
    "ObjectCoherence",
    "CrossViewConsistency",
]
