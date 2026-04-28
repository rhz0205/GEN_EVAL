from __future__ import annotations

from typing import Any

from visualization.base import BaseVisualizer
from visualization.depth import DepthVisualizer
from visualization.multiview import MultiViewMatchVisualizer
from visualization.semantic import SemanticVisualizer

VISUALIZER_REGISTRY: dict[str, type[BaseVisualizer]] = {
    "depth": DepthVisualizer,
    "semantic": SemanticVisualizer,
    "multiview": MultiViewMatchVisualizer,
}


def build_visualizer(name: str, config: dict[str, Any] | None = None) -> BaseVisualizer:
    visualizer_class = VISUALIZER_REGISTRY.get(name)
    if visualizer_class is None:
        expected = ", ".join(sorted(VISUALIZER_REGISTRY))
        raise ValueError(f"Unknown visualizer '{name}'. Expected one of: {expected}.")
    return visualizer_class(config=config)


__all__ = [
    "BaseVisualizer",
    "DepthVisualizer",
    "SemanticVisualizer",
    "MultiViewMatchVisualizer",
    "VISUALIZER_REGISTRY",
    "build_visualizer",
]
