from __future__ import annotations

import importlib
from typing import Any

from modules.base import BaseModule

_MODULE_SPECS: dict[str, tuple[str, str]] = {
    "video_integrity": ("modules.video_integrity", "VideoIntegrity"),
    "temporal_consistency": ("modules.temporal_consistency", "TemporalConsistency"),
    "instance_coherence": ("modules.instance_coherence", "InstanceCoherence"),
    "depth_consistency": ("modules.depth_consistency", "DepthConsistency"),
    "semantic_consistency": ("modules.semantic_consistency", "SemanticConsistency"),
    "instance_consistency": ("modules.instance_consistency", "InstanceConsistency"),
    "view_consistency": ("modules.view_consistency", "ViewConsistency"),
}

MODULE_REGISTRY: dict[str, tuple[str, str]] = dict(_MODULE_SPECS)


def _load_class(module_name: str, class_name: str) -> type[BaseModule]:
    module = importlib.import_module(module_name)
    loaded_class = getattr(module, class_name)
    return loaded_class


def build_module(name: str, config: dict[str, Any] | None = None) -> BaseModule:
    spec = MODULE_REGISTRY.get(name)
    if spec is None:
        expected = ", ".join(sorted(MODULE_REGISTRY))
        raise ValueError(f"Unknown module '{name}'. Expected one of: {expected}.")
    module_name, class_name = spec
    module_class = _load_class(module_name, class_name)
    return module_class(config=config)


def __getattr__(name: str) -> Any:
    class_names = {class_name: (module_name, class_name) for module_name, class_name in MODULE_REGISTRY.values()}
    if name in class_names:
        module_name, class_name = class_names[name]
        return _load_class(module_name, class_name)
    raise AttributeError(f"module 'modules' has no attribute '{name}'")


__all__ = [
    "BaseModule",
    "VideoIntegrity",
    "TemporalConsistency",
    "InstanceCoherence",
    "DepthConsistency",
    "SemanticConsistency",
    "InstanceConsistency",
    "ViewConsistency",
    "MODULE_REGISTRY",
    "build_module",
]
