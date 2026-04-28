from __future__ import annotations

from typing import Any

from reference.base import ReferenceGenerator
from reference.preparer import (
    REFERENCE_ALIASES,
    REFERENCE_REGISTRY,
    ReferencePreparer,
    build_reference_generator,
)

_CLASS_NAMES = {
    "OpenSeeDReference": "openseed_semantic",
    "DepthReference": "depth_reference",
    "ObjectTrackReference": "object_tracks",
    "PlanningResponseReference": "planning_response",
}


def __getattr__(name: str) -> Any:
    generator_name = _CLASS_NAMES.get(name)
    if generator_name is None:
        raise AttributeError(f"module 'reference' has no attribute '{name}'")
    generator = build_reference_generator(generator_name)
    return generator.__class__


__all__ = [
    "ReferenceGenerator",
    "ReferencePreparer",
    "OpenSeeDReference",
    "DepthReference",
    "ObjectTrackReference",
    "PlanningResponseReference",
    "REFERENCE_REGISTRY",
    "REFERENCE_ALIASES",
    "build_reference_generator",
]
