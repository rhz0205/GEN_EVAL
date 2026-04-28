from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ObjectTrack:

    object_id: str
    category: str
    boxes: list[dict[str, Any]] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectTrack":
        return cls(
            object_id=str(data.get("object_id", "")),
            category=str(data.get("category", "")),
            boxes=list(data.get("boxes", [])),
            attributes=dict(data.get("attributes", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "boxes": self.boxes,
            "attributes": self.attributes,
        }

@dataclass
class GenerationSample:

    sample_id: str
    generated_video: str
    reference_video: str | None = None
    prompt: str = ""
    objects: list[ObjectTrack] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationSample":
        object_items = data.get("objects", [])
        return cls(
            sample_id=str(data["sample_id"]),
            generated_video=str(data["generated_video"]),
            reference_video=data.get("reference_video"),
            prompt=str(data.get("prompt", "")),
            objects=[ObjectTrack.from_dict(item) for item in object_items],
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "generated_video": self.generated_video,
            "reference_video": self.reference_video,
            "prompt": self.prompt,
            "objects": [item.to_dict() for item in self.objects],
            "metadata": self.metadata,
        }
