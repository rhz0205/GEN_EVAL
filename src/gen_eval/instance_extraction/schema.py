from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EXPECTED_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)

@dataclass
class DetectionBox:
    frame_index: int
    bbox: list[float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "bbox": [float(value) for value in self.bbox],
            "confidence": float(self.confidence),
        }

@dataclass
class TrackDetection:
    frame_index: int
    bbox: list[float]
    confidence: float
    category: str
    class_scores: dict[str, float] = field(default_factory=dict)
    embedding: list[float] | None = None
    identity: str | None = None

@dataclass
class InstanceTrack:
    object_id: str
    category: str
    boxes: list[DetectionBox] = field(default_factory=list)
    features: list[list[float]] = field(default_factory=list)
    class_scores: list[dict[str, float]] = field(default_factory=list)
    identities: list[str] = field(default_factory=list)

    def add_detection(self, detection: TrackDetection) -> None:
        self.boxes.append(
            DetectionBox(
                frame_index=int(detection.frame_index),
                bbox=[float(value) for value in detection.bbox],
                confidence=float(detection.confidence),
            )
        )
        if detection.embedding is not None:
            self.features.append([float(value) for value in detection.embedding])
        if detection.class_scores:
            self.class_scores.append(
                {
                    str(key): float(value)
                    for key, value in detection.class_scores.items()
                }
            )
        if detection.identity:
            self.identities.append(str(detection.identity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "boxes": [item.to_dict() for item in self.boxes],
            "features": self.features,
            "class_scores": self.class_scores,
            "identities": self.identities,
        }

@dataclass
class ViewExtractionResult:
    view_name: str
    status: str
    tracks: list[InstanceTrack] = field(default_factory=list)
    reason: str | None = None

    def to_metadata_value(self) -> list[dict[str, Any]]:
        return [track.to_dict() for track in self.tracks]

@dataclass
class SampleExtractionResult:
    sample_id: str
    status: str
    instance_tracks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reason: str | None = None
    view_results: list[ViewExtractionResult] = field(default_factory=list)
