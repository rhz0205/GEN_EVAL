from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .schema import (
    EXPECTED_CAMERA_VIEWS,
    InstanceTrack,
    SampleExtractionResult,
    TrackDetection,
    ViewExtractionResult,
)


class DetectionBackend(Protocol):
    def is_available(self) -> tuple[bool, str | None]: ...

    def detect(self, frame_rgb: Any, frame_index: int) -> list[TrackDetection]: ...


class EmbeddingBackend(Protocol):
    def is_available(self) -> tuple[bool, str | None]: ...

    def embed_crops(self, crops: list[Any]) -> list[list[float]]: ...


@dataclass
class UnavailableDetectionBackend:
    reason: str = (
        "No concrete YOLO11 detector backend is configured. "
        "Plug in a local detector adapter later."
    )

    def is_available(self) -> tuple[bool, str | None]:
        return False, self.reason

    def detect(self, frame_rgb: Any, frame_index: int) -> list[TrackDetection]:
        raise RuntimeError(self.reason)


@dataclass
class UnavailableEmbeddingBackend:
    reason: str = (
        "No concrete ReID embedding backend is configured. "
        "Plug in a local embedding adapter later."
    )

    def is_available(self) -> tuple[bool, str | None]:
        return False, self.reason

    def embed_crops(self, crops: list[Any]) -> list[list[float]]:
        raise RuntimeError(self.reason)


@dataclass
class _TrackState:
    track: InstanceTrack
    last_frame_index: int
    last_bbox: list[float]
    last_embedding: list[float] | None = None
    missed_frames: int = 0


@dataclass
class InstanceTrackExtractor:
    detector: DetectionBackend = field(default_factory=UnavailableDetectionBackend)
    embedder: EmbeddingBackend = field(default_factory=UnavailableEmbeddingBackend)
    camera_videos_key: str = "camera_videos"
    detection_conf_threshold: float = 0.25
    iou_match_threshold: float = 0.30
    reid_match_threshold: float = 0.70
    max_missing_frames: int = 3
    batch_size: int = 16

    def extract_sample(
        self,
        sample_id: str,
        metadata: dict[str, Any] | None,
    ) -> SampleExtractionResult:
        metadata = metadata or {}
        camera_videos = metadata.get(self.camera_videos_key)
        if not isinstance(camera_videos, dict) or not camera_videos:
            return SampleExtractionResult(
                sample_id=sample_id,
                status="skipped",
                reason="metadata['camera_videos'] is required and must be a non-empty dict.",
            )

        detector_ok, detector_reason = self.detector.is_available()
        if not detector_ok:
            return SampleExtractionResult(
                sample_id=sample_id,
                status="skipped",
                reason=detector_reason or "Detector backend is unavailable.",
            )

        embedder_ok, embedder_reason = self.embedder.is_available()
        if not embedder_ok:
            return SampleExtractionResult(
                sample_id=sample_id,
                status="skipped",
                reason=embedder_reason or "Embedding backend is unavailable.",
            )

        instance_tracks: dict[str, list[dict[str, Any]]] = {}
        view_results: list[ViewExtractionResult] = []

        for view_name in EXPECTED_CAMERA_VIEWS:
            video_path = camera_videos.get(view_name)
            if video_path is None:
                view_results.append(
                    ViewExtractionResult(
                        view_name=view_name,
                        status="skipped",
                        reason="missing from camera_videos",
                    )
                )
                continue

            view_result = self._extract_view_tracks(view_name, str(video_path))
            view_results.append(view_result)
            if view_result.status == "success":
                instance_tracks[view_name] = view_result.to_metadata_value()

        if instance_tracks:
            return SampleExtractionResult(
                sample_id=sample_id,
                status="success",
                instance_tracks=instance_tracks,
                view_results=view_results,
            )

        reason = first_reason(view_results) or "No view produced instance tracks."
        return SampleExtractionResult(
            sample_id=sample_id,
            status="skipped",
            reason=reason,
            view_results=view_results,
        )

    def _extract_view_tracks(
        self,
        view_name: str,
        video_path: str,
    ) -> ViewExtractionResult:
        path = Path(video_path)
        if not path.exists():
            return ViewExtractionResult(
                view_name=view_name,
                status="skipped",
                reason="video path does not exist",
            )
        if not path.is_file():
            return ViewExtractionResult(
                view_name=view_name,
                status="skipped",
                reason="video path is not a file",
            )

        frames = self._read_all_frames(video_path)
        if not frames:
            return ViewExtractionResult(
                view_name=view_name,
                status="skipped",
                reason="no readable frames",
            )

        active_tracks: list[_TrackState] = []
        completed_tracks: list[InstanceTrack] = []
        next_track_index = 0

        for frame_index, frame_rgb in enumerate(frames):
            detections = [
                detection
                for detection in self.detector.detect(frame_rgb, frame_index)
                if detection.confidence >= self.detection_conf_threshold
            ]
            if detections:
                crops = [crop_bbox(frame_rgb, detection.bbox) for detection in detections]
                embeddings = self.embedder.embed_crops(crops)
                for detection, embedding in zip(detections, embeddings):
                    detection.embedding = [float(value) for value in embedding]

            matched_pairs, unmatched_track_indices, unmatched_detection_indices = (
                associate_detections_to_tracks(
                    active_tracks,
                    detections,
                    iou_match_threshold=self.iou_match_threshold,
                    reid_match_threshold=self.reid_match_threshold,
                )
            )

            for track_index, detection_index in matched_pairs:
                detection = detections[detection_index]
                track_state = active_tracks[track_index]
                detection.identity = track_state.track.object_id
                track_state.track.add_detection(detection)
                track_state.last_frame_index = frame_index
                track_state.last_bbox = list(detection.bbox)
                track_state.last_embedding = detection.embedding
                track_state.missed_frames = 0

            for detection_index in unmatched_detection_indices:
                detection = detections[detection_index]
                object_id = f"{view_name}_track_{next_track_index:04d}"
                next_track_index += 1
                detection.identity = object_id
                track = InstanceTrack(object_id=object_id, category=detection.category)
                track.add_detection(detection)
                active_tracks.append(
                    _TrackState(
                        track=track,
                        last_frame_index=frame_index,
                        last_bbox=list(detection.bbox),
                        last_embedding=detection.embedding,
                    )
                )

            completed_indices: list[int] = []
            for track_index in unmatched_track_indices:
                if track_index >= len(active_tracks):
                    continue
                active_tracks[track_index].missed_frames += 1
                if active_tracks[track_index].missed_frames > self.max_missing_frames:
                    completed_tracks.append(active_tracks[track_index].track)
                    completed_indices.append(track_index)

            if completed_indices:
                active_tracks = [
                    track_state
                    for index, track_state in enumerate(active_tracks)
                    if index not in set(completed_indices)
                ]

        completed_tracks.extend(track_state.track for track_state in active_tracks)
        completed_tracks = [track for track in completed_tracks if track.boxes]

        if not completed_tracks:
            return ViewExtractionResult(
                view_name=view_name,
                status="skipped",
                reason="no trackable detections were produced",
            )

        return ViewExtractionResult(
            view_name=view_name,
            status="success",
            tracks=completed_tracks,
        )

    def _read_all_frames(self, video_path: str) -> list[Any]:
        try:
            cv2 = _get_cv2()
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                return []

            frames: list[Any] = []
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            cap.release()
            return frames
        except Exception:
            return []


def associate_detections_to_tracks(
    active_tracks: list[_TrackState],
    detections: list[TrackDetection],
    *,
    iou_match_threshold: float,
    reid_match_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not active_tracks or not detections:
        return [], list(range(len(active_tracks))), list(range(len(detections)))

    candidate_pairs: list[tuple[float, int, int]] = []
    for track_index, track_state in enumerate(active_tracks):
        for detection_index, detection in enumerate(detections):
            iou = compute_iou(track_state.last_bbox, detection.bbox)
            if iou < iou_match_threshold:
                continue
            reid_score = cosine_similarity(track_state.last_embedding, detection.embedding)
            if reid_score is None or reid_score < reid_match_threshold:
                continue
            combined_score = 0.5 * iou + 0.5 * reid_score
            candidate_pairs.append((combined_score, track_index, detection_index))

    candidate_pairs.sort(reverse=True, key=lambda item: item[0])
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: list[tuple[int, int]] = []

    for _, track_index, detection_index in candidate_pairs:
        if track_index in used_tracks or detection_index in used_detections:
            continue
        used_tracks.add(track_index)
        used_detections.add(detection_index)
        matches.append((track_index, detection_index))

    unmatched_tracks = [
        index for index in range(len(active_tracks)) if index not in used_tracks
    ]
    unmatched_detections = [
        index for index in range(len(detections)) if index not in used_detections
    ]
    return matches, unmatched_tracks, unmatched_detections


def compute_iou(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0

    left_x1, left_y1, left_x2, left_y2 = [float(value) for value in left]
    right_x1, right_y1, right_x2, right_y2 = [float(value) for value in right]

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union_area = left_area + right_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def cosine_similarity(
    left: list[float] | None,
    right: list[float] | None,
) -> float | None:
    if left is None or right is None:
        return None
    if len(left) != len(right) or not left:
        return None

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        left_f = float(left_value)
        right_f = float(right_value)
        dot += left_f * right_f
        left_norm += left_f * left_f
        right_norm += right_f * right_f
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return dot / math.sqrt(left_norm * right_norm)


def crop_bbox(frame_rgb: Any, bbox: list[float]) -> Any:
    height = int(frame_rgb.shape[0])
    width = int(frame_rgb.shape[1])
    x1 = max(0, min(width, int(math.floor(float(bbox[0])))))
    y1 = max(0, min(height, int(math.floor(float(bbox[1])))))
    x2 = max(0, min(width, int(math.ceil(float(bbox[2])))))
    y2 = max(0, min(height, int(math.ceil(float(bbox[3])))))
    if x2 <= x1 or y2 <= y1:
        return frame_rgb[0:1, 0:1]
    return frame_rgb[y1:y2, x1:x2]


def first_reason(view_results: list[ViewExtractionResult]) -> str | None:
    for item in view_results:
        if item.reason:
            return item.reason
    return None


def _get_cv2() -> Any:
    import cv2  # type: ignore

    return cv2
