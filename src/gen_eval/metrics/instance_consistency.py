from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from gen_eval.schemas import GenerationSample, ObjectTrack

EXPECTED_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)

class InstanceConsistencyMetric:

    name = "instance_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.instance_tracks_key = str(
            self.config.get("instance_tracks_key", "instance_tracks")
        )
        self.object_tracks_key = str(
            self.config.get("object_tracks_key", "object_tracks")
        )
        self.objects_key = str(self.config.get("objects_key", "objects"))
        self.object_crops_key = str(
            self.config.get("object_crops_key", "object_crops")
        )
        self.object_features_key = str(
            self.config.get("object_features_key", "object_features")
        )
        self.object_class_scores_key = str(
            self.config.get("object_class_scores_key", "object_class_scores")
        )
        self.object_identities_key = str(
            self.config.get("object_identities_key", "object_identities")
        )

        self.feature_weight = 0.35
        self.class_weight = 0.20
        self.confidence_weight = 0.15
        self.geometry_weight = 0.30

    def evaluate(self, samples: list[GenerationSample]) -> dict[str, Any]:
        numpy_status = self._ensure_numpy()
        if numpy_status is not None:
            return {
                "metric": self.name,
                "status": "skipped",
                "num_samples": len(samples),
                "valid_sample_count": 0,
                "mean_instance_consistency_score": None,
                "details": {
                    "evaluated_samples": [],
                    "skipped_samples": [
                        {
                            "sample_id": getattr(sample, "sample_id", "unknown"),
                            "reason": numpy_status,
                        }
                        for sample in samples
                    ],
                    "failed_samples": [],
                },
                "reason": numpy_status,
            }

        evaluated_samples: list[dict[str, Any]] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        valid_scores: list[float] = []

        for sample in samples:
            sample_id = getattr(sample, "sample_id", None) or "unknown"
            try:
                sample_result = self._evaluate_sample(sample)
            except _SkipSample as exc:
                sample_result = {
                    "sample_id": sample_id,
                    "status": "skipped",
                    "reason": str(exc),
                }
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                sample_result = {
                    "sample_id": sample_id,
                    "status": "failed",
                    "reason": str(exc),
                }

            status = sample_result.get("status")
            score = sample_result.get("instance_consistency_score")
            if status == "success" and is_finite_number(score):
                evaluated_samples.append(
                    {
                        "sample_id": sample_id,
                        "instance_consistency_score": float(score),
                    }
                )
                valid_scores.append(float(score))
            elif status == "skipped":
                skipped_samples.append(
                    {
                        "sample_id": sample_id,
                        "reason": sample_result.get("reason", "unknown"),
                    }
                )
            elif status == "failed":
                failed_samples.append(
                    {
                        "sample_id": sample_id,
                        "reason": sample_result.get("reason", "unknown"),
                    }
                )

        mean_score = mean_or_none(valid_scores)
        if mean_score is not None:
            status = "success"
            reason = None
        else:
            status = "failed" if failed_samples else "skipped"
            reason = "No usable prepared tracks were evaluable for the fixed expected views."

        result: dict[str, Any] = {
            "metric": self.name,
            "status": status,
            "num_samples": len(samples),
            "valid_sample_count": len(valid_scores),
            "mean_instance_consistency_score": mean_score,
            "details": {
                "evaluated_samples": evaluated_samples,
                "skipped_samples": skipped_samples,
                "failed_samples": failed_samples,
            },
        }
        if reason is not None:
            result["reason"] = reason
        return result

    def _evaluate_sample(self, sample: GenerationSample) -> dict[str, Any]:
        metadata = sample.metadata or {}
        view_scores: list[float] = []

        for view in EXPECTED_CAMERA_VIEWS:
            tracks = self._collect_view_tracks(metadata, view)
            if not tracks:
                continue
            view_score = self._score_view_tracks(sample, tracks)
            if is_finite_number(view_score):
                view_scores.append(float(view_score))

        if not view_scores:
            tracks = self._collect_legacy_tracks(sample)
            if not tracks:
                raise _SkipSample(
                    "No usable prepared tracks found for expected views or legacy fallback."
                )
            view_score = self._score_view_tracks(sample, tracks)
            if is_finite_number(view_score):
                view_scores.append(float(view_score))

        if not view_scores:
            raise _SkipSample("No usable track-level signals were available for scoring.")

        return {
            "sample_id": sample.sample_id,
            "status": "success",
            "instance_consistency_score": mean_or_none(view_scores),
        }

    def _collect_view_tracks(
        self,
        metadata: dict[str, Any],
        view: str,
    ) -> list[dict[str, Any]]:
        tracks: list[dict[str, Any]] = []

        primary = metadata.get(self.instance_tracks_key)
        if isinstance(primary, dict):
            tracks.extend(self._normalize_track_items(primary.get(view)))

        secondary = metadata.get(self.object_tracks_key)
        if isinstance(secondary, dict):
            tracks.extend(self._normalize_track_items(secondary.get(view)))

        objects = metadata.get(self.objects_key)
        if isinstance(objects, dict):
            tracks.extend(self._normalize_track_items(objects.get(view)))

        self._merge_view_mapping(
            tracks,
            metadata.get(self.object_features_key),
            view,
            "features",
        )
        self._merge_view_mapping(
            tracks,
            metadata.get(self.object_class_scores_key),
            view,
            "class_scores",
        )
        self._merge_view_mapping(
            tracks,
            metadata.get(self.object_identities_key),
            view,
            "identities",
        )
        self._merge_view_mapping(
            tracks,
            metadata.get(self.object_crops_key),
            view,
            "crops",
        )

        return self._dedupe_tracks(tracks)

    def _collect_legacy_tracks(self, sample: GenerationSample) -> list[dict[str, Any]]:
        metadata = sample.metadata or {}
        tracks: list[dict[str, Any]] = []

        if sample.objects:
            for obj in sample.objects:
                tracks.append(self._track_from_objecttrack(obj))

        for key in (self.instance_tracks_key, self.object_tracks_key, self.objects_key):
            value = metadata.get(key)
            if isinstance(value, list):
                tracks.extend(self._normalize_track_items(value))

        features_value = metadata.get(self.object_features_key)
        if isinstance(features_value, dict):
            self._merge_track_mapping(tracks, features_value, "features")

        class_scores_value = metadata.get(self.object_class_scores_key)
        if isinstance(class_scores_value, dict):
            self._merge_track_mapping(tracks, class_scores_value, "class_scores")

        identities_value = metadata.get(self.object_identities_key)
        if isinstance(identities_value, dict):
            self._merge_track_mapping(tracks, identities_value, "identities")

        crops_value = metadata.get(self.object_crops_key)
        if isinstance(crops_value, dict):
            self._merge_track_mapping(tracks, crops_value, "crops")

        return self._dedupe_tracks(tracks)

    def _normalize_track_items(self, raw_items: Any) -> list[dict[str, Any]]:
        if raw_items is None:
            return []
        if isinstance(raw_items, list):
            normalized = []
            for item in raw_items:
                if isinstance(item, ObjectTrack):
                    normalized.append(self._track_from_objecttrack(item))
                elif isinstance(item, dict):
                    normalized.append(self._normalize_track_dict(item))
            return normalized
        return []

    def _track_from_objecttrack(self, obj: ObjectTrack) -> dict[str, Any]:
        return {
            "object_id": obj.object_id,
            "category": obj.category,
            "boxes": list(obj.boxes),
            "attributes": dict(obj.attributes),
        }

    def _normalize_track_dict(self, raw_track: dict[str, Any]) -> dict[str, Any]:
        track = dict(raw_track)
        if "boxes_2d" in track and "boxes" not in track:
            track["boxes"] = track["boxes_2d"]
        return track

    def _merge_view_mapping(
        self,
        tracks: list[dict[str, Any]],
        mapping: Any,
        view: str,
        field_name: str,
    ) -> None:
        if not isinstance(mapping, dict):
            return
        value = mapping.get(view)
        if isinstance(value, dict):
            self._merge_track_mapping(tracks, value, field_name)

    def _merge_track_mapping(
        self,
        tracks: list[dict[str, Any]],
        mapping: dict[str, Any],
        field_name: str,
    ) -> None:
        by_id = {
            str(track.get("object_id")): track
            for track in tracks
            if track.get("object_id")
        }
        for object_id, value in mapping.items():
            object_id = str(object_id)
            if object_id not in by_id:
                by_id[object_id] = {"object_id": object_id}
                tracks.append(by_id[object_id])
            by_id[object_id][field_name] = value

    def _dedupe_tracks(self, tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        anonymous_index = 0
        for track in tracks:
            track_id = str(track.get("object_id") or f"track_{anonymous_index}")
            if not track.get("object_id"):
                anonymous_index += 1
            if track_id in deduped:
                deduped[track_id] = self._merge_track_dicts(deduped[track_id], track)
            else:
                track["object_id"] = track_id
                deduped[track_id] = track
        return [track for track in deduped.values() if self._track_has_signal(track)]

    def _merge_track_dicts(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(left)
        for key, value in right.items():
            if key not in merged or merged[key] in (None, [], {}):
                merged[key] = value
            elif key == "boxes" and isinstance(merged[key], list) and isinstance(value, list):
                merged[key] = merged[key] + value
            elif (
                key == "attributes"
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = {**merged[key], **value}
        return merged

    def _track_has_signal(self, track: dict[str, Any]) -> bool:
        return any(
            key in track and track[key] not in (None, [], {})
            for key in ("boxes", "features", "class_scores", "identities", "crops")
        )

    def _score_view_tracks(
        self,
        sample: GenerationSample,
        tracks: list[dict[str, Any]],
    ) -> float | None:
        track_scores: list[float] = []
        for track in tracks:
            feature_score = self._feature_consistency(track)
            class_score = self._class_stability(track)
            confidence_score = self._confidence_stability(track)
            geometry_score = self._geometry_coherence(track, sample)

            if all(
                score is None
                for score in (
                    feature_score,
                    class_score,
                    confidence_score,
                    geometry_score,
                )
            ):
                continue

            track_scores.append(
                self.feature_weight * self._value_or_default(feature_score)
                + self.class_weight * self._value_or_default(class_score)
                + self.confidence_weight * self._value_or_default(confidence_score)
                + self.geometry_weight * self._value_or_default(geometry_score)
            )

        return mean_or_none(track_scores)

    def _feature_consistency(self, track: dict[str, Any]) -> float | None:
        raw_features = track.get("features")
        if raw_features is None:
            return None
        features = np.asarray(raw_features, dtype=np.float64)
        if features.ndim == 1:
            return 1.0
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        if features.shape[0] < 2:
            return 1.0

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = features / norms
        cosine_values = np.sum(normalized[1:] * normalized[:-1], axis=1)
        cosine_values = np.clip(cosine_values, -1.0, 1.0)
        return float((cosine_values.mean() + 1.0) / 2.0)

    def _class_stability(self, track: dict[str, Any]) -> float | None:
        class_scores = track.get("class_scores")
        identities = track.get("identities")
        if class_scores is not None:
            return self._label_stability_from_scores(class_scores)
        if identities is not None:
            return self._label_stability_from_labels(identities)
        return None

    def _confidence_stability(self, track: dict[str, Any]) -> float | None:
        class_scores = track.get("class_scores")
        if class_scores is None:
            return None

        confidences = []
        if isinstance(class_scores, dict):
            for value in class_scores.values():
                confidences.extend(self._extract_confidences(value))
        else:
            confidences.extend(self._extract_confidences(class_scores))

        if not confidences:
            return None
        confidences_array = np.asarray(confidences, dtype=np.float64)
        if confidences_array.size == 1:
            return float(np.clip(confidences_array[0], 0.0, 1.0))

        mean_conf = float(np.clip(confidences_array.mean(), 0.0, 1.0))
        std_conf = float(confidences_array.std())
        stability = 1.0 / (1.0 + std_conf)
        return float(mean_conf * stability)

    def _geometry_coherence(
        self,
        track: dict[str, Any],
        sample: GenerationSample,
    ) -> float | None:
        boxes = self._normalize_boxes(track.get("boxes"))
        if len(boxes) < 2:
            return None

        centers = []
        areas = []
        aspect_ratios = []
        frame_indices = []
        for item in boxes:
            bbox = item["bbox"]
            x1, y1, x2, y2 = bbox
            width = max(x2 - x1, 1e-8)
            height = max(y2 - y1, 1e-8)
            centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            areas.append(width * height)
            aspect_ratios.append(width / height)
            frame_indices.append(int(item["frame_index"]))

        centers_array = np.asarray(centers, dtype=np.float64)
        displacements = np.linalg.norm(np.diff(centers_array, axis=0), axis=1)
        motion_smoothness = (
            1.0 / (1.0 + float(displacements.std())) if displacements.size > 0 else 1.0
        )

        area_log = np.log(np.maximum(np.asarray(areas, dtype=np.float64), 1e-8))
        area_stability = 1.0 / (1.0 + float(area_log.std()))

        ratio_log = np.log(np.maximum(np.asarray(aspect_ratios, dtype=np.float64), 1e-8))
        ratio_stability = 1.0 / (1.0 + float(ratio_log.std()))

        sorted_frames = sorted(frame_indices)
        observed_span = max(sorted_frames) - min(sorted_frames) + 1
        missing_ratio = (
            0.0 if observed_span <= 0 else 1.0 - (len(sorted_frames) / observed_span)
        )

        metadata = sample.metadata or {}
        total_frames = int(
            metadata.get("num_frames", observed_span or len(sorted_frames) or 1)
        )
        valid_track_ratio = len(sorted_frames) / max(total_frames, len(sorted_frames), 1)

        score = (
            0.35 * motion_smoothness
            + 0.2 * area_stability
            + 0.2 * ratio_stability
            + 0.15 * (1.0 - max(0.0, min(missing_ratio, 1.0)))
            + 0.1 * min(1.0, valid_track_ratio)
        )
        return float(score)

    def _label_stability_from_scores(self, class_scores: Any) -> float | None:
        labels = []
        if isinstance(class_scores, dict):
            for value in class_scores.values():
                label = self._extract_top_label(value)
                if label is not None:
                    labels.append(label)
        elif isinstance(class_scores, list):
            for value in class_scores:
                label = self._extract_top_label(value)
                if label is not None:
                    labels.append(label)
        else:
            label = self._extract_top_label(class_scores)
            if label is not None:
                labels.append(label)
        return self._label_stability_from_labels(labels)

    def _label_stability_from_labels(self, labels: Any) -> float | None:
        if isinstance(labels, (str, int, float)):
            return 1.0
        if not isinstance(labels, list):
            return None
        normalized = [str(label) for label in labels if label is not None]
        if not normalized:
            return None
        if len(normalized) == 1:
            return 1.0
        counts = defaultdict(int)
        for label in normalized:
            counts[label] += 1
        return float(max(counts.values()) / len(normalized))

    def _extract_top_label(self, value: Any) -> str | None:
        if isinstance(value, dict):
            if "label" in value:
                return str(value["label"])
            if "predicted_class" in value:
                return str(value["predicted_class"])
            best_label = None
            best_score = -math.inf
            for key, raw_score in value.items():
                if isinstance(raw_score, (int, float)) and raw_score > best_score:
                    best_score = float(raw_score)
                    best_label = str(key)
            return best_label
        if isinstance(value, (str, int)):
            return str(value)
        return None

    def _extract_confidences(self, value: Any) -> list[float]:
        confidences: list[float] = []
        if isinstance(value, dict):
            if "confidence" in value and isinstance(value["confidence"], (int, float)):
                confidences.append(float(value["confidence"]))
            elif "score" in value and isinstance(value["score"], (int, float)):
                confidences.append(float(value["score"]))
            else:
                numeric_values = [
                    float(raw_score)
                    for raw_score in value.values()
                    if isinstance(raw_score, (int, float))
                ]
                if numeric_values:
                    confidences.append(max(numeric_values))
        elif isinstance(value, list):
            for item in value:
                confidences.extend(self._extract_confidences(item))
        elif isinstance(value, (int, float)):
            confidences.append(float(value))
        return [float(np.clip(score, 0.0, 1.0)) for score in confidences]

    def _normalize_boxes(self, raw_boxes: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_boxes, list):
            return []
        normalized = []
        for item in raw_boxes:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox") or item.get("box") or item.get("boxes_2d")
            frame_index = item.get("frame_index")
            if bbox is None or frame_index is None:
                continue
            bbox_array = np.asarray(bbox, dtype=np.float64).reshape(-1)
            if bbox_array.size != 4:
                continue
            normalized.append(
                {
                    "frame_index": int(frame_index),
                    "bbox": bbox_array.tolist(),
                }
            )
        normalized.sort(key=lambda entry: entry["frame_index"])
        return normalized

    def _value_or_default(self, value: float | None, default: float = 0.0) -> float:
        if value is None:
            return default
        return float(value)

    def _ensure_numpy(self) -> str | None:
        try:
            global np
            import numpy as np  # type: ignore
        except Exception as exc:
            return f"Required scoring dependency is unavailable: {exc}"
        return None

InstanceConsistency = InstanceConsistencyMetric

class _SkipSample(Exception):
    pass

def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))

def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
