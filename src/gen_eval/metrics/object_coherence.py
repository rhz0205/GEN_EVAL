from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from gen_eval.schemas import GenerationSample, ObjectTrack


class ObjectCoherenceMetric:
    name = "object_coherence"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.objects_key = str(config.get("objects_key", "objects"))
        self.object_tracks_key = str(config.get("object_tracks_key", "object_tracks"))
        self.object_crops_key = str(config.get("object_crops_key", "object_crops"))
        self.object_features_key = str(config.get("object_features_key", "object_features"))
        self.object_class_scores_key = str(
            config.get("object_class_scores_key", "object_class_scores")
        )
        self.object_identities_key = str(
            config.get("object_identities_key", "object_identities")
        )
        self.feature_weight = float(config.get("feature_weight", 0.35))
        self.class_weight = float(config.get("class_weight", 0.2))
        self.confidence_weight = float(config.get("confidence_weight", 0.15))
        self.geometry_weight = float(config.get("geometry_weight", 0.3))

    def evaluate(self, samples: list[GenerationSample]) -> dict[str, Any]:
        evaluated_samples = []
        skipped_samples = []
        failed_samples = []
        total_score = 0.0
        total_feature = 0.0
        total_class = 0.0
        total_confidence = 0.0
        total_geometry = 0.0

        for sample in samples:
            try:
                tracks = self._collect_tracks(sample)
                if not tracks:
                    raise _SkipSample(
                        "No usable object annotations were found. "
                        "Provide sample.objects or object metadata keys."
                    )
                sample_result = self._score_sample(sample, tracks)
            except _SkipSample as exc:
                skipped_samples.append({"sample_id": sample.sample_id, "reason": str(exc)})
                continue
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                failed_samples.append({"sample_id": sample.sample_id, "reason": str(exc)})
                continue

            evaluated_samples.append(sample_result)
            total_score += sample_result["score"]
            total_feature += sample_result["feature_consistency"]
            total_class += sample_result["class_stability"]
            total_confidence += sample_result["confidence_stability"]
            total_geometry += sample_result["geometry_coherence"]

        if not evaluated_samples:
            return self._result(
                score=None,
                num_samples=0,
                details={
                    "evaluated_samples": [],
                    "skipped_samples": skipped_samples,
                    "failed_samples": failed_samples,
                },
                status="failed" if failed_samples else "skipped",
                reason=(
                    "No usable object tracks, features, class scores, or boxes were available for evaluation."
                ),
            )

        num_samples = len(evaluated_samples)
        averages = {
            "object_coherence": total_score / num_samples,
            "feature_consistency": total_feature / num_samples,
            "class_stability": total_class / num_samples,
            "confidence_stability": total_confidence / num_samples,
            "geometry_coherence": total_geometry / num_samples,
        }
        return self._result(
            score=averages["object_coherence"],
            num_samples=num_samples,
            details={
                "average_results": averages,
                "evaluated_samples": evaluated_samples,
                "skipped_samples": skipped_samples,
                "failed_samples": failed_samples,
            },
            status="ok",
            reason=None,
        )

    def _collect_tracks(self, sample: GenerationSample) -> list[dict[str, Any]]:
        metadata = sample.metadata or {}
        tracks: list[dict[str, Any]] = []

        if sample.objects:
            for obj in sample.objects:
                tracks.append(self._track_from_objecttrack(obj))

        for key in (self.objects_key, self.object_tracks_key):
            value = metadata.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        tracks.append(self._normalize_track_dict(item))

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

    def _merge_track_mapping(
        self, tracks: list[dict[str, Any]], mapping: dict[str, Any], field_name: str
    ) -> None:
        by_id = {str(track.get("object_id")): track for track in tracks if track.get("object_id")}
        for object_id, value in mapping.items():
            object_id = str(object_id)
            if object_id not in by_id:
                by_id[object_id] = {"object_id": object_id}
                tracks.append(by_id[object_id])
            by_id[object_id][field_name] = value

    def _merge_track_dicts(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        merged = dict(left)
        for key, value in right.items():
            if key not in merged or merged[key] in (None, [], {}):
                merged[key] = value
            elif key == "boxes" and isinstance(merged[key], list) and isinstance(value, list):
                merged[key] = merged[key] + value
            elif key == "attributes" and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
        return merged

    def _track_has_signal(self, track: dict[str, Any]) -> bool:
        return any(
            key in track and track[key] not in (None, [], {})
            for key in ("boxes", "features", "class_scores", "identities", "crops")
        )

    def _score_sample(self, sample: GenerationSample, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        feature_scores = []
        class_scores = []
        confidence_scores = []
        geometry_scores = []
        usable_tracks = []

        for track in tracks:
            feature_score = self._feature_consistency(track)
            class_score = self._class_stability(track)
            confidence_score = self._confidence_stability(track)
            geometry_score = self._geometry_coherence(track, sample)

            if all(score is None for score in (feature_score, class_score, confidence_score, geometry_score)):
                continue

            usable_tracks.append(
                {
                    "object_id": str(track.get("object_id", "")),
                    "category": track.get("category"),
                    "feature_consistency": feature_score,
                    "class_stability": class_score,
                    "confidence_stability": confidence_score,
                    "geometry_coherence": geometry_score,
                }
            )
            if feature_score is not None:
                feature_scores.append(feature_score)
            if class_score is not None:
                class_scores.append(class_score)
            if confidence_score is not None:
                confidence_scores.append(confidence_score)
            if geometry_score is not None:
                geometry_scores.append(geometry_score)

        if not usable_tracks:
            raise _SkipSample(
                f"Sample '{sample.sample_id}' has object entries but no usable per-track signals for scoring."
            )

        feature_avg = self._mean_or_default(feature_scores)
        class_avg = self._mean_or_default(class_scores)
        confidence_avg = self._mean_or_default(confidence_scores)
        geometry_avg = self._mean_or_default(geometry_scores)
        score = (
            self.feature_weight * feature_avg
            + self.class_weight * class_avg
            + self.confidence_weight * confidence_avg
            + self.geometry_weight * geometry_avg
        )
        return {
            "sample_id": sample.sample_id,
            "score": score,
            "feature_consistency": feature_avg,
            "class_stability": class_avg,
            "confidence_stability": confidence_avg,
            "geometry_coherence": geometry_avg,
            "num_tracks": len(usable_tracks),
            "tracks": usable_tracks,
        }

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

    def _geometry_coherence(self, track: dict[str, Any], sample: GenerationSample) -> float | None:
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
        motion_smoothness = 1.0 / (1.0 + float(displacements.std())) if displacements.size > 0 else 1.0

        area_log = np.log(np.maximum(np.asarray(areas, dtype=np.float64), 1e-8))
        area_stability = 1.0 / (1.0 + float(area_log.std()))

        ratio_log = np.log(np.maximum(np.asarray(aspect_ratios, dtype=np.float64), 1e-8))
        ratio_stability = 1.0 / (1.0 + float(ratio_log.std()))

        sorted_frames = sorted(frame_indices)
        observed_span = max(sorted_frames) - min(sorted_frames) + 1
        missing_ratio = 0.0 if observed_span <= 0 else 1.0 - (len(sorted_frames) / observed_span)

        metadata = sample.metadata or {}
        total_frames = int(metadata.get("num_frames", observed_span or len(sorted_frames) or 1))
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

    def _mean_or_default(self, values: list[float], default: float = 0.0) -> float:
        if not values:
            return default
        return float(sum(values) / len(values))

    def _result(
        self,
        *,
        score: float | None,
        num_samples: int,
        details: dict[str, Any],
        status: str,
        reason: str | None,
    ) -> dict[str, Any]:
        result = {
            "metric": self.name,
            "score": score,
            "num_samples": num_samples,
            "details": details,
            "status": status,
        }
        if reason is not None:
            result["reason"] = reason
        return result


ObjectCoherence = ObjectCoherenceMetric


class _SkipSample(Exception):
    pass
