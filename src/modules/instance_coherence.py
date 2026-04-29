from __future__ import annotations

import math
from typing import Any

import numpy as np

from modules.base import BaseModule

EXPECTED_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)


class InstanceCoherence(BaseModule):
    name = "instance_coherence"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.object_features_key = str(self.config.get("object_features_key", "object_features"))
        self.expected_camera_views = tuple(self.config.get("expected_camera_views", EXPECTED_CAMERA_VIEWS))
        self.eps = float(self.config.get("eps", 1e-8))

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        evaluated_samples: list[dict[str, Any]] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        valid_scores: list[float] = []

        for sample in samples:
            sample_id = getattr(sample, "sample_id", None) or "unknown"
            try:
                sample_result = self._evaluate_sample(sample)
            except Exception as exc:
                sample_result = {
                    "sample_id": sample_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }

            status = sample_result.get("status")
            score = sample_result.get("instance_coherence_score")
            if status == "success" and is_finite_number(score):
                evaluated_samples.append(
                    {
                        "sample_id": sample_id,
                        "instance_coherence_score": float(score),
                    }
                )
                valid_scores.append(float(score))
            elif status == "failed":
                failed_samples.append(simplify_sample_result(sample_result))
            else:
                skipped_samples.append(simplify_sample_result(sample_result))

        mean_score = mean_or_none(valid_scores)
        if mean_score is not None:
            status = "success"
            reason = None
        else:
            status = "failed" if failed_samples else "skipped"
            reason = f"No valid instance coherence score from metadata['{self.object_features_key}']."

        result: dict[str, Any] = {
            "metric": self.name,
            "status": status,
            "num_samples": len(samples),
            "valid_sample_count": len(valid_scores),
            "skipped_sample_count": len(skipped_samples),
            "failed_sample_count": len(failed_samples),
            "mean_instance_coherence_score": mean_score,
            "details": {
                "evaluated_samples": evaluated_samples,
                "skipped_samples": skipped_samples,
                "failed_samples": failed_samples,
            },
        }
        if reason:
            result["reason"] = reason
        return result

    def _evaluate_sample(self, sample: Any) -> dict[str, Any]:
        sample_id = getattr(sample, "sample_id", None) or "unknown"
        metadata = getattr(sample, "metadata", None) or {}
        object_features = metadata.get(self.object_features_key)
        if not isinstance(object_features, dict) or not object_features:
            return skipped_result(sample_id, f"metadata['{self.object_features_key}'] must be a non-empty dict.")

        view_scores: list[float] = []
        for view in self.expected_camera_views:
            view_mapping = object_features.get(view)
            if not isinstance(view_mapping, dict) or not view_mapping:
                continue
            object_scores = self._score_view_mapping(view_mapping)
            if object_scores:
                view_scores.append(float(sum(object_scores) / len(object_scores)))

        if not view_scores:
            legacy_scores = self._score_view_mapping(object_features)
            if legacy_scores:
                view_scores.append(float(sum(legacy_scores) / len(legacy_scores)))

        if not view_scores:
            return skipped_result(sample_id, f"No usable object feature sequence found in metadata['{self.object_features_key}'].")

        return {
            "sample_id": sample_id,
            "status": "success",
            "instance_coherence_score": mean_or_none(view_scores),
        }

    def _score_view_mapping(self, view_mapping: dict[str, Any]) -> list[float]:
        scores: list[float] = []
        for raw_sequence in view_mapping.values():
            score = self._score_object_sequence(raw_sequence)
            if is_finite_number(score):
                scores.append(float(score))
        return scores

    def _score_object_sequence(self, raw_sequence: Any) -> float | None:
        features = self._normalize_feature_sequence(raw_sequence)
        if features is None or features.shape[0] < 2:
            return None
        acm = self._compute_acm(features)
        tji = self._compute_tji(features)
        return clamp01(float(acm / (1.0 + tji)))

    def _normalize_feature_sequence(self, raw_sequence: Any) -> np.ndarray | None:
        if raw_sequence is None:
            return None
        array = np.asarray(raw_sequence, dtype=np.float64)
        if array.ndim == 0:
            return None
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        if array.shape[0] < 1 or array.shape[1] < 1:
            return None
        if not np.isfinite(array).all():
            return None
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.maximum(norms, self.eps)
        return array / norms

    def _compute_acm(self, features: np.ndarray) -> float:
        cosine_values = np.sum(features[1:] * features[:-1], axis=1)
        cosine_values = np.clip(cosine_values, 0.0, 1.0)
        return float(cosine_values.mean())

    def _compute_tji(self, features: np.ndarray) -> float:
        if features.shape[0] < 3:
            return 0.0
        velocity = np.linalg.norm(features[1:] - features[:-1], axis=1)
        acceleration = np.linalg.norm(features[2:] - 2 * features[1:-1] + features[:-2], axis=1)
        denominator = 0.5 * (velocity[1:] + velocity[:-1]) + self.eps
        jitter = acceleration / denominator
        return max(0.0, float(jitter.mean()))


def skipped_result(sample_id: str, reason: str) -> dict[str, Any]:
    return {"sample_id": sample_id, "status": "skipped", "reason": reason}


def simplify_sample_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": result.get("sample_id", "unknown"),
        "reason": result.get("reason", "unknown"),
    }


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
