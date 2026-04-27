from __future__ import annotations

import math
from typing import Any

import numpy as np

from gen_eval.schemas import GenerationSample


class CrossViewConsistencyMetric:
    name = "cross_view_consistency"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.camera_videos_key = str(config.get("camera_videos_key", "camera_videos"))
        self.multi_view_videos_key = str(config.get("multi_view_videos_key", "multi_view_videos"))
        self.views_key = str(config.get("views_key", "views"))
        self.camera_pairs_key = str(config.get("camera_pairs_key", "camera_pairs"))
        self.cross_view_matches_key = str(config.get("cross_view_matches_key", "cross_view_matches"))
        self.cross_view_scores_key = str(config.get("cross_view_scores_key", "cross_view_scores"))
        self.cross_view_confidence_key = str(
            config.get("cross_view_confidence_key", "cross_view_confidence")
        )
        self.cross_view_features_key = str(
            config.get("cross_view_features_key", "cross_view_features")
        )
        self.score_weight = float(config.get("score_weight", 0.45))
        self.match_weight = float(config.get("match_weight", 0.3))
        self.feature_weight = float(config.get("feature_weight", 0.25))

    def evaluate(self, samples: list[GenerationSample]) -> dict[str, Any]:
        evaluated_samples = []
        skipped_samples = []
        failed_samples = []
        total_score = 0.0
        total_score_signal = 0.0
        total_match_signal = 0.0
        total_feature_signal = 0.0

        for sample in samples:
            try:
                sample_result = self._score_sample(sample)
            except _SkipSample as exc:
                skipped_samples.append({"sample_id": sample.sample_id, "reason": str(exc)})
                continue
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                failed_samples.append({"sample_id": sample.sample_id, "reason": str(exc)})
                continue

            evaluated_samples.append(sample_result)
            total_score += sample_result["score"]
            total_score_signal += sample_result["score_signal"]
            total_match_signal += sample_result["match_signal"]
            total_feature_signal += sample_result["feature_signal"]

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
                    "No usable precomputed cross-view evidence was found. "
                    "Provide cross_view_scores, cross_view_confidence, cross_view_matches, "
                    "or cross_view_features in sample metadata."
                ),
            )

        num_samples = len(evaluated_samples)
        averages = {
            "cross_view_consistency": total_score / num_samples,
            "score_signal": total_score_signal / num_samples,
            "match_signal": total_match_signal / num_samples,
            "feature_signal": total_feature_signal / num_samples,
        }
        return self._result(
            score=averages["cross_view_consistency"],
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

    def _score_sample(self, sample: GenerationSample) -> dict[str, Any]:
        metadata = sample.metadata or {}
        views = self._extract_views(metadata)
        pairs = self._extract_camera_pairs(metadata, views)

        if len(views) < 2 and not pairs:
            raise _SkipSample("multi-view data required")

        score_signal = self._aggregate_scores(metadata)
        match_signal = self._aggregate_matches(metadata, pairs)
        feature_signal = self._aggregate_features(metadata, pairs)

        if score_signal is None and match_signal is None and feature_signal is None:
            if views:
                raise _SkipSample("precomputed cross-view evidence required")
            raise _SkipSample("multi-view data required")

        weighted_total = 0.0
        weight_sum = 0.0
        if score_signal is not None:
            weighted_total += self.score_weight * score_signal
            weight_sum += self.score_weight
        if match_signal is not None:
            weighted_total += self.match_weight * match_signal
            weight_sum += self.match_weight
        if feature_signal is not None:
            weighted_total += self.feature_weight * feature_signal
            weight_sum += self.feature_weight

        score = weighted_total / weight_sum if weight_sum > 0 else 0.0
        return {
            "sample_id": sample.sample_id,
            "score": float(score),
            "score_signal": self._or_zero(score_signal),
            "match_signal": self._or_zero(match_signal),
            "feature_signal": self._or_zero(feature_signal),
            "num_views": len(views),
            "num_pairs": len(pairs),
        }

    def _extract_views(self, metadata: dict[str, Any]) -> list[str]:
        camera_videos = metadata.get(self.camera_videos_key)
        if isinstance(camera_videos, dict):
            return [str(key) for key in camera_videos.keys()]

        multi_view_videos = metadata.get(self.multi_view_videos_key)
        if isinstance(multi_view_videos, dict):
            return [str(key) for key in multi_view_videos.keys()]
        if isinstance(multi_view_videos, list):
            return [str(index) for index, _ in enumerate(multi_view_videos)]

        views = metadata.get(self.views_key)
        if isinstance(views, dict):
            return [str(key) for key in views.keys()]
        if isinstance(views, list):
            return [str(item) for item in views]
        return []

    def _extract_camera_pairs(
        self, metadata: dict[str, Any], views: list[str]
    ) -> list[tuple[str, str]]:
        raw_pairs = metadata.get(self.camera_pairs_key)
        if isinstance(raw_pairs, list):
            pairs = []
            for item in raw_pairs:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    pairs.append((str(item[0]), str(item[1])))
                elif isinstance(item, dict):
                    left = item.get("view_a") or item.get("left") or item.get("src")
                    right = item.get("view_b") or item.get("right") or item.get("dst")
                    if left is not None and right is not None:
                        pairs.append((str(left), str(right)))
            if pairs:
                return pairs

        if len(views) < 2:
            return []
        pairs = []
        for index, left in enumerate(views):
            for right in views[index + 1 :]:
                pairs.append((left, right))
        return pairs

    def _aggregate_scores(self, metadata: dict[str, Any]) -> float | None:
        values = []
        values.extend(self._extract_numeric_values(metadata.get(self.cross_view_scores_key)))
        values.extend(self._extract_numeric_values(metadata.get(self.cross_view_confidence_key)))
        if not values:
            return None
        clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
        return float(clipped.mean())

    def _aggregate_matches(
        self, metadata: dict[str, Any], pairs: list[tuple[str, str]]
    ) -> float | None:
        raw_matches = metadata.get(self.cross_view_matches_key)
        if raw_matches is None:
            return None

        pair_entries = self._resolve_pair_mapping(raw_matches, pairs)
        scores = []
        for _, value in pair_entries:
            score = self._score_match_entry(value)
            if score is not None:
                scores.append(score)
        if not scores:
            return None
        return float(sum(scores) / len(scores))

    def _aggregate_features(
        self, metadata: dict[str, Any], pairs: list[tuple[str, str]]
    ) -> float | None:
        raw_features = metadata.get(self.cross_view_features_key)
        if raw_features is None:
            return None

        if isinstance(raw_features, dict):
            if pairs:
                pair_scores = []
                for left, right in pairs:
                    if left not in raw_features or right not in raw_features:
                        continue
                    similarity = self._feature_similarity(raw_features[left], raw_features[right])
                    if similarity is not None:
                        pair_scores.append(similarity)
                if pair_scores:
                    return float(sum(pair_scores) / len(pair_scores))

            pair_entries = self._resolve_pair_mapping(raw_features, pairs)
            pair_scores = []
            for _, value in pair_entries:
                similarity = self._feature_similarity_from_pair_value(value)
                if similarity is not None:
                    pair_scores.append(similarity)
            if pair_scores:
                return float(sum(pair_scores) / len(pair_scores))

        elif isinstance(raw_features, list) and len(raw_features) >= 2:
            pair_scores = []
            for index in range(len(raw_features) - 1):
                similarity = self._feature_similarity(raw_features[index], raw_features[index + 1])
                if similarity is not None:
                    pair_scores.append(similarity)
            if pair_scores:
                return float(sum(pair_scores) / len(pair_scores))

        return None

    def _resolve_pair_mapping(
        self, raw_value: Any, pairs: list[tuple[str, str]]
    ) -> list[tuple[tuple[str, str] | None, Any]]:
        if isinstance(raw_value, list):
            resolved = []
            for item in raw_value:
                if isinstance(item, dict):
                    pair = self._extract_pair_from_dict(item)
                    resolved.append((pair, item))
            return resolved

        if isinstance(raw_value, dict):
            resolved = []
            for key, value in raw_value.items():
                pair = self._extract_pair_from_key(key)
                if pair is None and isinstance(value, dict):
                    pair = self._extract_pair_from_dict(value)
                resolved.append((pair, value))
            if resolved:
                return resolved

        return [((left, right), raw_value) for left, right in pairs]

    def _extract_pair_from_key(self, key: Any) -> tuple[str, str] | None:
        if isinstance(key, (list, tuple)) and len(key) == 2:
            return str(key[0]), str(key[1])
        if isinstance(key, str):
            for sep in ("|", "->", ",", ":"):
                if sep in key:
                    left, right = key.split(sep, 1)
                    return left.strip(), right.strip()
        return None

    def _extract_pair_from_dict(self, value: dict[str, Any]) -> tuple[str, str] | None:
        left = value.get("view_a") or value.get("left") or value.get("src")
        right = value.get("view_b") or value.get("right") or value.get("dst")
        if left is None or right is None:
            return None
        return str(left), str(right)

    def _score_match_entry(self, value: Any) -> float | None:
        if isinstance(value, dict):
            confidence = self._mean_from_numeric(value.get("confidence"))
            if confidence is None:
                confidence = self._mean_from_numeric(value.get("match_confidence"))

            inlier_ratio = value.get("inlier_ratio")
            if isinstance(inlier_ratio, (int, float)):
                inlier_ratio_value = float(np.clip(inlier_ratio, 0.0, 1.0))
            else:
                inlier_ratio_value = None

            valid_match_count = value.get("valid_match_count")
            if not isinstance(valid_match_count, (int, float)):
                valid_match_count = value.get("match_count")
            if isinstance(valid_match_count, (int, float)):
                count_score = 1.0 - math.exp(-float(valid_match_count) / 50.0)
            else:
                count_score = None

            matches = value.get("matches")
            if matches is not None:
                matches_array = np.asarray(matches)
                if matches_array.ndim >= 2 and matches_array.shape[0] > 0 and count_score is None:
                    count_score = 1.0 - math.exp(-float(matches_array.shape[0]) / 50.0)

            weighted = []
            if confidence is not None:
                weighted.append((0.45, confidence))
            if inlier_ratio_value is not None:
                weighted.append((0.35, inlier_ratio_value))
            if count_score is not None:
                weighted.append((0.2, count_score))
            if not weighted:
                return None
            weight_sum = sum(weight for weight, _ in weighted)
            return float(sum(weight * score for weight, score in weighted) / weight_sum)

        numeric_values = self._extract_numeric_values(value)
        if numeric_values:
            return float(np.clip(np.mean(numeric_values), 0.0, 1.0))
        return None

    def _feature_similarity_from_pair_value(self, value: Any) -> float | None:
        if isinstance(value, dict):
            left = value.get("feature_a") or value.get("left_feature") or value.get("src_feature")
            right = value.get("feature_b") or value.get("right_feature") or value.get("dst_feature")
            if left is not None and right is not None:
                return self._feature_similarity(left, right)
        return None

    def _feature_similarity(self, left: Any, right: Any) -> float | None:
        left_array = np.asarray(left, dtype=np.float64).reshape(-1)
        right_array = np.asarray(right, dtype=np.float64).reshape(-1)
        if left_array.size == 0 or right_array.size == 0 or left_array.shape != right_array.shape:
            return None
        left_norm = np.linalg.norm(left_array)
        right_norm = np.linalg.norm(right_array)
        if left_norm <= 1e-8 or right_norm <= 1e-8:
            return None
        cosine = float(np.dot(left_array, right_array) / (left_norm * right_norm))
        return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))

    def _extract_numeric_values(self, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, list):
            values: list[float] = []
            for item in value:
                values.extend(self._extract_numeric_values(item))
            return values
        if isinstance(value, dict):
            values: list[float] = []
            for item in value.values():
                values.extend(self._extract_numeric_values(item))
            return values
        return []

    def _mean_from_numeric(self, value: Any) -> float | None:
        values = self._extract_numeric_values(value)
        if not values:
            return None
        return float(np.clip(np.mean(values), 0.0, 1.0))

    def _or_zero(self, value: float | None) -> float:
        return 0.0 if value is None else float(value)

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


CrossViewConsistency = CrossViewConsistencyMetric


class _SkipSample(Exception):
    pass
