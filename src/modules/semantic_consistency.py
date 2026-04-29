from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from skimage.measure import label, regionprops
from skimage.morphology import erosion, square

from modules.base import BaseModule
from schemas import GenerationSample

EXPECTED_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)

SEMANTIC_WEIGHTS: tuple[float, float, float] = (0.5, 0.4, 0.1)
IGNORE_LABEL = -1


class SemanticConsistency(BaseModule):
    name = "semantic_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.weights = SEMANTIC_WEIGHTS
        self.erosion_k = int(self.config.get("erosion_k", 2))
        self.min_iou = float(self.config.get("min_iou", 0.1))
        self.ignore_label = int(self.config.get("ignore_label", IGNORE_LABEL))
        self.semantic_masks_key = str(self.config.get("semantic_masks_key", "semantic_masks"))
        self.num_classes_key = str(self.config.get("num_classes_key", "semantic_num_classes"))
        self.ignore_label_key = str(self.config.get("ignore_label_key", "semantic_ignore_label"))

    def evaluate(self, samples: list[GenerationSample]) -> dict[str, Any]:
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
            score = sample_result.get("semantic_consistency_score")
            if status == "success" and is_finite_number(score):
                valid_score = float(score)
                view_scores = sample_result.get("view_scores", {})
                evaluated_samples.append(
                    {
                        "sample_id": sample_id,
                        "semantic_consistency_score": valid_score,
                        "view_scores": view_scores if isinstance(view_scores, dict) else {},
                        "num_evaluated_views": len(view_scores) if isinstance(view_scores, dict) else 0,
                    }
                )
                valid_scores.append(valid_score)
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
            reason = "No evaluable semantic data was found."

        result: dict[str, Any] = {
            "metric": self.name,
            "status": status,
            "num_samples": len(samples),
            "valid_sample_count": len(valid_scores),
            "skipped_sample_count": len(skipped_samples),
            "failed_sample_count": len(failed_samples),
            "mean_semantic_consistency_score": mean_score,
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
        view_details: dict[str, dict[str, Any]] = {}

        for view in EXPECTED_CAMERA_VIEWS:
            try:
                masks, num_classes = self._load_view_masks(metadata, view)
            except _SkipSample:
                continue

            scores = self._tscs_score(
                masks=masks,
                num_classes=num_classes,
                weights=self.weights,
                erosion_k=self.erosion_k,
                min_iou=self.min_iou,
            )
            view_scores.append(float(scores["TSCS"]))
            view_details[view] = {
                "semantic_consistency_score": float(scores["TSCS"]),
                "s_lfr": float(scores["S_LFR"]),
                "s_sac": float(scores["S_SAC"]),
                "s_cds": float(scores["S_CDS"]),
                "num_classes": int(num_classes),
                "num_frames": int(masks.shape[0]),
            }

        if not view_scores:
            try:
                masks, num_classes = self._load_legacy_sample_masks(sample)
            except _SkipSample:
                return skipped_result(sample.sample_id, "No evaluable prepared semantic data found.")

            scores = self._tscs_score(
                masks=masks,
                num_classes=num_classes,
                weights=self.weights,
                erosion_k=self.erosion_k,
                min_iou=self.min_iou,
            )
            view_scores.append(float(scores["TSCS"]))
            view_details["legacy"] = {
                "semantic_consistency_score": float(scores["TSCS"]),
                "s_lfr": float(scores["S_LFR"]),
                "s_sac": float(scores["S_SAC"]),
                "s_cds": float(scores["S_CDS"]),
                "num_classes": int(num_classes),
                "num_frames": int(masks.shape[0]),
            }

        return {
            "sample_id": sample.sample_id,
            "status": "success",
            "semantic_consistency_score": mean_or_none(view_scores),
            "view_scores": view_details,
            "num_evaluated_views": len(view_details),
        }

    def _load_view_masks(self, metadata: dict[str, Any], view: str) -> tuple[np.ndarray, int]:
        semantic_masks = metadata.get(self.semantic_masks_key)
        if not isinstance(semantic_masks, dict):
            raise _SkipSample(f"metadata['{self.semantic_masks_key}'] must be a dict keyed by view.")
        masks = self._load_semantic_masks(semantic_masks.get(view))
        num_classes = self._resolve_num_classes(metadata, masks, view)
        self._resolve_ignore_label(metadata, view)
        return masks, num_classes

    def _load_legacy_sample_masks(self, sample: GenerationSample) -> tuple[np.ndarray, int]:
        metadata = sample.metadata or {}
        semantic_masks = metadata.get(self.semantic_masks_key)
        if isinstance(semantic_masks, dict):
            raise _SkipSample("semantic_masks must provide expected per-view entries.")
        if semantic_masks is None:
            raise _SkipSample("Missing prepared semantic_masks.")
        masks = self._load_semantic_masks(semantic_masks)
        num_classes = self._resolve_num_classes(metadata, masks, None)
        self._resolve_ignore_label(metadata, None)
        return masks, num_classes

    def _resolve_num_classes(self, metadata: dict[str, Any], masks: np.ndarray, view: str | None) -> int:
        raw_num_classes = metadata.get(self.num_classes_key)
        if isinstance(raw_num_classes, dict) and view is not None:
            value = raw_num_classes.get(view)
            if value is not None:
                return int(value)
        if raw_num_classes is not None and not isinstance(raw_num_classes, dict):
            return int(raw_num_classes)

        valid_masks = masks[masks != self.ignore_label]
        return int(np.max(valid_masks)) + 1 if valid_masks.size else 0

    def _resolve_ignore_label(self, metadata: dict[str, Any], view: str | None) -> int:
        raw_ignore_label = metadata.get(self.ignore_label_key)
        if isinstance(raw_ignore_label, dict) and view is not None:
            value = raw_ignore_label.get(view)
            if value is not None:
                if int(value) != self.ignore_label:
                    raise _SkipSample(
                        f"semantic ignore label for view {view} is {value}, expected {self.ignore_label}."
                    )
                return int(value)
        if raw_ignore_label is not None and not isinstance(raw_ignore_label, dict):
            if int(raw_ignore_label) != self.ignore_label:
                raise _SkipSample(
                    f"semantic ignore label is {raw_ignore_label}, expected {self.ignore_label}."
                )
            return int(raw_ignore_label)
        return self.ignore_label

    def _load_semantic_masks(self, raw_value: Any) -> np.ndarray:
        if raw_value is None:
            raise _SkipSample("semantic_masks entry is missing.")
        if isinstance(raw_value, np.ndarray):
            masks = raw_value
        elif isinstance(raw_value, list):
            masks = np.asarray(raw_value)
        elif isinstance(raw_value, str):
            path = Path(raw_value)
            if not path.exists():
                raise _SkipSample(f"semantic_masks path does not exist: {raw_value}")
            if path.suffix.lower() == ".npy":
                masks = np.load(path, allow_pickle=False)
            elif path.suffix.lower() == ".json":
                masks = np.asarray(json.loads(path.read_text(encoding="utf-8")))
            else:
                raise _SkipSample(f"Unsupported semantic_masks file format: {path.suffix or '<none>'}")
        else:
            raise _SkipSample("semantic_masks must be a numpy array, list, or .npy/.json path.")

        masks = np.asarray(masks)
        if masks.ndim != 3:
            raise _SkipSample("semantic_masks must have shape [T, H, W].")
        return masks.astype(np.int32, copy=False)

    def _compute_lfr_interior(self, masks: np.ndarray, num_classes: int, erosion_k: int) -> float:
        if masks.shape[0] < 2:
            return 1.0

        structure = square(max(1, erosion_k))
        interior = np.zeros(masks.shape, dtype=bool)
        for t in range(masks.shape[0]):
            interior_t = np.zeros(masks.shape[1:], dtype=bool)
            for class_id in range(num_classes):
                class_mask = masks[t] == class_id
                if class_mask.any():
                    interior_t |= erosion(class_mask, structure)
            interior[t] = interior_t

        flips = []
        for t in range(masks.shape[0] - 1):
            valid_pair = (masks[t] != self.ignore_label) & (masks[t + 1] != self.ignore_label)
            common_interior = interior[t] & interior[t + 1] & valid_pair
            if not common_interior.any():
                continue
            flips.append(np.mean(masks[t][common_interior] != masks[t + 1][common_interior]))
        if not flips:
            return 1.0
        return clamp01(1.0 - float(np.mean(flips)))

    def _class_components(self, mask: np.ndarray, class_id: int) -> tuple[np.ndarray | None, list[Any]]:
        cls = (mask == class_id).astype(np.uint8)
        if cls.sum() == 0:
            return None, []
        labeled = label(cls, connectivity=1)
        props = regionprops(labeled)
        return labeled, props

    def _overlap_counts(self, labeled_a: np.ndarray, props_a: list[Any], labeled_b: np.ndarray, props_b: list[Any]) -> np.ndarray:
        if not props_a or not props_b:
            return np.zeros((len(props_a), len(props_b)), dtype=np.int64)
        max_b = max(prop.label for prop in props_b)
        code = labeled_a.astype(np.int64) * (max_b + 1) + labeled_b.astype(np.int64)
        code = code[(labeled_a > 0) & (labeled_b > 0)]
        if code.size == 0:
            return np.zeros((len(props_a), len(props_b)), dtype=np.int64)
        unique, counts = np.unique(code, return_counts=True)
        overlap = np.zeros((labeled_a.max() + 1, labeled_b.max() + 1), dtype=np.int64)
        overlap.flat[unique] = counts
        idx_a = [prop.label for prop in props_a]
        idx_b = [prop.label for prop in props_b]
        return overlap[np.ix_(idx_a, idx_b)]

    def _compute_sac(self, masks: np.ndarray, num_classes: int, min_iou: float) -> float:
        if masks.shape[0] < 2:
            return 1.0
        iou_scores: list[tuple[float, float]] = []

        for t in range(masks.shape[0] - 1):
            mask_a = masks[t]
            mask_b = masks[t + 1]
            for class_id in range(num_classes):
                labeled_a, props_a = self._class_components(mask_a, class_id)
                labeled_b, props_b = self._class_components(mask_b, class_id)
                if labeled_a is None or labeled_b is None:
                    if labeled_a is not None and labeled_b is None:
                        for prop in props_a:
                            iou_scores.append((0.0, float(prop.area)))
                    continue

                areas_a = np.asarray([prop.area for prop in props_a], dtype=np.float64)
                areas_b = np.asarray([prop.area for prop in props_b], dtype=np.float64)
                overlap = self._overlap_counts(labeled_a, props_a, labeled_b, props_b)
                union = areas_a[:, None] + areas_b[None, :] - overlap
                with np.errstate(divide="ignore", invalid="ignore"):
                    iou = np.where(union > 0, overlap / union, 0.0)
                if iou.size == 0:
                    continue
                rows, cols = linear_sum_assignment(1.0 - iou)
                for row, col in zip(rows, cols):
                    score = float(iou[row, col]) if iou[row, col] >= min_iou else 0.0
                    iou_scores.append((score, float(areas_a[row])))
                if len(props_a) > len(props_b):
                    unmatched = set(range(len(props_a))) - set(rows)
                    for row in unmatched:
                        iou_scores.append((0.0, float(areas_a[row])))

        if not iou_scores:
            return 1.0
        ious, weights = zip(*iou_scores)
        weights_array = np.asarray(weights, dtype=np.float64)
        score = (np.asarray(ious, dtype=np.float64) * weights_array).sum() / (weights_array.sum() + 1e-8)
        return clamp01(float(score))

    def _compute_cds(self, masks: np.ndarray, num_classes: int) -> float:
        if masks.shape[0] < 2:
            return 1.0

        histograms: list[np.ndarray | None] = []
        bins = np.arange(num_classes + 1)
        for t in range(masks.shape[0]):
            valid_pixels = masks[t][masks[t] != self.ignore_label]
            valid_pixels = valid_pixels[(valid_pixels >= 0) & (valid_pixels < num_classes)]
            if valid_pixels.size == 0:
                histograms.append(None)
                continue
            hist, _ = np.histogram(valid_pixels, bins=bins)
            prob = hist.astype(np.float64)
            prob = (prob + 1e-8) / (prob.sum() + 1e-8 * num_classes)
            histograms.append(prob)

        jsd_scores = []
        for t in range(masks.shape[0] - 1):
            hist_a = histograms[t]
            hist_b = histograms[t + 1]
            if hist_a is None or hist_b is None:
                continue
            jsd_scores.append(jensenshannon(hist_a, hist_b) ** 2)
        if not jsd_scores:
            return 1.0
        return clamp01(1.0 - float(np.mean(jsd_scores)))

    def _tscs_score(
        self,
        *,
        masks: np.ndarray,
        num_classes: int,
        weights: tuple[float, float, float],
        erosion_k: int,
        min_iou: float,
    ) -> dict[str, Any]:
        masks = np.asarray(masks)
        if masks.ndim != 3:
            raise ValueError("Semantic masks must have shape [T, H, W].")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive after ignoring invalid pixels.")

        weight_lfr, weight_sac, weight_cds = weights
        s_lfr = self._compute_lfr_interior(masks, num_classes, erosion_k=erosion_k)
        s_sac = self._compute_sac(masks, num_classes, min_iou=min_iou)
        s_cds = self._compute_cds(masks, num_classes)
        tscs = float(weight_lfr * s_lfr + weight_sac * s_sac + weight_cds * s_cds)
        return {
            "TSCS": clamp01(tscs),
            "S_LFR": s_lfr,
            "S_SAC": s_sac,
            "S_CDS": s_cds,
        }

class _SkipSample(Exception):
    pass


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
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def clamp01(value: float) -> float:
    if not np.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
