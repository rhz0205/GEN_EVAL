from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
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
        self.approximate_palette = bool(self.config.get("approximate_palette", True))
        self.ignore_color = self._normalize_color(self.config.get("ignore_color", [255, 255, 0]))
        self.max_colors_auto = self.config.get("max_colors_auto", 1024)
        self.segmentation_video_key = str(self.config.get("segmentation_video_key", "segmentation_video"))
        self.segmentation_frames_key = str(self.config.get("segmentation_frames_key", "segmentation_frames"))
        self.semantic_masks_key = str(self.config.get("semantic_masks_key", "semantic_masks"))
        self.palette_key = str(self.config.get("palette_key", "segmentation_palette"))
        self.num_classes_key = str(self.config.get("num_classes_key", "semantic_num_classes"))

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
                evaluated_samples.append(
                    {
                        "sample_id": sample_id,
                        "semantic_consistency_score": valid_score,
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

        return {
            "sample_id": sample.sample_id,
            "status": "success",
            "semantic_consistency_score": mean_or_none(view_scores),
        }

    def _load_view_masks(self, metadata: dict[str, Any], view: str) -> tuple[np.ndarray, int]:
        if self.semantic_masks_key in metadata and isinstance(metadata[self.semantic_masks_key], dict):
            masks = self._load_semantic_masks(metadata[self.semantic_masks_key].get(view))
            num_classes = self._resolve_num_classes(metadata, masks, view)
            return masks, num_classes

        if self.segmentation_frames_key in metadata and isinstance(metadata[self.segmentation_frames_key], dict):
            video_rgb = self._load_segmentation_frames(metadata[self.segmentation_frames_key].get(view))
            palette = self._load_palette(self._resolve_palette(metadata, view))
            masks, num_classes, _ = self._labels_from_video_colormap(video_rgb, palette=palette)
            return masks, num_classes

        if self.segmentation_video_key in metadata and isinstance(metadata[self.segmentation_video_key], dict):
            video_rgb = self._load_segmentation_video(metadata[self.segmentation_video_key].get(view))
            palette = self._load_palette(self._resolve_palette(metadata, view))
            masks, num_classes, _ = self._labels_from_video_colormap(video_rgb, palette=palette)
            return masks, num_classes

        raise _SkipSample(f"No semantic data for view {view}.")

    def _load_legacy_sample_masks(self, sample: GenerationSample) -> tuple[np.ndarray, int]:
        metadata = sample.metadata or {}

        if self.semantic_masks_key in metadata and not isinstance(metadata[self.semantic_masks_key], dict):
            masks = self._load_semantic_masks(metadata[self.semantic_masks_key])
            num_classes = self._resolve_num_classes(metadata, masks, None)
            return masks, num_classes

        if self.segmentation_frames_key in metadata and not isinstance(metadata[self.segmentation_frames_key], dict):
            video_rgb = self._load_segmentation_frames(metadata[self.segmentation_frames_key])
            palette = self._load_palette(self._resolve_palette(metadata, None))
            masks, num_classes, _ = self._labels_from_video_colormap(video_rgb, palette=palette)
            return masks, num_classes

        if self.segmentation_video_key in metadata and not isinstance(metadata[self.segmentation_video_key], dict):
            video_rgb = self._load_segmentation_video(metadata[self.segmentation_video_key])
            palette = self._load_palette(self._resolve_palette(metadata, None))
            masks, num_classes, _ = self._labels_from_video_colormap(video_rgb, palette=palette)
            return masks, num_classes

        raise _SkipSample("Missing legacy single-sequence semantic data.")

    def _resolve_palette(self, metadata: dict[str, Any], view: str | None) -> Any:
        raw_palette = metadata.get(self.palette_key)
        if isinstance(raw_palette, dict) and view is not None:
            return raw_palette.get(view)
        return raw_palette

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

    def _load_segmentation_frames(self, raw_value: Any) -> np.ndarray:
        if raw_value is None:
            raise _SkipSample("segmentation_frames entry is missing.")
        if isinstance(raw_value, np.ndarray):
            frames = raw_value
        elif isinstance(raw_value, list):
            if not raw_value:
                raise _SkipSample("segmentation_frames is empty.")
            if all(isinstance(item, str) for item in raw_value):
                frame_arrays = []
                for item in raw_value:
                    path = Path(item)
                    if not path.exists():
                        raise _SkipSample(f"segmentation frame path does not exist: {item}")
                    with Image.open(path) as image:
                        frame_arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
                frames = np.stack(frame_arrays, axis=0)
            else:
                frames = np.asarray(raw_value)
        elif isinstance(raw_value, str):
            path = Path(raw_value)
            if not path.exists():
                raise _SkipSample(f"segmentation_frames path does not exist: {raw_value}")
            if path.suffix.lower() == ".npy":
                frames = np.load(path, allow_pickle=False)
            elif path.suffix.lower() == ".json":
                frames = np.asarray(json.loads(path.read_text(encoding="utf-8")))
            elif path.suffix.lower() in {".gif", ".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                frames = np.asarray(imageio.mimread(path), dtype=np.uint8)
            else:
                raise _SkipSample(f"Unsupported segmentation_frames file format: {path.suffix or '<none>'}")
        else:
            raise _SkipSample("segmentation_frames must be an array, list of frame paths, or supported file path.")

        return self._ensure_video_rgb(np.asarray(frames))

    def _load_segmentation_video(self, raw_value: Any) -> np.ndarray:
        if raw_value is None:
            raise _SkipSample("segmentation_video entry is missing.")
        if not isinstance(raw_value, str):
            raise _SkipSample("segmentation_video must be a file path.")
        path = Path(raw_value)
        if not path.exists():
            raise _SkipSample(f"segmentation_video path does not exist: {raw_value}")
        frames = np.asarray(imageio.mimread(path), dtype=np.uint8)
        return self._ensure_video_rgb(frames)

    def _load_palette(self, raw_palette: Any) -> np.ndarray | dict[tuple[int, int, int], int] | None:
        if raw_palette is None:
            return None
        if isinstance(raw_palette, dict):
            converted: dict[tuple[int, int, int], int] = {}
            for key, value in raw_palette.items():
                if isinstance(key, str):
                    rgb = tuple(int(part) for part in key.split(","))
                else:
                    rgb = tuple(int(part) for part in key)
                converted[rgb] = int(value)
            return converted
        palette = np.asarray(raw_palette)
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise _SkipSample("segmentation_palette must be shape [K, 3] or a color-to-id dict.")
        return palette.astype(np.uint8, copy=False)

    def _ensure_video_rgb(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.ndim != 4:
            raise _SkipSample("Prepared segmentation RGB data must have 4 dimensions.")
        if frames.shape[-1] == 3:
            video_rgb = frames
        elif frames.shape[1] == 3:
            video_rgb = np.transpose(frames, (0, 2, 3, 1))
        else:
            raise _SkipSample("Prepared segmentation RGB data must be [T,H,W,3] or [T,3,H,W].")
        if video_rgb.dtype != np.uint8:
            video_rgb = np.clip(np.rint(video_rgb), 0, 255).astype(np.uint8)
        return video_rgb

    def _labels_from_video_colormap(
        self,
        video_rgb: np.ndarray,
        palette: np.ndarray | dict[tuple[int, int, int], int] | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        video_rgb = self._ensure_video_rgb(video_rgb)
        pixels = video_rgb.reshape(-1, 3).astype(np.int16)

        if isinstance(palette, dict):
            palette_array = self._palette_dict_to_array(palette)
        elif isinstance(palette, np.ndarray):
            palette_array = palette.astype(np.uint8, copy=False)
        else:
            palette_array = self._build_palette_from_video(video_rgb, max_colors=self.max_colors_auto)

        labels = self._map_pixels_to_palette(pixels, palette_array, approximate=self.approximate_palette)
        num_classes = palette_array.shape[0]

        if self.ignore_color is not None:
            ignore_mask = np.all(pixels == np.asarray(self.ignore_color, dtype=np.int16), axis=1)
            labels[ignore_mask] = self.ignore_label

        masks = labels.reshape(video_rgb.shape[0], video_rgb.shape[1], video_rgb.shape[2]).astype(np.int32)
        return masks, num_classes, palette_array

    def _palette_dict_to_array(self, palette: dict[tuple[int, int, int], int]) -> np.ndarray:
        size = max(palette.values()) + 1 if palette else 0
        palette_array = np.zeros((size, 3), dtype=np.uint8)
        for rgb, class_id in sorted(palette.items(), key=lambda item: item[1]):
            palette_array[class_id] = np.asarray(rgb, dtype=np.uint8)
        return palette_array

    def _build_palette_from_video(self, video_rgb: np.ndarray, max_colors: int | None = None) -> np.ndarray:
        flat = video_rgb.reshape(-1, 3)
        unique_colors = np.unique(flat, axis=0)
        if max_colors is not None and unique_colors.shape[0] > max_colors:
            raise _SkipSample(
                f"Unique colors={unique_colors.shape[0]} exceed max_colors={max_colors}. Prepared segmentation video does not look like a clean color map."
            )
        return unique_colors.astype(np.uint8, copy=False)

    def _map_pixels_to_palette(self, pixels: np.ndarray, palette: np.ndarray, *, approximate: bool) -> np.ndarray:
        if approximate:
            tree = cKDTree(palette.astype(np.float32))
            _, nearest = tree.query(pixels.astype(np.float32), k=1)
            return nearest.astype(np.int32)

        lookup = {tuple(map(int, palette[i])): i for i in range(palette.shape[0])}
        return np.asarray([lookup.get(tuple(pixel), 0) for pixel in pixels], dtype=np.int32)

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

    def _normalize_color(self, raw_color: Any) -> tuple[int, int, int] | None:
        if raw_color is None:
            return None
        if len(raw_color) != 3:
            raise ValueError("ignore_color must contain exactly 3 values.")
        return tuple(int(value) for value in raw_color)


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
