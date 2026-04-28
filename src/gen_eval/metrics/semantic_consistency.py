"""Semantic consistency metric for fixed multi-view prepared segmentation data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gen_eval.schemas import GenerationSample

EXPECTED_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)


class SemanticConsistencyMetric:
    """Measure temporal consistency from prepared semantic masks or segmentations."""

    name = "semantic_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.weights = (0.4, 0.4, 0.2)
        self.erosion_k = 2
        self.min_iou = 0.1
        self.approximate_palette = bool(self.config.get("approximate_palette", True))
        self.ignore_color = self._normalize_color(
            self.config.get("ignore_color", [255, 255, 0])
        )
        self.max_colors_auto = self.config.get("max_colors_auto", 1024)
        self.segmentation_video_key = str(
            self.config.get("segmentation_video_key", "segmentation_video")
        )
        self.segmentation_frames_key = str(
            self.config.get("segmentation_frames_key", "segmentation_frames")
        )
        self.semantic_masks_key = str(
            self.config.get("semantic_masks_key", "semantic_masks")
        )
        self.palette_key = str(self.config.get("palette_key", "segmentation_palette"))
        self.num_classes_key = str(
            self.config.get("num_classes_key", "semantic_num_classes")
        )

    def evaluate(self, samples: list[GenerationSample]) -> dict[str, Any]:
        runtime, reason = self._get_runtime()
        if runtime is None:
            return {
                "metric": self.name,
                "status": "skipped",
                "num_samples": len(samples),
                "valid_sample_count": 0,
                "mean_semantic_consistency_score": None,
                "details": {
                    "evaluated_samples": [],
                    "skipped_samples": [
                        {
                            "sample_id": getattr(sample, "sample_id", "unknown"),
                            "reason": reason,
                        }
                        for sample in samples
                    ],
                    "failed_samples": [],
                },
                "reason": reason,
            }

        evaluated_samples: list[dict[str, Any]] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        valid_scores: list[float] = []

        for sample in samples:
            sample_id = getattr(sample, "sample_id", None) or "unknown"
            try:
                sample_result = self._evaluate_sample(sample, runtime)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                sample_result = {
                    "sample_id": sample_id,
                    "status": "failed",
                    "reason": str(exc),
                }

            status = sample_result.get("status")
            score = sample_result.get("semantic_consistency_score")
            if status == "success" and isinstance(score, (int, float)):
                evaluated_samples.append(
                    {
                        "sample_id": sample_id,
                        "semantic_consistency_score": float(score),
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
            reason = (
                "No prepared semantic data was evaluable for the fixed expected views."
            )

        result: dict[str, Any] = {
            "metric": self.name,
            "status": status,
            "num_samples": len(samples),
            "valid_sample_count": len(valid_scores),
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

    def _evaluate_sample(
        self,
        sample: GenerationSample,
        runtime: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = sample.metadata or {}
        view_scores: list[float] = []

        for view in EXPECTED_CAMERA_VIEWS:
            try:
                masks, num_classes = self._load_view_masks(metadata, runtime, view)
            except _SkipSample:
                continue

            scores = self._tscs_score(
                masks=masks,
                num_classes=num_classes,
                runtime=runtime,
                weights=self.weights,
                erosion_k=self.erosion_k,
                min_iou=self.min_iou,
            )
            view_scores.append(float(scores["TSCS"]))

        if not view_scores:
            try:
                masks, num_classes = self._load_legacy_sample_masks(sample, runtime)
            except _SkipSample:
                return {
                    "sample_id": sample.sample_id,
                    "status": "skipped",
                    "reason": (
                        "No evaluable prepared semantic data found for expected views."
                    ),
                }

            scores = self._tscs_score(
                masks=masks,
                num_classes=num_classes,
                runtime=runtime,
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

    def _get_runtime(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            global np
            import numpy as np  # type: ignore
            import imageio.v2 as imageio  # type: ignore
            from PIL import Image
            from scipy.optimize import linear_sum_assignment
            from scipy.spatial import cKDTree
            from scipy.spatial.distance import jensenshannon
            from skimage.measure import label, regionprops
            from skimage.morphology import erosion, square
        except Exception as exc:
            return None, f"Required scoring dependencies are unavailable: {exc}"

        return {
            "imageio": imageio,
            "Image": Image,
            "linear_sum_assignment": linear_sum_assignment,
            "cKDTree": cKDTree,
            "jensenshannon": jensenshannon,
            "label": label,
            "regionprops": regionprops,
            "erosion": erosion,
            "square": square,
        }, None

    def _load_view_masks(
        self,
        metadata: dict[str, Any],
        runtime: dict[str, Any],
        view: str,
    ) -> tuple[np.ndarray, int]:
        if self.semantic_masks_key in metadata and isinstance(
            metadata[self.semantic_masks_key], dict
        ):
            masks = self._load_semantic_masks(metadata[self.semantic_masks_key].get(view))
            num_classes = self._resolve_num_classes(metadata, masks, view)
            return masks, num_classes

        if self.segmentation_frames_key in metadata and isinstance(
            metadata[self.segmentation_frames_key], dict
        ):
            video_rgb = self._load_segmentation_frames(
                metadata[self.segmentation_frames_key].get(view),
                runtime,
            )
            palette = self._load_palette(self._resolve_palette(metadata, view))
            masks, num_classes, _ = self._labels_from_video_colormap(
                video_rgb,
                runtime,
                palette=palette,
            )
            return masks, num_classes

        if self.segmentation_video_key in metadata and isinstance(
            metadata[self.segmentation_video_key], dict
        ):
            video_rgb = self._load_segmentation_video(
                metadata[self.segmentation_video_key].get(view),
                runtime,
            )
            palette = self._load_palette(self._resolve_palette(metadata, view))
            masks, num_classes, _ = self._labels_from_video_colormap(
                video_rgb,
                runtime,
                palette=palette,
            )
            return masks, num_classes

        raise _SkipSample(f"No semantic data for view {view}.")

    def _load_legacy_sample_masks(
        self,
        sample: GenerationSample,
        runtime: dict[str, Any],
    ) -> tuple[np.ndarray, int]:
        metadata = sample.metadata or {}

        if self.semantic_masks_key in metadata and not isinstance(
            metadata[self.semantic_masks_key], dict
        ):
            masks = self._load_semantic_masks(metadata[self.semantic_masks_key])
            num_classes = self._resolve_num_classes(metadata, masks, None)
            return masks, num_classes

        if self.segmentation_frames_key in metadata and not isinstance(
            metadata[self.segmentation_frames_key], dict
        ):
            video_rgb = self._load_segmentation_frames(
                metadata[self.segmentation_frames_key], runtime
            )
            palette = self._load_palette(self._resolve_palette(metadata, None))
            masks, num_classes, _ = self._labels_from_video_colormap(
                video_rgb,
                runtime,
                palette=palette,
            )
            return masks, num_classes

        if self.segmentation_video_key in metadata and not isinstance(
            metadata[self.segmentation_video_key], dict
        ):
            video_rgb = self._load_segmentation_video(
                metadata[self.segmentation_video_key], runtime
            )
            palette = self._load_palette(self._resolve_palette(metadata, None))
            masks, num_classes, _ = self._labels_from_video_colormap(
                video_rgb,
                runtime,
                palette=palette,
            )
            return masks, num_classes

        raise _SkipSample("Missing legacy single-sequence semantic data.")

    def _resolve_palette(self, metadata: dict[str, Any], view: str | None) -> Any:
        raw_palette = metadata.get(self.palette_key)
        if isinstance(raw_palette, dict) and view is not None:
            return raw_palette.get(view)
        return raw_palette

    def _resolve_num_classes(
        self,
        metadata: dict[str, Any],
        masks: np.ndarray,
        view: str | None,
    ) -> int:
        raw_num_classes = metadata.get(self.num_classes_key)
        if isinstance(raw_num_classes, dict) and view is not None:
            value = raw_num_classes.get(view)
            if value is not None:
                return int(value)
        elif raw_num_classes is not None:
            return int(raw_num_classes)
        return int(np.max(masks)) + 1 if masks.size else 0

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
                raise _SkipSample(
                    f"Unsupported semantic_masks file format: {path.suffix or '<none>'}"
                )
        else:
            raise _SkipSample(
                "semantic_masks must be a numpy array, list, or .npy/.json path."
            )

        masks = np.asarray(masks)
        if masks.ndim != 3:
            raise _SkipSample("semantic_masks must have shape [T, H, W].")
        return masks.astype(np.int32, copy=False)

    def _load_segmentation_frames(
        self,
        raw_value: Any,
        runtime: dict[str, Any],
    ) -> np.ndarray:
        if raw_value is None:
            raise _SkipSample("segmentation_frames entry is missing.")

        imageio = runtime["imageio"]
        image_class = runtime["Image"]

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
                        raise _SkipSample(
                            f"segmentation frame path does not exist: {item}"
                        )
                    with image_class.open(path) as image:
                        frame_arrays.append(
                            np.asarray(image.convert("RGB"), dtype=np.uint8)
                        )
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
                raise _SkipSample(
                    f"Unsupported segmentation_frames file format: {path.suffix or '<none>'}"
                )
        else:
            raise _SkipSample(
                "segmentation_frames must be an array, list of frame paths, or supported file path."
            )

        return self._ensure_video_rgb(np.asarray(frames))

    def _load_segmentation_video(
        self,
        raw_value: Any,
        runtime: dict[str, Any],
    ) -> np.ndarray:
        if raw_value is None:
            raise _SkipSample("segmentation_video entry is missing.")
        if not isinstance(raw_value, str):
            raise _SkipSample("segmentation_video must be a file path.")
        path = Path(raw_value)
        if not path.exists():
            raise _SkipSample(f"segmentation_video path does not exist: {raw_value}")
        frames = np.asarray(runtime["imageio"].mimread(path), dtype=np.uint8)
        return self._ensure_video_rgb(frames)

    def _load_palette(
        self,
        raw_palette: Any,
    ) -> np.ndarray | dict[tuple[int, int, int], int] | None:
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
            raise _SkipSample(
                "segmentation_palette must be shape [K, 3] or a color-to-id dict."
            )
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
            raise _SkipSample(
                "Prepared segmentation RGB data must be [T,H,W,3] or [T,3,H,W]."
            )
        if video_rgb.dtype != np.uint8:
            video_rgb = np.clip(np.rint(video_rgb), 0, 255).astype(np.uint8)
        return video_rgb

    def _labels_from_video_colormap(
        self,
        video_rgb: np.ndarray,
        runtime: dict[str, Any],
        palette: np.ndarray | dict[tuple[int, int, int], int] | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        ckd_tree = runtime["cKDTree"]
        video_rgb = self._ensure_video_rgb(video_rgb)
        pixels = video_rgb.reshape(-1, 3).astype(np.int16)

        if isinstance(palette, dict):
            items = sorted(palette.items(), key=lambda item: item[1])
            size = max(palette.values()) + 1
            palette_array = np.zeros((size, 3), dtype=np.uint8)
            for rgb, class_id in items:
                palette_array[class_id] = np.asarray(rgb, dtype=np.uint8)
            labels = self._map_pixels_to_palette(
                pixels,
                palette_array,
                ckd_tree,
                approximate=self.approximate_palette,
            )
            num_classes = palette_array.shape[0]
        elif isinstance(palette, np.ndarray):
            palette_array = palette.astype(np.uint8, copy=False)
            labels = self._map_pixels_to_palette(
                pixels,
                palette_array,
                ckd_tree,
                approximate=self.approximate_palette,
            )
            num_classes = palette_array.shape[0]
        else:
            palette_array = self._build_palette_from_video(
                video_rgb,
                max_colors=self.max_colors_auto,
            )
            labels = self._map_pixels_to_palette(
                pixels,
                palette_array,
                ckd_tree,
                approximate=self.approximate_palette,
            )
            num_classes = palette_array.shape[0]

        if self.ignore_color is not None:
            ignore_mask = np.all(
                pixels == np.asarray(self.ignore_color, dtype=np.int16),
                axis=1,
            )
            match_idx = np.where(
                np.all(
                    palette_array == np.asarray(self.ignore_color, dtype=np.uint8),
                    axis=1,
                )
            )[0]
            if match_idx.size > 0:
                labels[ignore_mask] = int(match_idx[0])

        masks = labels.reshape(
            video_rgb.shape[0],
            video_rgb.shape[1],
            video_rgb.shape[2],
        ).astype(np.int32)
        return masks, num_classes, palette_array

    def _build_palette_from_video(
        self,
        video_rgb: np.ndarray,
        max_colors: int | None = None,
    ) -> np.ndarray:
        flat = video_rgb.reshape(-1, 3)
        unique_colors = np.unique(flat, axis=0)
        if max_colors is not None and unique_colors.shape[0] > max_colors:
            raise _SkipSample(
                f"Unique colors={unique_colors.shape[0]} exceed max_colors={max_colors}. "
                "Prepared segmentation video does not look like a clean color map."
            )
        return unique_colors.astype(np.uint8, copy=False)

    def _map_pixels_to_palette(
        self,
        pixels: np.ndarray,
        palette: np.ndarray,
        ckd_tree: Any,
        *,
        approximate: bool,
    ) -> np.ndarray:
        if approximate:
            tree = ckd_tree(palette.astype(np.float32))
            _, nearest = tree.query(pixels.astype(np.float32), k=1)
            return nearest.astype(np.int32)

        lut = {tuple(map(int, palette[i])): i for i in range(palette.shape[0])}
        return np.asarray([lut.get(tuple(pixel), 0) for pixel in pixels], dtype=np.int32)

    def _compute_lfr_interior(
        self,
        masks: np.ndarray,
        num_classes: int,
        runtime: dict[str, Any],
        erosion_k: int,
    ) -> float:
        if masks.shape[0] < 2:
            return 1.0
        square = runtime["square"]
        erosion = runtime["erosion"]
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
            common_interior = interior[t] & interior[t + 1]
            if not common_interior.any():
                continue
            flips.append(
                np.mean(masks[t][common_interior] != masks[t + 1][common_interior])
            )
        if not flips:
            return 1.0
        return 1.0 - float(np.mean(flips))

    def _class_components(
        self,
        mask: np.ndarray,
        class_id: int,
        runtime: dict[str, Any],
    ) -> tuple[np.ndarray | None, list[Any]]:
        cls = (mask == class_id).astype(np.uint8)
        if cls.sum() == 0:
            return None, []
        labeled = runtime["label"](cls, connectivity=1)
        props = runtime["regionprops"](labeled)
        return labeled, props

    def _overlap_counts(
        self,
        labeled_a: np.ndarray,
        props_a: list[Any],
        labeled_b: np.ndarray,
        props_b: list[Any],
    ) -> np.ndarray:
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

    def _compute_sac(
        self,
        masks: np.ndarray,
        num_classes: int,
        runtime: dict[str, Any],
        min_iou: float,
    ) -> float:
        if masks.shape[0] < 2:
            return 1.0
        linear_sum_assignment = runtime["linear_sum_assignment"]
        iou_scores: list[tuple[float, float]] = []

        for t in range(masks.shape[0] - 1):
            mask_a = masks[t]
            mask_b = masks[t + 1]
            for class_id in range(num_classes):
                labeled_a, props_a = self._class_components(mask_a, class_id, runtime)
                labeled_b, props_b = self._class_components(mask_b, class_id, runtime)
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
        return float(
            (np.asarray(ious, dtype=np.float64) * weights_array).sum()
            / (weights_array.sum() + 1e-8)
        )

    def _compute_cds(
        self,
        masks: np.ndarray,
        num_classes: int,
        runtime: dict[str, Any],
    ) -> float:
        if masks.shape[0] < 2:
            return 1.0
        jensenshannon = runtime["jensenshannon"]
        histograms = []
        for t in range(masks.shape[0]):
            hist, _ = np.histogram(masks[t], bins=np.arange(num_classes + 1))
            prob = hist.astype(np.float64)
            prob = (prob + 1e-8) / (prob.sum() + 1e-8 * num_classes)
            histograms.append(prob)

        jsd_scores = []
        for t in range(masks.shape[0] - 1):
            jsd_scores.append(jensenshannon(histograms[t], histograms[t + 1]) ** 2)
        return float(1.0 - np.mean(jsd_scores))

    def _tscs_score(
        self,
        *,
        masks: np.ndarray,
        num_classes: int,
        runtime: dict[str, Any],
        weights: tuple[float, float, float],
        erosion_k: int,
        min_iou: float,
    ) -> dict[str, Any]:
        masks = np.asarray(masks)
        if masks.ndim != 3:
            raise ValueError("Semantic masks must have shape [T, H, W].")
        weight_lfr, weight_sac, weight_cds = weights
        s_lfr = self._compute_lfr_interior(
            masks,
            num_classes,
            runtime,
            erosion_k=erosion_k,
        )
        s_sac = self._compute_sac(masks, num_classes, runtime, min_iou=min_iou)
        s_cds = self._compute_cds(masks, num_classes, runtime)
        tscs = float(weight_lfr * s_lfr + weight_sac * s_sac + weight_cds * s_cds)
        return {
            "TSCS": tscs,
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


SemanticConsistency = SemanticConsistencyMetric


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
