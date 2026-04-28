"""CLIP-based temporal consistency metric for fixed multi-view videos."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

EXPECTED_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)


class TemporalConsistencyMetric:
    """Measure CLIP-based temporal stability across fixed driving-camera views."""

    name = "temporal_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.camera_videos_key = self.config.get("camera_videos_key", "camera_videos")
        self.device = self.config.get("device", "cuda")
        self.weight_path = (
            self.config.get("weight_path")
            or self.config.get("clip_weight_path")
            or self.config.get("local_save_path")
        )
        self.batch_size = int(self.config.get("batch_size", 16))

        self._torch = None
        self._clip = None
        self._cv2 = None
        self._pil_image = None
        self._clip_model = None
        self._preprocess = None

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        evaluated_samples: list[dict[str, Any]] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        valid_scores: list[float] = []

        runtime_status = self._ensure_clip()
        if runtime_status is not None:
            return {
                "metric": self.name,
                "status": "skipped",
                "num_samples": len(samples),
                "valid_sample_count": 0,
                "mean_temporal_consistency_score": None,
                "details": {
                    "evaluated_samples": [],
                    "skipped_samples": [
                        {
                            "sample_id": getattr(sample, "sample_id", "unknown"),
                            "reason": runtime_status,
                        }
                        for sample in samples
                    ],
                    "failed_samples": [],
                },
                "reason": runtime_status,
            }

        for sample in samples:
            sample_id = getattr(sample, "sample_id", None) or "unknown"

            try:
                sample_result = self._evaluate_sample(sample)
            except Exception as exc:  # noqa: BLE001
                sample_result = {
                    "sample_id": sample_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }

            status = sample_result.get("status")
            score = sample_result.get("temporal_consistency_score")

            if status == "success" and is_finite_number(score):
                evaluated_samples.append(
                    {
                        "sample_id": sample_id,
                        "temporal_consistency_score": float(score),
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
            reason = "No sample produced a valid temporal_consistency_score."

        result: dict[str, Any] = {
            "metric": self.name,
            "status": status,
            "num_samples": len(samples),
            "valid_sample_count": len(valid_scores),
            "mean_temporal_consistency_score": mean_score,
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
        camera_videos = metadata.get(self.camera_videos_key)

        if not isinstance(camera_videos, dict) or not camera_videos:
            return {
                "sample_id": sample_id,
                "status": "skipped",
                "reason": "metadata['camera_videos'] is required and must be a non-empty dict.",
            }

        normalized_videos = {
            str(view): str(path) for view, path in camera_videos.items() if path is not None
        }
        missing_views = [
            view for view in EXPECTED_CAMERA_VIEWS if view not in normalized_videos
        ]
        if missing_views:
            return {
                "sample_id": sample_id,
                "status": "skipped",
                "reason": f"Missing expected camera views: {', '.join(missing_views)}.",
            }

        view_scores: list[float] = []
        for view in EXPECTED_CAMERA_VIEWS:
            view_score = self._evaluate_view_video(normalized_videos[view])
            if is_finite_number(view_score):
                view_scores.append(float(view_score))

        if not view_scores:
            return {
                "sample_id": sample_id,
                "status": "skipped",
                "reason": "No expected camera view produced a valid temporal consistency score.",
            }

        return {
            "sample_id": sample_id,
            "status": "success",
            "temporal_consistency_score": mean_or_none(view_scores),
        }

    def _evaluate_view_video(self, video_path: str) -> float | None:
        path = Path(video_path)
        if not path.exists() or not path.is_file():
            return None

        frames = self._read_all_frames(video_path)
        if len(frames) < 2:
            return None

        features = self._extract_clip_features(frames)
        if features is None or len(features) < 2:
            return None

        return self._compute_temporal_consistency_score(features)

    def _compute_temporal_consistency_score(self, features: Any) -> float | None:
        torch = self._torch
        if torch is None:
            raise RuntimeError("torch is not initialized")
        if features is None or len(features) < 2:
            return None

        with torch.no_grad():
            adjacent_similarities = torch.nn.functional.cosine_similarity(
                features[:-1],
                features[1:],
                dim=-1,
            )
            adjacent_similarities = torch.clamp(adjacent_similarities, min=0.0)
            acm = float(adjacent_similarities.mean().item())

            if len(features) >= 3:
                velocity = (features[1:] - features[:-1]).norm(dim=1)
                acceleration = (
                    features[2:] - 2 * features[1:-1] + features[:-2]
                ).norm(dim=1)
                denom = 0.5 * (velocity[1:] + velocity[:-1]) + 1e-8
                tji_tensor = (acceleration / denom).mean()
                tji = float(tji_tensor.item())
            else:
                tji = 0.0

            score = acm / (1.0 + tji)
            if not math.isfinite(score):
                return 0.0
            return clamp01(float(score))

    def _ensure_clip(self) -> str | None:
        if self._clip_model is not None and self._preprocess is not None:
            return None

        try:
            import torch  # type: ignore

            self._torch = torch
        except Exception as exc:  # noqa: BLE001
            return f"torch is required for temporal_consistency: {type(exc).__name__}: {exc}"

        if self.device == "cuda" and not self._torch.cuda.is_available():
            self.device = "cpu"

        try:
            import clip  # type: ignore

            self._clip = clip
        except Exception as exc:  # noqa: BLE001
            return f"clip package is required for temporal_consistency: {type(exc).__name__}: {exc}"

        try:
            from PIL import Image  # type: ignore

            self._pil_image = Image
        except Exception as exc:  # noqa: BLE001
            return f"PIL is required for temporal_consistency: {type(exc).__name__}: {exc}"

        if not self.weight_path:
            return (
                "CLIP weight path is required. Set config['clip_weight_path'] to a local CLIP .pt file."
            )

        weight_path = Path(str(self.weight_path)).expanduser().resolve()
        if not weight_path.exists():
            return f"CLIP weight path does not exist: {weight_path}"

        try:
            model, preprocess = self._clip.load(
                str(weight_path),
                device=self.device,
                jit=False,
            )
            model.eval()
            self._clip_model = model
            self._preprocess = preprocess
            return None
        except Exception as exc:  # noqa: BLE001
            self._clip_model = None
            self._preprocess = None
            return f"Failed to load CLIP model: {type(exc).__name__}: {exc}"

    def _extract_clip_features(self, frames: list[Any]) -> Any:
        torch = self._torch
        if torch is None:
            raise RuntimeError("torch is not initialized")
        if self._clip_model is None or self._preprocess is None:
            raise RuntimeError("CLIP model is not initialized")

        images = []
        for frame_rgb in frames:
            pil_img = self._pil_image.fromarray(frame_rgb)
            images.append(self._preprocess(pil_img))

        all_features = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                batch_tensor = torch.stack(batch, dim=0).to(self.device)
                features = self._clip_model.encode_image(batch_tensor)
                features = torch.nn.functional.normalize(features, dim=-1, p=2)
                all_features.append(features)

        if not all_features:
            return None
        return torch.cat(all_features, dim=0)

    def _read_all_frames(self, video_path: str) -> list[Any]:
        cv2 = self._get_cv2()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return []

        frames: list[Any] = []
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        finally:
            cap.release()
        return frames

    def _get_cv2(self) -> Any:
        if self._cv2 is not None:
            return self._cv2

        try:
            import cv2  # type: ignore

            self._cv2 = cv2
            return cv2
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"cv2 is required for temporal_consistency: {exc}") from exc


TemporalConsistency = TemporalConsistencyMetric
TEMPORAL_CONSISTENCY = TemporalConsistencyMetric


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
