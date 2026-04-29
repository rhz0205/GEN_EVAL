from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from modules.base import BaseModule

EXPECTED_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)


class AppearanceConsistency(BaseModule):
    name = "appearance_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.camera_videos_key = self.config.get("camera_videos_key", "camera_videos")
        self.expected_camera_views = tuple(self.config.get("expected_camera_views", EXPECTED_CAMERA_VIEWS))
        self.device = self.config.get("device", "cuda")
        self.repo_path = self.config.get("repo_path")
        self.weight_path = self.config.get("weight_path")
        self.model_name = self.config.get("model_name", "dino_vitb16")
        self.use_fp16 = bool(self.config.get("use_fp16", False))
        self.strict_load = bool(self.config.get("strict_load", True))
        self.image_size = int(self.config.get("image_size", 224))
        self.batch_size = int(self.config.get("batch_size", 16))
        self.eps = float(self.config.get("eps", 1e-8))
        self._dino_model: Any | None = None
        self._transform: Any | None = None

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        runtime_error = self._ensure_dino()
        if runtime_error is not None:
            return {
                "metric": self.name,
                "status": "skipped",
                "num_samples": len(samples),
                "valid_sample_count": 0,
                "skipped_sample_count": 0,
                "failed_sample_count": 0,
                "mean_appearance_consistency_score": None,
                "details": {
                    "evaluated_samples": [],
                    "skipped_samples": [],
                    "failed_samples": [],
                },
                "reason": runtime_error,
            }

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
            score = sample_result.get("appearance_consistency_score")
            if status == "success" and is_finite_number(score):
                valid_score = float(score)
                valid_scores.append(valid_score)
                evaluated_samples.append(
                    {
                        "sample_id": sample_id,
                        "appearance_consistency_score": valid_score,
                    }
                )
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
            reason = "No valid appearance consistency score."

        result: dict[str, Any] = {
            "metric": self.name,
            "status": status,
            "num_samples": len(samples),
            "valid_sample_count": len(valid_scores),
            "skipped_sample_count": len(skipped_samples),
            "failed_sample_count": len(failed_samples),
            "mean_appearance_consistency_score": mean_score,
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
            return skipped_result(sample_id, f"metadata['{self.camera_videos_key}'] must be a non-empty dict.")

        generated_videos = {str(view): str(path) for view, path in camera_videos.items() if path is not None}
        missing_views = [view for view in self.expected_camera_views if view not in generated_videos]
        if missing_views:
            return skipped_result(sample_id, f"Missing camera views: {', '.join(missing_views)}.")

        view_scores: list[float] = []
        for view in self.expected_camera_views:
            view_score = self._evaluate_view_video(generated_videos[view])
            if is_finite_number(view_score):
                view_scores.append(float(view_score))

        if not view_scores:
            return skipped_result(sample_id, "No valid video score.")

        return {
            "sample_id": sample_id,
            "status": "success",
            "appearance_consistency_score": mean_or_none(view_scores),
        }

    def _evaluate_view_video(self, video_path: str) -> float | None:
        path = Path(video_path)
        if not path.is_file():
            return None
        frames = self._read_all_frames(path)
        if len(frames) < 2:
            return None
        features = self._extract_dino_features(frames)
        if features is None or len(features) < 2:
            return None
        return self._compute_appearance_consistency_score(features)

    def _compute_appearance_consistency_score(self, generated_features: Any) -> float | None:
        if generated_features is None or len(generated_features) < 2:
            return None
        with torch.no_grad():
            acm = self._compute_acm(generated_features)
            tji = self._compute_tji(generated_features)
            score = acm / (1.0 + tji)
            return clamp01(float(score))

    def _compute_acm(self, features: Any) -> float:
        adjacent_similarities = torch.nn.functional.cosine_similarity(features[:-1], features[1:], dim=-1)
        adjacent_similarities = torch.clamp(adjacent_similarities, min=0.0)
        return clamp01(float(adjacent_similarities.mean().item()))

    def _compute_tji(self, features: Any) -> float:
        if len(features) < 3:
            return 0.0
        velocity = (features[1:] - features[:-1]).norm(dim=1)
        acceleration = (features[2:] - 2 * features[1:-1] + features[:-2]).norm(dim=1)
        denominator = 0.5 * (velocity[1:] + velocity[:-1]) + self.eps
        jitter = (acceleration / denominator).mean()
        return max(0.0, float(jitter.item()))

    def _ensure_dino(self) -> str | None:
        if self._dino_model is not None and self._transform is not None:
            return None
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        if not self.repo_path:
            return "Missing DINO repo path: set config['repo_path']."
        repo_path = Path(str(self.repo_path)).expanduser().resolve()
        if not repo_path.exists():
            return f"DINO repo path does not exist: {repo_path}"

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        if not self.weight_path:
            return "Missing DINO weight path: set config['weight_path']."
        weight_path = Path(str(self.weight_path)).expanduser().resolve()
        if not weight_path.exists():
            return f"DINO weight path does not exist: {weight_path}"

        try:
            model = torch.hub.load(str(repo_path), self.model_name, source="local", pretrained=False)
            model.to(self.device)
            state_dict = torch.load(str(weight_path), map_location=self.device)
            model.load_state_dict(state_dict, strict=self.strict_load)
            if self.use_fp16 and self.device == "cuda":
                model = model.half()
            model.eval()
            self._dino_model = model
            self._transform = self._build_transform()
        except Exception as exc:
            self._dino_model = None
            self._transform = None
            return f"Failed to load DINO model: {exc}"

        return None

    def _build_transform(self) -> Any:
        def robust_to_tensor(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                if x.dtype == torch.uint8:
                    return x.float() / 255.0
                return x
            return TF.to_tensor(x)

        return transforms.Compose(
            [
                transforms.Lambda(robust_to_tensor),
                transforms.Resize(
                    self.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(self.image_size),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def _extract_dino_features(self, frames: list[Any]) -> Any:
        if self._dino_model is None or self._transform is None:
            raise RuntimeError("DINO model is not initialized")

        images = [self._transform(Image.fromarray(frame_rgb)) for frame_rgb in frames]
        features_list: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                batch_tensor = torch.stack(batch, dim=0).to(self.device)
                if self.use_fp16 and self.device == "cuda":
                    batch_tensor = batch_tensor.half()

                features = self._dino_model(batch_tensor)
                features = torch.nn.functional.normalize(features, dim=-1, p=2)
                features_list.append(features)

        if not features_list:
            return None
        return torch.cat(features_list, dim=0)

    def _read_all_frames(self, video_path: Path) -> list[Any]:
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
