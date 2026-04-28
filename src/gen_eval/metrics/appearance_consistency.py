"""DINO-based appearance consistency metric for fixed multi-view videos."""

from __future__ import annotations

import math
import sys
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


class AppearanceConsistencyMetric:
    """Measure DINO-based appearance stability across fixed driving-camera views."""

    name = "appearance_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.camera_videos_key = self.config.get("camera_videos_key", "camera_videos")
        self.device = self.config.get("device", "cuda")
        self.repo_path = (
            self.config.get("repo_path")
            or self.config.get("repo_or_dir")
            or self.config.get("dino_repo_path")
            or self.config.get("local_repo_path")
        )
        self.weight_path = (
            self.config.get("weight_path")
            or self.config.get("weights_path")
            or self.config.get("dino_weight_path")
            or self.config.get("local_save_path")
        )
        self.model_name = self.config.get("model_name", "dino_vitb16")
        self.use_fp16 = bool(self.config.get("use_fp16", False))
        self.strict_load = bool(self.config.get("strict_load", True))
        self.image_size = int(
            self.config.get("image_size", self.config.get("resize", 224))
        )
        self.batch_size = int(self.config.get("batch_size", 16))

        self._torch = None
        self._cv2 = None
        self._torchvision_transforms = None
        self._torchvision_tf = None
        self._pil_image = None
        self._dino_model = None
        self._transform = None

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        evaluated_samples: list[dict[str, Any]] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        valid_scores: list[float] = []

        runtime_status = self._ensure_dino()
        if runtime_status is not None:
            return {
                "metric": self.name,
                "status": "skipped",
                "num_samples": len(samples),
                "valid_sample_count": 0,
                "mean_appearance_consistency_score": None,
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
            score = sample_result.get("appearance_consistency_score")

            if status == "success" and is_finite_number(score):
                evaluated_samples.append(
                    {
                        "sample_id": sample_id,
                        "appearance_consistency_score": float(score),
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
            reason = "No sample produced a valid appearance_consistency_score."

        result: dict[str, Any] = {
            "metric": self.name,
            "status": status,
            "num_samples": len(samples),
            "valid_sample_count": len(valid_scores),
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
                "reason": "No expected camera view produced a valid appearance consistency score.",
            }

        return {
            "sample_id": sample_id,
            "status": "success",
            "appearance_consistency_score": mean_or_none(view_scores),
        }

    def _evaluate_view_video(self, video_path: str) -> float | None:
        path = Path(video_path)
        if not path.exists() or not path.is_file():
            return None

        frames = self._read_all_frames(video_path)
        if len(frames) < 2:
            return None

        features = self._extract_dino_features(frames)
        if features is None or len(features) < 2:
            return None

        return self._compute_appearance_consistency_score(features)

    def _compute_appearance_consistency_score(self, features: Any) -> float | None:
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
                tji_score = float(torch.exp(-0.5 * tji_tensor).item())
            else:
                tji_score = 1.0

            score = 0.5 * clamp01(acm) + 0.5 * clamp01(tji_score)
            if not math.isfinite(score):
                return 0.0
            return clamp01(float(score))

    def _ensure_dino(self) -> str | None:
        if self._dino_model is not None and self._transform is not None:
            return None

        try:
            import torch  # type: ignore

            self._torch = torch
        except Exception as exc:  # noqa: BLE001
            return f"torch is required for appearance_consistency: {type(exc).__name__}: {exc}"

        if self.device == "cuda" and not self._torch.cuda.is_available():
            self.device = "cpu"

        try:
            from PIL import Image  # type: ignore

            self._pil_image = Image
        except Exception as exc:  # noqa: BLE001
            return f"PIL is required for appearance_consistency: {type(exc).__name__}: {exc}"

        try:
            from torchvision import transforms  # type: ignore
            from torchvision.transforms import functional as TF  # type: ignore

            self._torchvision_transforms = transforms
            self._torchvision_tf = TF
        except Exception as exc:  # noqa: BLE001
            return f"torchvision is required for appearance_consistency: {type(exc).__name__}: {exc}"

        if not self.repo_path:
            return (
                "DINO repo path is required. Set config['repo_or_dir'] or "
                "config['dino_repo_path'] to the local DINO repository."
            )

        repo_path = Path(str(self.repo_path)).expanduser().resolve()
        if not repo_path.exists():
            return f"DINO repo path does not exist: {repo_path}"

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        if not self.weight_path:
            return (
                "DINO weights path is required. Set config['weights_path'] or "
                "config['dino_weight_path'] to a local DINO .pth file."
            )

        weight_path = Path(str(self.weight_path)).expanduser().resolve()
        if not weight_path.exists():
            return f"DINO weights path does not exist: {weight_path}"

        try:
            model = self._torch.hub.load(
                str(repo_path),
                self.model_name,
                source="local",
                pretrained=False,
            )
            model.to(self.device)

            state_dict = self._torch.load(str(weight_path), map_location=self.device)
            model.load_state_dict(state_dict, strict=self.strict_load)

            if self.use_fp16 and self.device == "cuda":
                model = model.half()

            model.eval()
            self._dino_model = model
            self._transform = self._build_transform()
            return None
        except Exception as exc:  # noqa: BLE001
            self._dino_model = None
            self._transform = None
            return f"Failed to load DINO model: {type(exc).__name__}: {exc}"

    def _build_transform(self) -> Any:
        transforms = self._torchvision_transforms
        TF = self._torchvision_tf
        if transforms is None or TF is None:
            raise RuntimeError("torchvision transforms are not initialized")

        def robust_to_tensor(x: Any) -> Any:
            torch = self._torch
            if torch is None:
                raise RuntimeError("torch is not initialized")

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
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

    def _extract_dino_features(self, frames: list[Any]) -> Any:
        torch = self._torch
        if torch is None:
            raise RuntimeError("torch is not initialized")
        if self._dino_model is None or self._transform is None:
            raise RuntimeError("DINO model is not initialized")

        images = []
        for frame_rgb in frames:
            pil_img = self._pil_image.fromarray(frame_rgb)
            images.append(self._transform(pil_img))

        features_list = []
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
            raise RuntimeError(f"cv2 is required for appearance_consistency: {exc}") from exc


AppearanceConsistency = AppearanceConsistencyMetric


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
