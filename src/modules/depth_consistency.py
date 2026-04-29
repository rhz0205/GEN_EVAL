from __future__ import annotations

import contextlib
import importlib
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from modules.base import BaseModule

MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}

EXPECTED_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)


class DepthConsistency(BaseModule):
    name = "depth_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.camera_videos_key = self.config.get("camera_videos_key", "camera_videos")
        self.device = self.config.get("device", "cuda")
        self.encoder = self.config.get("encoder", "vits")
        self.weight_path = self.config.get("weight_path", "pretrained_models/depth")
        self.depth_checkpoint_path = self.config.get("depth_checkpoint_path")
        self.repo_path = self.config.get("repo_path")
        self.video_depth_module = self.config.get("video_depth_module", "video_depth_anything.video_depth")
        self.video_depth_class = self.config.get("video_depth_class", "VideoDepthAnything")
        self.target_fps = int(self.config.get("target_fps", 12))
        self.max_res = int(self.config.get("max_res", 400))
        self.input_size = int(self.config.get("input_size", 400))
        self.target_size = tuple(self.config.get("target_size", [450, 800]))
        self.depth_fp32 = bool(self.config.get("depth_fp32", False))
        self.silence_depth_stdout = bool(self.config.get("silence_depth_stdout", True))
        self.model_path = self.config.get("model_path", "pretrained_models/dinov2")
        self.batch_size = int(self.config.get("batch_size", 16))
        self.depth_l2_scale = float(self.config.get("depth_l2_scale", 1.0))
        self._depth_engine: Any | None = None
        self._dino_processor: Any | None = None
        self._dino_model: Any | None = None

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        runtime_error = self._ensure_runtime()
        if runtime_error is not None:
            return {
                "metric": self.name,
                "status": "skipped",
                "num_samples": len(samples),
                "valid_sample_count": 0,
                "skipped_sample_count": 0,
                "failed_sample_count": 0,
                "mean_depth_consistency_score": None,
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
            score = sample_result.get("depth_consistency_score")

            if status == "success" and is_finite_number(score):
                evaluated_samples.append(
                    {
                        "sample_id": sample_id,
                        "depth_consistency_score": float(score),
                    }
                )
                valid_scores.append(float(score))
            elif status == "skipped":
                skipped_samples.append(simplify_sample_result(sample_result))
            elif status == "failed":
                failed_samples.append(simplify_sample_result(sample_result))

        mean_score = mean_or_none(valid_scores)
        if mean_score is not None:
            status = "success"
            reason = None
        else:
            status = "failed" if failed_samples else "skipped"
            reason = "No sample produced a valid depth_consistency_score."

        result: dict[str, Any] = {
            "metric": self.name,
            "status": status,
            "num_samples": len(samples),
            "valid_sample_count": len(valid_scores),
            "skipped_sample_count": len(skipped_samples),
            "failed_sample_count": len(failed_samples),
            "mean_depth_consistency_score": mean_score,
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

        normalized_videos = {str(view): str(path) for view, path in camera_videos.items() if path is not None}
        missing_views = [view for view in EXPECTED_CAMERA_VIEWS if view not in normalized_videos]
        if missing_views:
            return skipped_result(sample_id, f"Missing expected camera views: {', '.join(missing_views)}.")

        view_scores: list[float] = []
        for view in EXPECTED_CAMERA_VIEWS:
            view_score = self._evaluate_view_video(normalized_videos[view])
            if is_finite_number(view_score):
                view_scores.append(float(view_score))

        if not view_scores:
            return skipped_result(sample_id, "No expected camera view produced a valid depth consistency score.")

        return {
            "sample_id": sample_id,
            "status": "success",
            "depth_consistency_score": mean_or_none(view_scores),
        }

    def _evaluate_view_video(self, video_path: str) -> float | None:
        path = Path(video_path)
        if not path.exists() or not path.is_file():
            return None
        frames = self._read_all_frames(video_path)
        if len(frames) < 2:
            return None
        depths = self._infer_depth(frames)
        if depths is None or len(depths) < 2:
            return None
        depth_rgb = self._render_depth(depths)
        avg_l2 = self._compute_depth_l2(depth_rgb)
        if avg_l2 is None:
            return None
        return self._l2_to_score(avg_l2)

    def _ensure_runtime(self) -> str | None:
        device_error = self._ensure_device()
        if device_error is not None:
            return device_error
        depth_error = self._ensure_depth_engine()
        if depth_error is not None:
            return depth_error
        dino_error = self._ensure_dinov2()
        if dino_error is not None:
            return dino_error
        return None

    def _ensure_device(self) -> str | None:
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        return None

    def _ensure_depth_engine(self) -> str | None:
        if self._depth_engine is not None:
            return None
        if self.encoder not in MODEL_CONFIGS:
            return f"Unsupported depth encoder: {self.encoder}. Expected one of {sorted(MODEL_CONFIGS)}."

        if self.repo_path:
            repo_path = str(Path(self.repo_path).expanduser().resolve())
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        try:
            module = importlib.import_module(self.video_depth_module)
            depth_cls = getattr(module, self.video_depth_class)
        except Exception as exc:
            return f"Failed to import {self.video_depth_module}.{self.video_depth_class}: {type(exc).__name__}: {exc}"

        if self.depth_checkpoint_path:
            checkpoint_path = Path(str(self.depth_checkpoint_path)).expanduser().resolve()
        else:
            checkpoint_path = Path(str(self.weight_path)).expanduser().resolve() / f"metric_video_depth_anything_{self.encoder}.pth"

        if not checkpoint_path.exists():
            return f"Depth checkpoint not found: {checkpoint_path}"

        try:
            engine = depth_cls(**MODEL_CONFIGS[self.encoder])
            state = torch.load(str(checkpoint_path), map_location="cpu")
            engine.load_state_dict(state, strict=True)
            self._depth_engine = engine.to(self.device).eval()
            return None
        except Exception as exc:
            self._depth_engine = None
            return f"Failed to load VideoDepthAnything: {type(exc).__name__}: {exc}"

    def _ensure_dinov2(self) -> str | None:
        if self._dino_model is not None and self._dino_processor is not None:
            return None
        model_path = Path(str(self.model_path)).expanduser().resolve()
        if not model_path.exists():
            return f"DINOv2 local model path not found: {model_path}"

        try:
            processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
            model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(self.device).eval()
            self._dino_processor = processor
            self._dino_model = model
            return None
        except Exception as exc:
            self._dino_processor = None
            self._dino_model = None
            return f"Failed to load DINOv2 from {model_path}: {type(exc).__name__}: {exc}"

    def _infer_depth(self, frames: list[Any]) -> Any | None:
        if self._depth_engine is None:
            raise RuntimeError("depth engine is not initialized")

        context = open(os.devnull, "w") if self.silence_depth_stdout else None
        if context is None:
            with torch.autocast(
                device_type=self.device,
                dtype=torch.float16,
                enabled=(self.device == "cuda" and not self.depth_fp32),
            ):
                depths, _ = self._depth_engine.infer_video_depth(
                    frames,
                    self.target_fps,
                    input_size=self.input_size,
                    target_size=self.target_size,
                    device=self.device,
                    fp32=self.depth_fp32,
                )
        else:
            with context as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    with torch.autocast(
                        device_type=self.device,
                        dtype=torch.float16,
                        enabled=(self.device == "cuda" and not self.depth_fp32),
                    ):
                        depths, _ = self._depth_engine.infer_video_depth(
                            frames,
                            self.target_fps,
                            input_size=self.input_size,
                            target_size=self.target_size,
                            device=self.device,
                            fp32=self.depth_fp32,
                        )
        return depths

    def _render_depth(self, depths: Any) -> Any:
        frames = np.asarray(depths)
        frames = np.where(np.isfinite(frames), frames, 0.0)
        mn = float(frames.min()) if frames.size else 0.0
        mx = float(frames.max()) if frames.size else 0.0
        if mx > mn:
            depth_norm = (frames - mn) / (mx - mn)
        else:
            depth_norm = np.zeros_like(frames, dtype=np.float32)

        cmap = cm.get_cmap("inferno")
        return (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)

    def _compute_depth_l2(self, depth_frames_rgb: Any) -> float | None:
        if depth_frames_rgb is None or len(depth_frames_rgb) < 2:
            return None

        imgs = [Image.fromarray(frame).convert("RGB") for frame in depth_frames_rgb]
        all_feats: list[Any] = []

        for start in range(0, len(imgs), self.batch_size):
            batch = imgs[start : start + self.batch_size]
            inputs = self._dino_processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self._dino_model(**inputs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output.detach().cpu()
            else:
                feats = outputs.last_hidden_state[:, 0].detach().cpu()
            all_feats.append(feats)

        if not all_feats:
            return None

        feats = torch.cat(all_feats, dim=0)
        diffs = feats[1:] - feats[:-1]
        l2_distances = torch.norm(diffs, p=2, dim=-1)
        return float(l2_distances.mean().item())

    def _l2_to_score(self, avg_l2: float) -> float:
        score = math.exp(-self.depth_l2_scale * float(avg_l2))
        return clamp01(score)

    def _read_all_frames(self, video_path: str) -> list[Any]:
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
                frames.append(self._resize_max_res(frame_rgb))
        finally:
            cap.release()
        return frames

    def _resize_max_res(self, frame_rgb: Any) -> Any:
        if self.max_res <= 0:
            return frame_rgb
        height, width = frame_rgb.shape[:2]
        max_side = max(height, width)
        if max_side <= self.max_res:
            return frame_rgb
        scale = self.max_res / float(max_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)


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
