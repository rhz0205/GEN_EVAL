from __future__ import annotations

import contextlib
import importlib
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from reference.base import ReferenceGenerator

DEFAULT_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)

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


def sanitize_path_part(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) or "unknown"


def normalize_path(path: str | Path) -> str:
    return os.path.normpath(str(path))


def load_torch_state(path: str | Path, *, map_location: str) -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            return torch.load(str(path), map_location=map_location)


class DepthReference(ReferenceGenerator):
    name = "depthanything_depth"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.input_key = str(self.config.get("input_key", "camera_videos"))
        self.output_key = str(self.config.get("output_key", "depth_maps"))
        self.num_frames_key = str(self.config.get("num_frames_key", "depth_num_frames"))
        self.device = str(self.config.get("device", "cuda"))
        self.encoder = str(self.config.get("encoder", "vits"))
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
        self.force = bool(self.config.get("force", False))
        expected_camera_views = self.config.get("expected_camera_views")
        if isinstance(expected_camera_views, list) and expected_camera_views:
            self.expected_camera_views = tuple(str(view) for view in expected_camera_views)
        else:
            self.expected_camera_views = DEFAULT_CAMERA_VIEWS
        self._depth_engine: Any | None = None

    def prepare_sample(self, sample: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        sample_id = str(sample.get("sample_id") or "unknown")
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"sample {sample_id} metadata must be an object.")

        camera_videos = metadata.get(self.input_key)
        if not isinstance(camera_videos, dict) or not camera_videos:
            raise ValueError(f"sample {sample_id} metadata['{self.input_key}'] must be a non-empty dict.")

        runtime_error = self._ensure_runtime()
        if runtime_error is not None:
            raise RuntimeError(runtime_error)

        sample_dir = output_dir / self.name / sanitize_path_part(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        depth_maps: dict[str, str] = {}
        depth_num_frames: dict[str, int] = {}

        for view in self.expected_camera_views:
            video_path = camera_videos.get(view)
            if not video_path:
                continue

            depth_path = sample_dir / f"{view}.npy"
            if depth_path.exists() and not self.force:
                stored = np.load(depth_path, mmap_mode="r")
                depth_maps[view] = normalize_path(depth_path)
                depth_num_frames[view] = int(stored.shape[0]) if stored.ndim > 0 else 0
                continue

            frames = self._read_all_frames(str(video_path))
            if len(frames) < 2:
                raise ValueError(f"video {video_path} has fewer than 2 readable frames.")

            depths = self._infer_depth(frames)
            if depths is None:
                raise RuntimeError(f"depth inference returned no result for video: {video_path}")

            depth_array = np.asarray(depths)
            np.save(depth_path, depth_array)
            depth_maps[view] = normalize_path(depth_path)
            depth_num_frames[view] = int(depth_array.shape[0]) if depth_array.ndim > 0 else 0

        if not depth_maps:
            raise ValueError(f"sample {sample_id} produced no depth reference files.")

        return {
            self.output_key: depth_maps,
            self.num_frames_key: depth_num_frames,
        }

    def _ensure_runtime(self) -> str | None:
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        return self._ensure_depth_engine()

    def _ensure_depth_engine(self) -> str | None:
        if self._depth_engine is not None:
            return None
        if self.encoder not in MODEL_CONFIGS:
            return f"Unsupported depth encoder: {self.encoder}. Expected one of {sorted(MODEL_CONFIGS)}."

        if self.repo_path:
            repo_path = str(Path(str(self.repo_path)).expanduser().resolve())
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        try:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
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
            state = load_torch_state(checkpoint_path, map_location="cpu")
            engine.load_state_dict(state, strict=True)
            self._depth_engine = engine.to(self.device).eval()
            return None
        except Exception as exc:
            self._depth_engine = None
            return f"Failed to load VideoDepthAnything: {type(exc).__name__}: {exc}"

    def _infer_depth(self, frames: list[Any]) -> Any | None:
        if self._depth_engine is None:
            raise RuntimeError("depth engine is not initialized")

        frame_array = np.asarray(frames)
        if frame_array.size == 0:
            return None

        context = open(os.devnull, "w") if self.silence_depth_stdout else None
        if context is None:
            with torch.autocast(
                device_type=self.device,
                dtype=torch.float16,
                enabled=(self.device == "cuda" and not self.depth_fp32),
            ):
                depths, _ = self._depth_engine.infer_video_depth(
                    frame_array,
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
                            frame_array,
                            self.target_fps,
                            input_size=self.input_size,
                            target_size=self.target_size,
                            device=self.device,
                            fp32=self.depth_fp32,
                        )
        return depths

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
