from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from reference.base import ReferenceGenerator

DEFAULT_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)


def normalize_path(path: str | Path) -> str:
    return os.path.normpath(str(path))


def sanitize_path_part(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) or "unknown"


def read_video_frames(video_path: str | Path, frame_stride: int = 1, max_frames: int | None = None) -> list[np.ndarray]:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    import cv2

    stride = max(1, int(frame_stride))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {path}")

    frames: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            if frame_index % stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                if max_frames is not None and len(frames) >= int(max_frames):
                    break
            frame_index += 1
    finally:
        cap.release()
    return frames


class OpenSeeDAdapter:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.repo_path = config.get("repo_path")
        self.config_path = config.get("config_path")
        self.weight_path = config.get("weight_path")
        self.device = str(config.get("device", "cuda"))
        self.vocabulary = list(config.get("vocabulary", []))
        self.ignore_label = int(config.get("ignore_label", -1))
        self.max_frames = config.get("max_frames")
        self.frame_stride = int(config.get("frame_stride", 1))
        self._model: Any | None = None
        self._predictor: Any | None = None

    @property
    def num_classes(self) -> int:
        return len(self.vocabulary)

    def infer_video(self, video_path: str | Path) -> np.ndarray:
        frames = read_video_frames(video_path=video_path, frame_stride=self.frame_stride, max_frames=self.max_frames)
        if not frames:
            raise ValueError(f"No frames can be read from video: {video_path}")
        return self.infer_frames(frames)

    def infer_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        self._ensure_model()
        masks: list[np.ndarray] = []
        for frame_rgb in frames:
            masks.append(self.infer_frame(frame_rgb))
        return np.stack(masks, axis=0).astype(np.int32, copy=False)

    def infer_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        self._ensure_model()
        raise NotImplementedError(
            "OpenSeeD single-frame inference is not implemented yet. Connect this adapter to your local OpenSeeD predictor and return a [H, W] int32 semantic mask."
        )

    def _ensure_model(self) -> None:
        if self._model is not None or self._predictor is not None:
            return
        if self.repo_path:
            repo_path = str(Path(self.repo_path).expanduser().resolve())
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
        if not self.config_path:
            raise ValueError("OpenSeeD config_path is required.")
        if not self.weight_path:
            raise ValueError("OpenSeeD weight_path is required.")
        if not self.vocabulary:
            raise ValueError("OpenSeeD vocabulary must not be empty.")
        raise NotImplementedError(
            "OpenSeeD model loading is not implemented yet. Load your local OpenSeeD config, checkpoint, and predictor here."
        )


class OpenSeeDReference(ReferenceGenerator):
    name = "openseed_semantic"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.input_key = str(self.config.get("input_key", "camera_videos"))
        self.output_key = str(self.config.get("output_key", "semantic_masks"))
        self.num_classes_key = str(self.config.get("num_classes_key", "semantic_num_classes"))
        self.ignore_label_key = str(self.config.get("ignore_label_key", "semantic_ignore_label"))
        self.repo_path = self.config.get("repo_path")
        self.config_path = self.config.get("config_path")
        self.weight_path = self.config.get("weight_path")
        self.device = str(self.config.get("device", "cuda"))
        self.force = bool(self.config.get("force", False))
        self.vocabulary = list(self.config.get("vocabulary", []))
        expected_camera_views = self.config.get("expected_camera_views")
        if isinstance(expected_camera_views, list) and expected_camera_views:
            self.expected_camera_views = tuple(str(view) for view in expected_camera_views)
        else:
            self.expected_camera_views = DEFAULT_CAMERA_VIEWS
        adapter_config = dict(self.config)
        adapter_config.setdefault("ignore_label", -1)
        self.adapter = OpenSeeDAdapter(adapter_config)

    def prepare_sample(self, sample: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        sample_id = str(sample.get("sample_id") or "unknown")
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"sample {sample_id} metadata must be an object.")

        camera_videos = metadata.get(self.input_key)
        if not isinstance(camera_videos, dict) or not camera_videos:
            raise ValueError(f"sample {sample_id} metadata['{self.input_key}'] must be a non-empty dict.")

        sample_dir = output_dir / self.name / sanitize_path_part(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        semantic_masks: dict[str, str] = {}
        for view in self.expected_camera_views:
            video_path = camera_videos.get(view)
            if not video_path:
                continue

            mask_path = sample_dir / f"{view}.npy"
            if mask_path.exists() and not self.force:
                semantic_masks[view] = normalize_path(mask_path)
                continue

            if not mask_path.exists() or self.force:
                masks = self.adapter.infer_video(video_path)
                np.save(mask_path, masks.astype(np.int32, copy=False))

            semantic_masks[view] = normalize_path(mask_path)

        if not semantic_masks:
            raise ValueError(f"sample {sample_id} produced no semantic mask files.")

        return {
            self.output_key: semantic_masks,
            self.num_classes_key: len(self.vocabulary),
            self.ignore_label_key: -1,
        }
