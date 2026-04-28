"""Depth-based temporal consistency metric."""

from __future__ import annotations

import contextlib
import importlib
import math
import os
import sys
from pathlib import Path
from typing import Any

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

class DepthConsistencyMetric:
    """Measure temporal consistency from estimated depth dynamics."""

    name = "depth_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

        # Current recommended mode.
        self.mode = self.config.get("mode", "self")

        # Runtime.
        self.device = self.config.get("device", "cuda")

        # Video-Depth-Anything settings.
        self.encoder = self.config.get("encoder", "vits")
        self.weight_path = (
            self.config.get("weight_path")
            or self.config.get("pretrained_model_path")
            or self.config.get("depth_model_dir")
            or "pretrained_models/depth"
        )
        self.depth_checkpoint_path = (
            self.config.get("depth_checkpoint_path")
            or self.config.get("depth_model_path")
        )

        # Path that makes the import below available.
        # Example: /di/group/renhongze/GEN_EVAL/src
        # so that import gen_eval.third_party.video_depth_anything.video_depth works.
        self.repo_path = self.config.get("repo_path") or self.config.get("video_depth_repo_path")
        self.video_depth_module = self.config.get(
            "video_depth_module",
            "gen_eval.third_party.video_depth_anything.video_depth",
        )
        self.video_depth_class = self.config.get("video_depth_class", "VideoDepthAnything")

        self.target_fps = int(self.config.get("target_fps", 12))
        self.max_res = int(self.config.get("max_res", 400))
        self.input_size = int(self.config.get("input_size", 400))
        self.target_size = tuple(self.config.get("target_size", [450, 800]))
        self.depth_fp32 = bool(self.config.get("depth_fp32", False))
        self.silence_depth_stdout = bool(self.config.get("silence_depth_stdout", True))

        # DINOv2 settings.
        self.model_path = (
            self.config.get("model_path")
            or self.config.get("dinov2_model_path")
            or self.config.get("dino_model_path")
            or self.config.get("dino_path")
            or "pretrained_models/dinov2"
        )
        self.batch_size = int(self.config.get("batch_size", self.config.get("dino_batch_size", 16)))

        # Frame sampling.
        self.num_frames = int(self.config.get("num_frames", 8))
        self.frame_positions = self.config.get("frame_positions")
        self.min_frames = int(self.config.get("min_frames", 3))

        # View handling.
        self.use_all_views = bool(self.config.get("use_all_views", True))
        self.camera_videos_key = self.config.get("camera_videos_key", "camera_videos")
        self.keep_front_tele = bool(self.config.get("keep_front_tele", False))
        self.exclude_views = set(self.config.get("exclude_views", []))
        if not self.keep_front_tele:
            self.exclude_views.add("camera_front_tele")

        # Score conversion.
        # raw_l2 is lower-is-better. score is higher-is-better.
        # score = exp(-depth_l2_scale * avg_l2_distance)
        self.depth_l2_scale = float(self.config.get("depth_l2_scale", 1.0))
        self.score_key = self.config.get("score_key", "depth_consistency_score")

        # Optional depth visualization.
        self.save_depth_visualizations = bool(
            self.config.get("save_visualizations", self.config.get("save_depth_visualizations", False))
        )
        self.depth_visualization_dir = self.config.get(
            "depth_visualization_dir",
            "outputs/depth_visualizations",
        )
        self.max_depth_visualizations = int(self.config.get("max_depth_visualizations", 20))
        self._depth_visualization_count = 0

        # Lazy runtime objects.
        self._torch = None
        self._np = None
        self._cv2 = None
        self._cm = None
        self._imageio = None
        self._pil_image = None

        self._depth_engine = None
        self._dino_processor = None
        self._dino_model = None

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        """Evaluate depth consistency over manifest samples."""
        evaluated_samples: list[dict[str, Any]] = []
        valid_scores: list[float] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []

        runtime_status = self._ensure_runtime()
        if runtime_status is not None:
            return {
                "metric": self.name,
                "score": None,
                "num_samples": 0,
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
                "status": "skipped",
                "reason": runtime_status,
            }

        for sample in samples:
            sample_id = getattr(sample, "sample_id", None) or "unknown"

            try:
                if self.mode != "self":
                    sample_result = {
                        "sample_id": sample_id,
                        "metric": self.name,
                        "mode": self.mode,
                        "score": None,
                        "status": "skipped",
                        "reason": f"Unsupported depth_consistency mode: {self.mode}",
                    }
                else:
                    sample_result = self._evaluate_self_sample(sample)

            except Exception as exc:  # noqa: BLE001
                sample_result = {
                    "sample_id": sample_id,
                    "metric": self.name,
                    "mode": self.mode,
                    "score": None,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }

            evaluated_samples.append(sample_result)

            status = sample_result.get("status")
            score = sample_result.get("score")

            if status == "success" and isinstance(score, (int, float)) and math.isfinite(float(score)):
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

        if valid_scores:
            final_score = float(sum(valid_scores) / len(valid_scores))
            status = "success"
            reason = None
        else:
            final_score = None
            status = "skipped" if not failed_samples else "failed"
            reason = "No sample produced a valid depth consistency score."

        result = {
            "metric": self.name,
            "score": final_score,
            "num_samples": len(valid_scores),
            "details": {
                "evaluated_samples": evaluated_samples,
                "skipped_samples": skipped_samples,
                "failed_samples": failed_samples,
            },
            "status": status,
        }

        if reason:
            result["reason"] = reason

        return result

    # ------------------------------------------------------------------
    # Sample-level evaluation
    # ------------------------------------------------------------------

    def _evaluate_self_sample(self, sample: Any) -> dict[str, Any]:
        sample_id = getattr(sample, "sample_id", None) or "unknown"

        if self.use_all_views:
            return self._evaluate_all_views(sample)

        video_path = getattr(sample, "generated_video", None)
        if not video_path:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "self",
                "score": None,
                "status": "skipped",
                "reason": "sample.generated_video is missing.",
            }

        video_result = self._evaluate_single_video(str(video_path), sample_id=sample_id, view_name=None)

        if video_result.get("status") != "success":
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "self",
                "score": None,
                "status": video_result.get("status", "skipped"),
                "reason": video_result.get("reason", "single video evaluation failed"),
                "video_result": video_result,
            }

        score = self._select_score(video_result)

        return {
            "sample_id": sample_id,
            "metric": self.name,
            "mode": "self",
            "score": score,
            "status": "success",
            "video_path": str(video_path),
            "video_result": video_result,
        }

    def _evaluate_all_views(self, sample: Any) -> dict[str, Any]:
        sample_id = getattr(sample, "sample_id", None) or "unknown"
        metadata = getattr(sample, "metadata", None) or {}
        camera_videos = metadata.get(self.camera_videos_key)

        if not isinstance(camera_videos, dict) or not camera_videos:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "self",
                "score": None,
                "status": "skipped",
                "reason": "metadata['camera_videos'] is required when use_all_views=true.",
            }

        view_results: dict[str, Any] = {}
        view_scores: list[float] = []
        view_l2s: list[float] = []

        for view, path in sorted(camera_videos.items()):
            view = str(view)
            if view in self.exclude_views:
                continue

            result = self._evaluate_single_video(str(path), sample_id=sample_id, view_name=view)
            view_results[view] = result

            if result.get("status") == "success":
                score = self._select_score(result)
                raw_l2 = result.get("avg_l2_distance")
                if math.isfinite(score):
                    view_scores.append(score)
                if isinstance(raw_l2, (int, float)) and math.isfinite(float(raw_l2)):
                    view_l2s.append(float(raw_l2))

        if not view_scores:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "self",
                "score": None,
                "status": "skipped",
                "reason": "No camera view produced a valid depth consistency score.",
                "view_results": view_results,
            }

        score = float(sum(view_scores) / len(view_scores))
        avg_l2 = float(sum(view_l2s) / len(view_l2s)) if view_l2s else None

        return {
            "sample_id": sample_id,
            "metric": self.name,
            "mode": "self",
            "score": score,
            "status": "success",
            "num_views": len(view_scores),
            "avg_l2_distance": avg_l2,
            "view_results": view_results,
        }

    def _select_score(self, result: dict[str, Any]) -> float:
        value = result.get(self.score_key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)

        for key in ("depth_consistency_score", "score"):
            value = result.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)

        return 0.0

    # ------------------------------------------------------------------
    # Video-level evaluation
    # ------------------------------------------------------------------

    def _evaluate_single_video(
        self,
        video_path: str,
        sample_id: str,
        view_name: str | None,
    ) -> dict[str, Any]:
        path = Path(video_path)

        if not path.exists():
            return {
                "video_path": video_path,
                "status": "skipped",
                "reason": "video path does not exist",
            }

        if not path.is_file():
            return {
                "video_path": video_path,
                "status": "skipped",
                "reason": "video path is not a file",
            }

        frames, frame_indices = self._read_sampled_frames(video_path)
        if len(frames) < self.min_frames:
            return {
                "video_path": video_path,
                "status": "skipped",
                "reason": f"not enough readable frames: {len(frames)} < {self.min_frames}",
                "sampled_frame_indices": frame_indices,
            }

        depths = self._infer_depth(frames)
        if depths is None or len(depths) < self.min_frames:
            return {
                "video_path": video_path,
                "status": "skipped",
                "reason": "depth inference returned insufficient frames",
                "sampled_frame_indices": frame_indices,
            }

        depth_rgb = self._render_depth(depths)

        if self.save_depth_visualizations:
            self._maybe_save_depth_visualization(
                depth_rgb=depth_rgb,
                sample_id=sample_id,
                view_name=view_name or "generated_video",
            )

        avg_l2 = self._compute_depth_l2(depth_rgb)
        if avg_l2 is None:
            return {
                "video_path": video_path,
                "status": "skipped",
                "reason": "failed to compute DINOv2 depth L2 distance",
                "sampled_frame_indices": frame_indices,
            }

        consistency_score = self._l2_to_score(avg_l2)

        return {
            "video_path": video_path,
            "status": "success",
            "score": consistency_score,
            "depth_consistency_score": consistency_score,
            "avg_l2_distance": float(avg_l2),
            "num_frames": len(depth_rgb),
            "num_transitions": max(0, len(depth_rgb) - 1),
            "sampled_frame_indices": frame_indices,
        }

    # ------------------------------------------------------------------
    # Runtime initialization
    # ------------------------------------------------------------------

    def _ensure_runtime(self) -> str | None:
        torch_status = self._ensure_torch()
        if torch_status is not None:
            return torch_status

        depth_status = self._ensure_depth_engine()
        if depth_status is not None:
            return depth_status

        dino_status = self._ensure_dinov2()
        if dino_status is not None:
            return dino_status

        return None

    def _ensure_torch(self) -> str | None:
        if self._torch is not None:
            return None

        try:
            import torch  # type: ignore

            self._torch = torch
        except Exception as exc:  # noqa: BLE001
            return f"torch is required for depth_consistency: {type(exc).__name__}: {exc}"

        if self.device == "cuda" and not self._torch.cuda.is_available():
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
        except Exception as exc:  # noqa: BLE001
            return (
                "Failed to import VideoDepthAnything. Set config['video_depth_repo_path'] "
                "to a local path that makes "
                f"{self.video_depth_module}.{self.video_depth_class} importable. "
                f"Error: {type(exc).__name__}: {exc}"
            )

        if self.depth_checkpoint_path:
            ckpt_path = Path(str(self.depth_checkpoint_path)).expanduser().resolve()
        else:
            ckpt_path = (
                Path(str(self.weight_path)).expanduser().resolve()
                / f"metric_video_depth_anything_{self.encoder}.pth"
            )

        if not ckpt_path.exists():
            return f"Depth checkpoint not found: {ckpt_path}"

        try:
            engine = depth_cls(**MODEL_CONFIGS[self.encoder])
            state = self._torch.load(str(ckpt_path), map_location="cpu")
            engine.load_state_dict(state, strict=True)
            engine = engine.to(self.device).eval()
            self._depth_engine = engine
            return None

        except Exception as exc:  # noqa: BLE001
            self._depth_engine = None
            return f"Failed to load VideoDepthAnything: {type(exc).__name__}: {exc}"

    def _ensure_dinov2(self) -> str | None:
        if self._dino_model is not None and self._dino_processor is not None:
            return None

        model_path = Path(str(self.model_path)).expanduser().resolve()
        if not model_path.exists():
            return f"DINOv2 local model path not found: {model_path}"

        try:
            from transformers import AutoImageProcessor, AutoModel  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return f"transformers is required for DINOv2: {type(exc).__name__}: {exc}"

        try:
            processor = AutoImageProcessor.from_pretrained(
                str(model_path),
                local_files_only=True,
            )
            model = AutoModel.from_pretrained(
                str(model_path),
                local_files_only=True,
            ).to(self.device).eval()

            self._dino_processor = processor
            self._dino_model = model
            return None

        except Exception as exc:  # noqa: BLE001
            self._dino_processor = None
            self._dino_model = None
            return f"Failed to load DINOv2 from {model_path}: {type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # Depth inference and scoring
    # ------------------------------------------------------------------

    def _infer_depth(self, frames: list[Any]) -> Any | None:
        if self._depth_engine is None:
            raise RuntimeError("depth engine is not initialized")

        torch = self._torch
        if torch is None:
            raise RuntimeError("torch is not initialized")

        try:
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

        except Exception:
            raise

    def _render_depth(self, depths: Any) -> Any:
        np = self._get_np()
        cm = self._get_cm()

        frames = np.asarray(depths)
        frames = np.where(np.isfinite(frames), frames, 0.0)

        mn = float(frames.min()) if frames.size else 0.0
        mx = float(frames.max()) if frames.size else 0.0

        if mx > mn:
            depth_norm = (frames - mn) / (mx - mn)
        else:
            depth_norm = np.zeros_like(frames, dtype=np.float32)

        cmap = cm.get_cmap("inferno")
        depth_rgb = (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)
        return depth_rgb

    def _compute_depth_l2(self, depth_frames_rgb: Any) -> float | None:
        if depth_frames_rgb is None or len(depth_frames_rgb) < 2:
            return None

        torch = self._torch
        if torch is None:
            raise RuntimeError("torch is not initialized")

        pil_image = self._get_pil_image()

        imgs = [pil_image.fromarray(frame).convert("RGB") for frame in depth_frames_rgb]
        all_feats = []

        for start in range(0, len(imgs), self.batch_size):
            batch = imgs[start : start + self.batch_size]
            inputs = self._dino_processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._dino_model(**inputs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feats = outputs.pooler_output.detach().cpu()
            else:
                # Fallback for models without pooler_output.
                feats = outputs.last_hidden_state[:, 0].detach().cpu()

            all_feats.append(feats)

        feats = torch.cat(all_feats, dim=0)
        diffs = feats[1:] - feats[:-1]
        l2_distances = torch.norm(diffs, p=2, dim=-1)

        return float(l2_distances.mean().item())

    def _l2_to_score(self, avg_l2: float) -> float:
        score = math.exp(-self.depth_l2_scale * float(avg_l2))
        return clamp01(score)

    # ------------------------------------------------------------------
    # Video reading and visualization
    # ------------------------------------------------------------------

    def _read_sampled_frames(self, video_path: str) -> tuple[list[Any], list[int]]:
        cv2 = self._get_cv2()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return [], []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            cap.release()
            return [], []

        indices = self._sample_frame_indices(frame_count)

        frames = []
        valid_indices = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = self._resize_max_res(frame_rgb)
            frames.append(frame_rgb)
            valid_indices.append(idx)

        cap.release()
        return frames, valid_indices

    def _sample_frame_indices(self, frame_count: int) -> list[int]:
        if frame_count <= 0:
            return []

        if self.frame_positions:
            indices = []
            for pos in self.frame_positions:
                try:
                    p = float(pos)
                except (TypeError, ValueError):
                    continue
                p = max(0.0, min(1.0, p))
                indices.append(int(round(p * (frame_count - 1))))
            return sorted(set(indices))

        n = max(1, self.num_frames)
        if n == 1:
            return [frame_count // 2]

        indices = []
        for i in range(n):
            pos = (i + 1) / (n + 1)
            idx = int(round(pos * (frame_count - 1)))
            indices.append(idx)

        return sorted(set(indices))

    def _resize_max_res(self, frame_rgb: Any) -> Any:
        if self.max_res <= 0:
            return frame_rgb

        cv2 = self._get_cv2()
        h, w = frame_rgb.shape[:2]
        max_side = max(h, w)

        if max_side <= self.max_res:
            return frame_rgb

        scale = self.max_res / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _maybe_save_depth_visualization(
        self,
        depth_rgb: Any,
        sample_id: str,
        view_name: str,
    ) -> None:
        if not self.save_depth_visualizations:
            return

        if self._depth_visualization_count >= self.max_depth_visualizations:
            return

        try:
            imageio = self._get_imageio()

            out_dir = Path(self.depth_visualization_dir) / sanitize_filename(sample_id)
            out_dir.mkdir(parents=True, exist_ok=True)

            gif_path = out_dir / f"{sanitize_filename(view_name)}_depth.gif"
            imageio.mimsave(str(gif_path), list(depth_rgb), fps=6, loop=0)

            self._depth_visualization_count += 1

        except Exception:
            return

    # ------------------------------------------------------------------
    # Lazy imports
    # ------------------------------------------------------------------

    def _get_np(self) -> Any:
        if self._np is not None:
            return self._np
        import numpy as np  # type: ignore

        self._np = np
        return np

    def _get_cv2(self) -> Any:
        if self._cv2 is not None:
            return self._cv2
        import cv2  # type: ignore

        self._cv2 = cv2
        return cv2

    def _get_cm(self) -> Any:
        if self._cm is not None:
            return self._cm
        import matplotlib.cm as cm  # type: ignore

        self._cm = cm
        return cm

    def _get_imageio(self) -> Any:
        if self._imageio is not None:
            return self._imageio
        import imageio  # type: ignore

        self._imageio = imageio
        return imageio

    def _get_pil_image(self) -> Any:
        if self._pil_image is not None:
            return self._pil_image
        from PIL import Image  # type: ignore

        self._pil_image = Image
        return Image

# Legacy aliases kept for compatibility with older imports.
DepthConsistency = DepthConsistencyMetric
DEPTH_CONSISTENCY = DepthConsistencyMetric

def clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))

def sanitize_filename(name: str) -> str:
    allowed = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            allowed.append(ch)
        else:
            allowed.append("_")
    text = "".join(allowed)
    return text[:180] if len(text) > 180 else text
