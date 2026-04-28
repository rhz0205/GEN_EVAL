"""Temporal consistency metric for GEN_EVAL.

This implementation provides a reference-free / self-consistency mode for
generated videos.

The metric uses CLIP image embeddings extracted from sampled video frames and
computes:

- ACM: Adjacent-frame Cosine similarity Mean.
  Higher means adjacent frames are more visually/semantically consistent.

- TJI: Temporal Jerkiness Index in CLIP feature space.
  Higher means the feature trajectory has stronger abrupt second-order change.

- TJI score: exp(-0.5 * TJI).
  Higher means smoother temporal evolution.

- TS: Temporal self-consistency score.
  TS = ACM / (1 + TJI)

This file is manifest-driven and does not depend on WorldLens directory
conventions, WorldLens utils, or online model downloads.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

class TemporalConsistencyMetric:
    """Reference-free temporal consistency metric."""

    name = "temporal_consistency"

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or {}

        # Current recommended mode for your data.
        # "self": evaluate generated videos without reference_video.
        self.mode = self.config.get("mode", "self")

        # Runtime / model settings.
        self.device = self.config.get("device", "cuda")
        self.clip_weight_path = (
            self.config.get("weight_path")
            or self.config.get("clip_weight_path")
            or self.config.get("local_save_path")
        )

        # Only used if explicitly allowed. This may trigger CLIP cache/download
        # behavior depending on the installed clip package, so it is disabled by
        # default for offline servers.
        self.clip_model_name = self.config.get("clip_model_name", "ViT-B/32")
        self.allow_clip_model_name = bool(self.config.get("allow_clip_model_name", False))

        # Frame sampling.
        self.num_frames = int(self.config.get("num_frames", 8))
        self.frame_positions = self.config.get("frame_positions")
        self.resize = int(self.config.get("resize", 224))
        self.batch_size = int(self.config.get("batch_size", 16))

        # View handling.
        # False: only sample.generated_video, usually camera_front.mp4.
        # True: evaluate each video in metadata["camera_videos"] and average.
        self.use_all_views = bool(self.config.get("use_all_views", True))
        self.camera_videos_key = self.config.get("camera_videos_key", "camera_videos")
        self.keep_front_tele = bool(self.config.get("keep_front_tele", False))
        self.exclude_views = set(self.config.get("exclude_views", []))
        if not self.keep_front_tele:
            self.exclude_views.add("camera_front_tele")

        # Score options.
        self.score_key = self.config.get("score_key", "ts")
        self.min_frames = int(self.config.get("min_frames", 3))

        # Lazy runtime objects.
        self._torch = None
        self._clip = None
        self._cv2 = None
        self._pil_image = None
        self._clip_model = None
        self._preprocess = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        """Evaluate temporal consistency over manifest samples."""
        details: list[dict[str, Any]] = []
        valid_scores: list[float] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []

        runtime_status = self._ensure_clip()
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
                        "reason": f"Unsupported temporal_consistency mode: {self.mode}",
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

            details.append(sample_result)

            status = sample_result.get("status")
            score = sample_result.get("score")

            if status == "ok" and isinstance(score, (int, float)) and math.isfinite(float(score)):
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
            status = "ok"
            reason = None
        else:
            final_score = None
            status = "skipped" if not failed_samples else "failed"
            reason = "No sample produced a valid temporal consistency score."

        result = {
            "metric": self.name,
            "score": final_score,
            "num_samples": len(valid_scores),
            "details": {
                "evaluated_samples": details,
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

        video_result = self._evaluate_single_video(str(video_path))

        if video_result.get("status") != "ok":
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "self",
                "score": None,
                "status": video_result.get("status", "skipped"),
                "reason": video_result.get("reason", "single video evaluation failed"),
                "video_result": video_result,
            }

        score = float(video_result.get(self.score_key, video_result.get("ts", 0.0)))

        return {
            "sample_id": sample_id,
            "metric": self.name,
            "mode": "self",
            "score": score,
            "status": "ok",
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

        for view, path in sorted(camera_videos.items()):
            view = str(view)
            if view in self.exclude_views:
                continue

            result = self._evaluate_single_video(str(path))
            view_results[view] = result

            if (
                result.get("status") == "ok"
                and isinstance(result.get(self.score_key, result.get("ts")), (int, float))
            ):
                score = float(result.get(self.score_key, result.get("ts")))
                if math.isfinite(score):
                    view_scores.append(score)

        if not view_scores:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "self",
                "score": None,
                "status": "skipped",
                "reason": "No camera view produced a valid temporal consistency score.",
                "view_results": view_results,
            }

        score = float(sum(view_scores) / len(view_scores))

        return {
            "sample_id": sample_id,
            "metric": self.name,
            "mode": "self",
            "score": score,
            "status": "ok",
            "num_views": len(view_scores),
            "view_results": view_results,
        }

    # ------------------------------------------------------------------
    # Video-level evaluation
    # ------------------------------------------------------------------

    def _evaluate_single_video(self, video_path: str) -> dict[str, Any]:
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

        frame_indices = self._sample_frame_indices(video_path)
        if len(frame_indices) < self.min_frames:
            return {
                "video_path": video_path,
                "status": "skipped",
                "reason": f"not enough sampled frames: {len(frame_indices)} < {self.min_frames}",
                "sampled_frame_indices": frame_indices,
            }

        frames = []
        valid_indices = []
        for idx in frame_indices:
            frame = self._read_frame(video_path, idx)
            if frame is None:
                continue
            frames.append(frame)
            valid_indices.append(idx)

        if len(frames) < self.min_frames:
            return {
                "video_path": video_path,
                "status": "skipped",
                "reason": f"not enough readable frames: {len(frames)} < {self.min_frames}",
                "sampled_frame_indices": frame_indices,
                "readable_frame_indices": valid_indices,
            }

        features = self._extract_clip_features(frames)
        metrics = self._compute_temporal_metrics(features)

        if metrics is None:
            return {
                "video_path": video_path,
                "status": "skipped",
                "reason": "failed to compute temporal metrics",
                "sampled_frame_indices": frame_indices,
                "readable_frame_indices": valid_indices,
            }

        result = {
            "video_path": video_path,
            "status": "ok",
            "score": metrics["ts"],
            "acm": metrics["acm"],
            "video_sim": metrics["video_sim"],
            "num_frames": len(frames),
            "num_transitions": metrics["num_transitions"],
            "tji": metrics["tji"],
            "tji_score": metrics["tji_score"],
            "ts": metrics["ts"],
            "sampled_frame_indices": frame_indices,
            "readable_frame_indices": valid_indices,
        }
        return result

    def _compute_temporal_metrics(self, features: Any) -> Optional[dict[str, Any]]:
        torch = self._torch
        if torch is None:
            raise RuntimeError("torch is not initialized")

        if features is None or len(features) < 2:
            return None

        with torch.no_grad():
            sims = torch.nn.functional.cosine_similarity(
                features[:-1],
                features[1:],
                dim=-1,
            )
            sims = torch.clamp(sims, min=0.0)

            acm = float(sims.mean().item())
            video_sim = float(sims.sum().item())
            num_transitions = int(len(features) - 1)

            if len(features) >= 3:
                velocity = (features[1:] - features[:-1]).norm(dim=1)
                acceleration = (
                    features[2:] - 2 * features[1:-1] + features[:-2]
                ).norm(dim=1)

                denom = 0.5 * (velocity[1:] + velocity[:-1]) + 1e-8
                tji_tensor = (acceleration / denom).mean()
                tji = float(tji_tensor.item())
                tji_score = float(torch.exp(-0.5 * tji_tensor).item())
            else:
                tji = 0.0
                tji_score = 1.0

            ts = acm / (1.0 + tji)
            if not math.isfinite(ts):
                ts = 0.0

        return {
            "acm": clamp01(acm),
            "video_sim": video_sim,
            "num_transitions": num_transitions,
            "tji": float(tji),
            "tji_score": clamp01(tji_score),
            "ts": clamp01(float(ts)),
        }

    # ------------------------------------------------------------------
    # CLIP utilities
    # ------------------------------------------------------------------

    def _ensure_clip(self) -> Optional[str]:
        """Initialize CLIP lazily.

        Returns None if ready, otherwise a reason string.
        """
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

        model_source = None

        if self.clip_weight_path:
            weight_path = Path(str(self.clip_weight_path)).expanduser().resolve()
            if not weight_path.exists():
                return f"CLIP weight path does not exist: {weight_path}"
            model_source = str(weight_path)
        elif self.allow_clip_model_name:
            # This may rely on local CLIP cache or trigger download depending on
            # clip package behavior. Disabled by default for offline servers.
            model_source = self.clip_model_name
        else:
            return (
                "CLIP weight path is required. Set config['clip_weight_path'] to a local "
                "CLIP .pt file, or set allow_clip_model_name=true only if the model is "
                "already cached locally."
            )

        try:
            model, preprocess = self._clip.load(model_source, device=self.device, jit=False)
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

            return torch.cat(all_features, dim=0)

    # ------------------------------------------------------------------
    # Video utilities
    # ------------------------------------------------------------------

    def _sample_frame_indices(self, video_path: str) -> list[int]:
        info = self._inspect_video(video_path)
        frame_count = int(info.get("frame_count") or 0)

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

    def _inspect_video(self, video_path: str) -> dict[str, Any]:
        cv2 = self._get_cv2()
        path = Path(video_path)

        info = {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "readable": False,
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration": None,
        }

        if not path.exists() or not path.is_file():
            return info

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return info

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        duration = None
        if frame_count > 0 and fps > 0:
            duration = float(frame_count / fps)

        info.update(
            {
                "readable": frame_count > 0 and width > 0 and height > 0,
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": duration,
            }
        )
        return info

    def _read_frame(self, video_path: str, frame_idx: int) -> Any | None:
        try:
            cv2 = self._get_cv2()
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                return None

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            cap.release()

            if not ok or frame_bgr is None:
                return None

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return frame_rgb

        except Exception:
            return None

    def _get_cv2(self) -> Any:
        if self._cv2 is not None:
            return self._cv2

        try:
            import cv2  # type: ignore

            self._cv2 = cv2
            return cv2

        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"cv2 is required for temporal_consistency: {exc}") from exc

# Backward-compatible aliases.
TemporalConsistency = TemporalConsistencyMetric
TEMPORAL_CONSISTENCY = TemporalConsistencyMetric

def clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
