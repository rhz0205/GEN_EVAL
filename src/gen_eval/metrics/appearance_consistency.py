"""DINO-based appearance consistency metric."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

class AppearanceConsistencyMetric:
    """Measure frame-to-frame appearance stability from sampled embeddings."""

    name = "appearance_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

        # Current recommended mode for your data.
        # "self": evaluate generated videos without reference_video.
        self.mode = self.config.get("mode", "self")

        # Runtime / model settings.
        self.device = self.config.get("device", "cuda")

        # DINO official repo path. It should point to the local dino repo root.
        # The repo root should support:
        # torch.hub.load(repo_or_dir, model_name, source="local", pretrained=False)
        self.repo_path = (
            self.config.get("repo_path")
            or self.config.get("repo_or_dir")
            or self.config.get("dino_repo_path")
            or self.config.get("local_repo_path")
        )

        # DINO weight path, e.g. dino_vitbase16_pretrain.pth.
        self.weight_path = (
            self.config.get("weight_path")
            or self.config.get("weights_path")
            or self.config.get("dino_weight_path")
            or self.config.get("local_save_path")
        )

        self.model_name = self.config.get("model_name", "dino_vitb16")
        self.use_fp16 = bool(self.config.get("use_fp16", False))
        self.strict_load = bool(self.config.get("strict_load", True))

        # Frame sampling.
        self.num_frames = int(self.config.get("num_frames", 8))
        self.frame_positions = self.config.get("frame_positions")
        self.image_size = int(self.config.get("image_size", self.config.get("resize", 224)))
        self.batch_size = int(self.config.get("batch_size", 16))
        self.min_frames = int(self.config.get("min_frames", 3))

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
        # Available keys: acm, tji_score, ts, balanced_score.
        self.score_key = self.config.get("score_key", "balanced_score")
        self.balanced_acm_weight = float(self.config.get("balanced_acm_weight", 0.5))
        self.balanced_tji_weight = float(self.config.get("balanced_tji_weight", 0.5))

        # Lazy runtime objects.
        self._torch = None
        self._cv2 = None
        self._torchvision_transforms = None
        self._torchvision_tf = None
        self._pil_image = None
        self._dino_model = None
        self._transform = None

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        """Evaluate appearance consistency across manifest samples."""
        evaluated_samples: list[dict[str, Any]] = []
        valid_scores: list[float] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []

        runtime_status = self._ensure_dino()
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
                        "reason": f"Unsupported appearance_consistency mode: {self.mode}",
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
            reason = "No sample produced a valid appearance consistency score."

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

        video_result = self._evaluate_single_video(str(video_path))

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

        for view, path in sorted(camera_videos.items()):
            view = str(view)
            if view in self.exclude_views:
                continue

            result = self._evaluate_single_video(str(path))
            view_results[view] = result

            if result.get("status") == "success":
                score = self._select_score(result)
                if math.isfinite(score):
                    view_scores.append(score)

        if not view_scores:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "self",
                "score": None,
                "status": "skipped",
                "reason": "No camera view produced a valid appearance consistency score.",
                "view_results": view_results,
            }

        score = float(sum(view_scores) / len(view_scores))

        return {
            "sample_id": sample_id,
            "metric": self.name,
            "mode": "self",
            "score": score,
            "status": "success",
            "num_views": len(view_scores),
            "view_results": view_results,
        }

    def _select_score(self, result: dict[str, Any]) -> float:
        value = result.get(self.score_key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)

        # Fallback order.
        for key in ("balanced_score", "ts", "acm", "tji_score"):
            value = result.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)

        return 0.0

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

        features = self._extract_dino_features(frames)
        metrics = self._compute_appearance_metrics(features)

        if metrics is None:
            return {
                "video_path": video_path,
                "status": "skipped",
                "reason": "failed to compute appearance metrics",
                "sampled_frame_indices": frame_indices,
                "readable_frame_indices": valid_indices,
            }

        result = {
            "video_path": video_path,
            "status": "success",
            "score": metrics["balanced_score"],
            "acm": metrics["acm"],
            "video_sim": metrics["video_sim"],
            "num_frames": len(frames),
            "num_transitions": metrics["num_transitions"],
            "tji": metrics["tji"],
            "tji_score": metrics["tji_score"],
            "ts": metrics["ts"],
            "balanced_score": metrics["balanced_score"],
            "sampled_frame_indices": frame_indices,
            "readable_frame_indices": valid_indices,
        }
        return result

    def _compute_appearance_metrics(self, features: Any) -> dict[str, Any] | None:
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

            balanced_score = weighted_average(
                [
                    (acm, self.balanced_acm_weight),
                    (tji_score, self.balanced_tji_weight),
                ]
            )

        return {
            "acm": clamp01(acm),
            "video_sim": video_sim,
            "num_transitions": num_transitions,
            "tji": float(tji),
            "tji_score": clamp01(tji_score),
            "ts": clamp01(float(ts)),
            "balanced_score": clamp01(float(balanced_score)),
        }

    # ------------------------------------------------------------------
    # DINO utilities
    # ------------------------------------------------------------------

    def _ensure_dino(self) -> str | None:
        """Initialize DINO lazily.

        Returns None if ready, otherwise a reason string.
        """
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

        # torch.hub.load(..., source='local') normally accepts repo path directly.
        # Adding it to sys.path also helps if the repo uses relative imports.
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

        return torch.cat(features_list, dim=0)

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
            raise RuntimeError(f"cv2 is required for appearance_consistency: {exc}") from exc

# Legacy alias kept for compatibility with older imports.
AppearanceConsistency = AppearanceConsistencyMetric

def clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))

def weighted_average(items: list[tuple[float | None, float]]) -> float:
    total = 0.0
    weight_sum = 0.0

    for value, weight in items:
        if value is None:
            continue
        if weight <= 0:
            continue
        total += clamp01(float(value)) * float(weight)
        weight_sum += float(weight)

    if weight_sum <= 0:
        return 0.0

    return clamp01(total / weight_sum)
