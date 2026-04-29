from __future__ import annotations

import importlib
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import torch

from modules.base import BaseModule
from modules.video_integrity import EXPECTED_CAMERA_VIEWS, get_cv2, inspect_video

ADJACENT_CAMERA_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("camera_front", "camera_cross_left", "left", "right"),
    ("camera_front", "camera_cross_right", "right", "left"),
    ("camera_cross_left", "camera_rear_left", "left", "right"),
    ("camera_cross_right", "camera_rear_right", "right", "left"),
    ("camera_rear_left", "camera_rear", "right", "left"),
    ("camera_rear_right", "camera_rear", "left", "right"),
)


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


class ViewConsistency(BaseModule):
    name = "view_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.camera_videos_key = self.config.get("camera_videos_key", "camera_videos")
        self.device = self.config.get("device", "cuda")
        self.loftr_repo_path = self.config.get("repo_path") or self.config.get(
            "loftr_repo_path"
        )
        self.loftr_weight_path = (
            self.config.get("weight_path")
            or self.config.get("loftr_weight_path")
            or self.config.get("local_save_path")
        )

        resize = self.config.get("resize")
        if isinstance(resize, (list, tuple)) and len(resize) >= 2:
            self.resize_width = int(resize[0])
            self.resize_height = int(resize[1])
        else:
            self.resize_width = int(self.config.get("resize_width", 640))
            self.resize_height = int(self.config.get("resize_height", 480))

        self.crop_ratio = 1.0 / 3.0
        self.match_count_ref = int(self.config.get("match_count_ref", 50))
        self.align_multiple = int(self.config.get("align_multiple", 8))
        self._matcher = None
        self._torch = None

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        evaluated_samples: list[dict[str, Any]] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        valid_scores: list[float] = []

        for sample in samples:
            sample_id = getattr(sample, "sample_id", None) or "unknown"
            try:
                sample_result = self._evaluate_sample(sample)
            except Exception as exc:
                failed_samples.append(
                    {"sample_id": sample_id, "reason": f"{type(exc).__name__}: {exc}"}
                )
                continue

            status = sample_result.get("status")
            score = sample_result.get("view_consistency_score")
            if status == "success" and is_finite_number(score):
                evaluated_samples.append(
                    {
                        "sample_id": sample_id,
                        "view_consistency_score": float(score),
                        "pair_scores": sample_result.get("pair_scores", {}),
                        "num_evaluated_pairs": sample_result.get(
                            "num_evaluated_pairs", 0
                        ),
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

        view_consistency_score = mean_or_none(valid_scores)
        if view_consistency_score is not None:
            status = "success"
            reason = None
        else:
            status = "failed" if failed_samples else "skipped"
            reason = "No sample produced a valid view_consistency_score."

        result: dict[str, Any] = {
            "metric": self.name,
            "status": status,
            "num_samples": len(samples),
            "valid_sample_count": len(valid_scores),
            "skipped_sample_count": len(skipped_samples),
            "failed_sample_count": len(failed_samples),
            "valid_evaluated_count": len(valid_scores),
            "mean_view_consistency_score": view_consistency_score,
            "view_consistency_score": view_consistency_score,
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
            return self._skipped_sample(
                sample_id,
                "metadata['camera_videos'] is required and must be a non-empty dict.",
            )

        if not self._loftr_is_configured():
            return self._skipped_sample(
                sample_id, "LoFTR repo_path/weight_path is not configured."
            )

        matcher_status = self._ensure_loftr()
        if matcher_status is not None:
            return {
                "sample_id": sample_id,
                "status": "failed",
                "reason": matcher_status,
            }

        normalized_videos = {
            str(view): str(path)
            for view, path in camera_videos.items()
            if path is not None
        }
        missing_views = [
            view for view in EXPECTED_CAMERA_VIEWS if view not in normalized_videos
        ]
        if missing_views:
            return self._skipped_sample(
                sample_id, f"Missing expected camera views: {', '.join(missing_views)}."
            )

        video_infos = {
            view: inspect_video(normalized_videos[view])
            for view in EXPECTED_CAMERA_VIEWS
        }
        unreadable_views = [
            view for view, info in video_infos.items() if not bool(info.get("readable"))
        ]
        if unreadable_views:
            return self._skipped_sample(
                sample_id,
                f"Unreadable expected camera views: {', '.join(unreadable_views)}.",
            )

        pair_scores: list[float] = []
        pair_details: dict[str, dict[str, Any]] = {}
        for cam_a, cam_b, side_a, side_b in ADJACENT_CAMERA_PAIRS:
            pair_result = self._evaluate_pair(
                normalized_videos[cam_a],
                normalized_videos[cam_b],
                video_infos[cam_a],
                video_infos[cam_b],
                side_a,
                side_b,
            )
            pair_name = f"{cam_a}|{cam_b}"
            if pair_result is None:
                return self._skipped_sample(
                    sample_id, f"Unable to evaluate adjacent pair {cam_a}|{cam_b}."
                )
            pair_score = pair_result.get("score")
            if not is_finite_number(pair_score):
                return self._skipped_sample(
                    sample_id, f"Unable to score adjacent pair {cam_a}|{cam_b}."
                )
            pair_scores.append(float(pair_score))
            pair_details[pair_name] = pair_result

        return {
            "sample_id": sample_id,
            "status": "success",
            "view_consistency_score": mean_or_none(pair_scores),
            "pair_scores": pair_details,
            "num_evaluated_pairs": len(pair_details),
        }

    def _evaluate_pair(
        self,
        path_a: str,
        path_b: str,
        info_a: dict[str, Any],
        info_b: dict[str, Any],
        side_a: str,
        side_b: str,
    ) -> dict[str, Any] | None:
        frame_count_a = int(info_a.get("frame_count") or 0)
        frame_count_b = int(info_b.get("frame_count") or 0)
        frame_count = min(frame_count_a, frame_count_b)
        if frame_count <= 0:
            return None

        cv2 = get_cv2()
        cap_a = cv2.VideoCapture(str(path_a))
        cap_b = cv2.VideoCapture(str(path_b))
        if not cap_a.isOpened() or not cap_b.isOpened():
            cap_a.release()
            cap_b.release()
            return None

        frame_scores: list[float] = []
        confidence_values: list[float] = []
        match_counts: list[int] = []
        failed_frames = 0
        try:
            for _ in range(frame_count):
                ok_a, frame_bgr_a = cap_a.read()
                ok_b, frame_bgr_b = cap_b.read()
                if not ok_a or frame_bgr_a is None or not ok_b or frame_bgr_b is None:
                    failed_frames += 1
                    continue

                frame_a = cv2.cvtColor(frame_bgr_a, cv2.COLOR_BGR2RGB)
                frame_b = cv2.cvtColor(frame_bgr_b, cv2.COLOR_BGR2RGB)
                gray_a = self._preprocess_frame(frame_a)
                gray_b = self._preprocess_frame(frame_b)
                crop_a = self._crop_edge(gray_a, side_a)
                crop_b = self._crop_edge(gray_b, side_b)
                crop_a, crop_b = self._align_pair_for_loftr(crop_a, crop_b)

                match_result = self._match_loftr(crop_a, crop_b)
                mconf = match_result.get("mconf")
                if mconf is None or len(mconf) == 0:
                    frame_scores.append(0.0)
                    match_counts.append(0)
                    confidence_values.append(0.0)
                    continue

                confidences = [float(value) for value in mconf]
                mean_confidence = sum(confidences) / len(confidences)
                coverage_factor = min(
                    1.0, len(confidences) / max(1, self.match_count_ref)
                )
                frame_scores.append(mean_confidence * coverage_factor)
                match_counts.append(len(confidences))
                confidence_values.append(mean_confidence)
        finally:
            cap_a.release()
            cap_b.release()

        score = mean_or_none(frame_scores)
        if score is None:
            return None
        return {
            "score": score,
            "num_scored_frames": len(frame_scores),
            "num_failed_frames": failed_frames,
            "mean_match_confidence": mean_or_none(confidence_values),
            "mean_num_matches": mean_or_none([float(value) for value in match_counts]),
        }

    def _loftr_is_configured(self) -> bool:
        return bool(self.loftr_repo_path and self.loftr_weight_path)

    def _ensure_loftr(self) -> str | None:
        if self._matcher is not None:
            return None

        try:
            self._torch = torch
        except Exception as exc:
            return (
                f"torch is required for LoFTR evaluation: {type(exc).__name__}: {exc}"
            )

        if self.device == "cuda" and not self._torch.cuda.is_available():
            self.device = "cpu"

        if self.loftr_repo_path:
            repo_root = Path(str(self.loftr_repo_path)).expanduser().resolve()
            repo_candidates = [repo_root, repo_root / "src"]
            for candidate in repo_candidates:
                candidate_text = str(candidate)
                if candidate.exists() and candidate_text not in sys.path:
                    sys.path.insert(0, candidate_text)

        try:
            LoFTR, default_cfg = self._import_loftr_symbols()
        except Exception as exc:
            return (
                "Failed to import official LoFTR. Provide config['loftr_repo_path'] "
                f"pointing to the LoFTR repo. Error: {type(exc).__name__}: {exc}"
            )

        try:
            model = LoFTR(config=default_cfg)
            weight_path = Path(str(self.loftr_weight_path)).expanduser().resolve()
            if not weight_path.exists():
                return f"LoFTR checkpoint not found: {weight_path}"

            ckpt = load_torch_state(weight_path, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict)
            model.eval().to(self.device)
            self._matcher = model
            return None
        except Exception as exc:
            self._matcher = None
            return f"Failed to initialize LoFTR: {type(exc).__name__}: {exc}"

    def _preprocess_frame(self, frame_rgb: Any) -> Any:
        cv2 = get_cv2()
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.resize_width, self.resize_height))

    def _crop_edge(self, img: Any, side: str) -> Any:
        width = img.shape[1]
        crop_width = max(1, int(width * self.crop_ratio))
        if side == "left":
            return img[:, :crop_width]
        return img[:, width - crop_width :]

    def _align_pair_for_loftr(self, img_a: Any, img_b: Any) -> tuple[Any, Any]:
        cv2 = get_cv2()
        common_height = min(int(img_a.shape[0]), int(img_b.shape[0]))
        common_width = min(int(img_a.shape[1]), int(img_b.shape[1]))
        align_multiple = max(1, int(self.align_multiple))

        aligned_height = (common_height // align_multiple) * align_multiple
        aligned_width = (common_width // align_multiple) * align_multiple
        if aligned_height <= 0:
            aligned_height = common_height
        if aligned_width <= 0:
            aligned_width = common_width

        if (
            int(img_a.shape[0]) != aligned_height
            or int(img_a.shape[1]) != aligned_width
        ):
            img_a = cv2.resize(img_a, (aligned_width, aligned_height))
        if (
            int(img_b.shape[0]) != aligned_height
            or int(img_b.shape[1]) != aligned_width
        ):
            img_b = cv2.resize(img_b, (aligned_width, aligned_height))
        return img_a, img_b

    def _to_tensor(self, gray: Any) -> Any:
        if self._torch is None:
            raise RuntimeError("torch is not initialized")
        tensor = self._torch.from_numpy(gray.astype("float32")) / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def _match_loftr(self, gray_a: Any, gray_b: Any) -> dict[str, Any]:
        if self._matcher is None:
            raise RuntimeError("LoFTR matcher is not initialized")
        if self._torch is None:
            raise RuntimeError("torch is not initialized")

        data = {"image0": self._to_tensor(gray_a), "image1": self._to_tensor(gray_b)}
        with self._torch.no_grad():
            self._matcher(data)

        mkpts0 = data.get("mkpts0_f")
        mkpts1 = data.get("mkpts1_f")
        mconf = data.get("mconf")
        if mkpts0 is None or mkpts1 is None or mconf is None:
            return {"mconf": []}
        if getattr(mconf, "ndim", None) == 0:
            return {"mconf": []}
        return {"mconf": mconf.detach().cpu().numpy()}

    def _skipped_sample(self, sample_id: str, reason: str) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "status": "skipped",
            "reason": reason,
        }

    def _import_loftr_symbols(self) -> tuple[Any, Any]:
        attempted: list[str] = []
        for module_name in ("src.loftr", "loftr"):
            attempted.append(module_name)
            try:
                module = importlib.import_module(module_name)
                return getattr(module, "LoFTR"), getattr(module, "default_cfg")
            except Exception:
                continue
        raise ImportError(
            f"Unable to import LoFTR symbols from: {', '.join(attempted)}"
        )


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
