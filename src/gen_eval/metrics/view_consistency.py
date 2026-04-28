"""Cross-view consistency metric for multi-camera video manifests."""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Any

DEFAULT_CAMERA_PAIRS: list[tuple[str, str, str, str]] = [
    ("camera_front", "camera_cross_left", "left", "right"),
    ("camera_front", "camera_cross_right", "right", "left"),
    ("camera_cross_left", "camera_rear_left", "left", "right"),
    ("camera_cross_right", "camera_rear_right", "right", "left"),
    ("camera_rear_left", "camera_rear", "right", "left"),
    ("camera_rear_right", "camera_rear", "left", "right"),
]

class ViewConsistencyMetric:
    """Manifest-based cross-view consistency metric."""

    name = "view_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

        # Main mode:
        # - video_integrity: inspect raw multi-view video metadata
        # - loftr: run LoFTR on adjacent camera pairs
        # - precomputed: aggregate cross-view scores/matches/features from metadata
        # - auto: precomputed -> loftr if configured -> video_integrity
        self.mode = self.config.get("mode", "loftr")

        # Metadata keys.
        self.camera_videos_key = self.config.get("camera_videos_key", "camera_videos")
        self.views_key = self.config.get("views_key", "views")
        self.camera_pairs_key = self.config.get("camera_pairs_key", "camera_pairs")
        self.cross_view_scores_key = self.config.get("cross_view_scores_key", "cross_view_scores")
        self.cross_view_confidence_key = self.config.get(
            "cross_view_confidence_key", "cross_view_confidence"
        )
        self.cross_view_matches_key = self.config.get("cross_view_matches_key", "cross_view_matches")
        self.cross_view_features_key = self.config.get("cross_view_features_key", "cross_view_features")

        # View settings.
        self.expected_num_views = self.config.get("expected_num_views")
        self.expected_views = self.config.get("expected_views")

        # camera_front_tele is often redundant with camera_front in your current
        # workflow. It is excluded by default unless explicitly enabled.
        self.keep_front_tele = bool(self.config.get("keep_front_tele", False))
        self.exclude_views = set(self.config.get("exclude_views", []))
        if not self.keep_front_tele:
            self.exclude_views.add("camera_front_tele")

        # Score weights for video_integrity mode.
        self.presence_weight = float(self.config.get("presence_weight", 0.25))
        self.readability_weight = float(self.config.get("readability_weight", 0.25))
        self.frame_count_weight = float(self.config.get("frame_count_weight", 0.20))
        self.resolution_weight = float(self.config.get("resolution_weight", 0.15))
        self.fps_weight = float(self.config.get("fps_weight", 0.10))
        self.duration_weight = float(self.config.get("duration_weight", 0.05))

        # Tolerances for video_integrity mode.
        self.frame_count_tolerance = float(self.config.get("frame_count_tolerance", 0.05))
        self.fps_tolerance = float(self.config.get("fps_tolerance", 0.05))
        self.duration_tolerance = float(self.config.get("duration_tolerance", 0.05))

        # Score weights for precomputed mode.
        self.score_weight = float(self.config.get("score_weight", 0.40))
        self.match_weight = float(self.config.get("match_weight", 0.30))
        self.feature_weight = float(self.config.get("feature_weight", 0.30))

        # LoFTR settings.
        self.device = self.config.get("device", "cuda")
        self.loftr_repo_path = self.config.get("repo_path") or self.config.get("loftr_repo_path")
        self.loftr_weight_path = self.config.get("weight_path") or self.config.get("loftr_weight_path") or self.config.get(
            "local_save_path"
        )
        self.loftr_config = self.config.get("loftr_config", "outdoor")
        self.num_frames = int(self.config.get("num_frames", 3))
        self.frame_positions = self.config.get("frame_positions")  # optional list in [0, 1]
        resize = self.config.get("resize")
        if isinstance(resize, (list, tuple)) and len(resize) >= 2:
            self.resize_width = int(resize[0])
            self.resize_height = int(resize[1])
        else:
            self.resize_width = int(self.config.get("resize_width", 640))
            self.resize_height = int(self.config.get("resize_height", 480))
        self.crop_ratio = float(self.config.get("crop_ratio", 1.0))
        self.conf_threshold = float(self.config.get("conf_threshold", 0.0))
        self.target_matches = float(self.config.get("target_matches", 200))
        self.loftr_conf_weight = float(self.config.get("loftr_conf_weight", 0.70))
        self.loftr_count_weight = float(self.config.get("loftr_count_weight", 0.30))
        self.max_pairs_per_sample = self.config.get("max_pairs_per_sample")
        self.fail_if_loftr_unavailable = bool(self.config.get("fail_if_loftr_unavailable", False))

        # Optional configured camera pairs. Accepted forms:
        # [["cam_a", "cam_b"], ...]
        # [["cam_a", "cam_b", "side_a", "side_b"], ...]
        self.config_camera_pairs = self.config.get("camera_pairs")

        # Visualization settings for LoFTR mode.
        self.save_visualizations = bool(self.config.get("save_visualizations", False))
        self.visualization_dir = self.config.get(
            "visualization_dir", "outputs/cross_view_visualizations"
        )
        self.max_visualizations = int(self.config.get("max_visualizations", 100))
        self.visualize_min_conf = float(
            self.config.get("visualize_min_conf", self.conf_threshold)
        )
        self.max_visual_matches = int(self.config.get("max_visual_matches", 300))
        self._visualization_count = 0

        self._matcher = None
        self._torch = None
        self._cv2 = None
        self._np = None

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        """Evaluate cross-view consistency over samples."""
        evaluated_samples: list[dict[str, Any]] = []
        valid_scores: list[float] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []

        for sample in samples:
            sample_id = getattr(sample, "sample_id", None) or "unknown"

            try:
                sample_result = self._evaluate_sample(sample)
            except Exception as exc:  # noqa: BLE001
                failed_result = {
                    "sample_id": sample_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                evaluated_samples.append(failed_result)
                failed_samples.append(failed_result)
                continue

            evaluated_samples.append(sample_result)

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
            reason = "No sample produced a valid cross-view consistency score."

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

    def _evaluate_sample(self, sample: Any) -> dict[str, Any]:
        sample_id = getattr(sample, "sample_id", None) or "unknown"
        metadata = getattr(sample, "metadata", None) or {}

        if self.mode == "video_integrity":
            return self._evaluate_video_integrity(sample_id, metadata)

        if self.mode == "loftr":
            return self._evaluate_loftr(sample_id, metadata)

        if self.mode == "precomputed":
            return self._evaluate_precomputed(sample_id, metadata)

        if self.mode == "auto":
            if self._has_precomputed_evidence(metadata):
                return self._evaluate_precomputed(sample_id, metadata)
            if self._loftr_is_configured():
                return self._evaluate_loftr(sample_id, metadata)
            return self._evaluate_video_integrity(sample_id, metadata)

        return {
            "sample_id": sample_id,
            "metric": self.name,
            "score": None,
            "status": "skipped",
            "reason": f"Unsupported view_consistency mode: {self.mode}",
        }

    # ------------------------------------------------------------------
    # Mode 1: raw multi-view video integrity
    # ------------------------------------------------------------------

    def _evaluate_video_integrity(self, sample_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        camera_videos = metadata.get(self.camera_videos_key)

        if not isinstance(camera_videos, dict) or not camera_videos:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "score": None,
                "status": "skipped",
                "reason": "metadata['camera_videos'] is required for video_integrity mode.",
            }

        camera_videos = self._filter_camera_videos(camera_videos)

        if not camera_videos:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "score": None,
                "status": "skipped",
                "reason": "No usable camera videos after view filtering.",
            }

        expected_views = self._resolve_expected_views(metadata, camera_videos)
        expected_num_views = len(expected_views)

        video_infos: dict[str, dict[str, Any]] = {}
        for view in expected_views:
            path = camera_videos.get(view)
            if path is None:
                video_infos[view] = {
                    "path": None,
                    "exists": False,
                    "readable": False,
                    "reason": "missing from camera_videos",
                }
                continue

            video_infos[view] = self._inspect_video(path)

        existing_views = [view for view, info in video_infos.items() if bool(info.get("exists"))]
        readable_views = [view for view, info in video_infos.items() if bool(info.get("readable"))]

        presence_score = safe_div(len(existing_views), expected_num_views)
        readability_score = safe_div(len(readable_views), expected_num_views)

        readable_infos = [video_infos[v] for v in readable_views]

        frame_count_score = self._numeric_consistency_score(
            [info.get("frame_count") for info in readable_infos],
            tolerance=self.frame_count_tolerance,
        )
        fps_score = self._numeric_consistency_score(
            [info.get("fps") for info in readable_infos],
            tolerance=self.fps_tolerance,
        )
        duration_score = self._numeric_consistency_score(
            [info.get("duration") for info in readable_infos],
            tolerance=self.duration_tolerance,
        )
        resolution_score = self._resolution_consistency_score(
            [(info.get("width"), info.get("height")) for info in readable_infos]
        )

        score = weighted_average(
            [
                (presence_score, self.presence_weight),
                (readability_score, self.readability_weight),
                (frame_count_score, self.frame_count_weight),
                (resolution_score, self.resolution_weight),
                (fps_score, self.fps_weight),
                (duration_score, self.duration_weight),
            ]
        )

        status = "ok" if readable_views else "skipped"
        reason = None if readable_views else "No readable camera videos."

        result = {
            "sample_id": sample_id,
            "metric": self.name,
            "mode": "video_integrity",
            "score": score if readable_views else None,
            "status": status,
            "view_count": len(camera_videos),
            "expected_num_views": expected_num_views,
            "existing_view_count": len(existing_views),
            "readable_view_count": len(readable_views),
            "component_scores": {
                "presence": presence_score,
                "readability": readability_score,
                "frame_count_consistency": frame_count_score,
                "resolution_consistency": resolution_score,
                "fps_consistency": fps_score,
                "duration_consistency": duration_score,
            },
            "video_infos": video_infos,
        }

        if reason:
            result["reason"] = reason

        return result

    # ------------------------------------------------------------------
    # Mode 2: LoFTR-based CVC
    # ------------------------------------------------------------------

    def _evaluate_loftr(self, sample_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        camera_videos = metadata.get(self.camera_videos_key)

        if not isinstance(camera_videos, dict) or not camera_videos:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "loftr",
                "score": None,
                "status": "skipped",
                "reason": "metadata['camera_videos'] is required for loftr mode.",
            }

        camera_videos = self._filter_camera_videos(camera_videos)

        if not camera_videos:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "loftr",
                "score": None,
                "status": "skipped",
                "reason": "No usable camera videos after view filtering.",
            }

        matcher_status = self._ensure_loftr()
        if matcher_status is not None:
            status = "failed" if self.fail_if_loftr_unavailable else "skipped"
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "loftr",
                "score": None,
                "status": status,
                "reason": matcher_status,
            }

        camera_pairs = self._resolve_camera_pairs(metadata, camera_videos)
        if self.max_pairs_per_sample is not None:
            camera_pairs = camera_pairs[: int(self.max_pairs_per_sample)]

        if not camera_pairs:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "loftr",
                "score": None,
                "status": "skipped",
                "reason": "No valid camera pairs available for LoFTR evaluation.",
            }

        pair_results = []
        pair_scores = []

        for cam_a, cam_b, side_a, side_b in camera_pairs:
            path_a = camera_videos.get(cam_a)
            path_b = camera_videos.get(cam_b)

            if not path_a or not path_b:
                pair_results.append(
                    {
                        "pair": f"{cam_a}|{cam_b}",
                        "status": "skipped",
                        "reason": "one or both camera videos missing",
                    }
                )
                continue

            pair_result = self._evaluate_loftr_pair(
                sample_id=sample_id,
                cam_a=cam_a,
                cam_b=cam_b,
                path_a=str(path_a),
                path_b=str(path_b),
                side_a=side_a,
                side_b=side_b,
            )
            pair_results.append(pair_result)

            if (
                pair_result.get("status") == "ok"
                and isinstance(pair_result.get("score"), (int, float))
                and math.isfinite(float(pair_result["score"]))
            ):
                pair_scores.append(float(pair_result["score"]))

        if not pair_scores:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "loftr",
                "score": None,
                "status": "skipped",
                "reason": "No camera pair produced a valid LoFTR score.",
                "pair_results": pair_results,
            }

        sample_score = float(sum(pair_scores) / len(pair_scores))

        total_conf = sum(float(r.get("confidence_sum", 0.0)) for r in pair_results)
        total_matches = sum(int(r.get("valid_matches", 0)) for r in pair_results)
        mean_confidence = total_conf / total_matches if total_matches > 0 else 0.0

        return {
            "sample_id": sample_id,
            "metric": self.name,
            "mode": "loftr",
            "score": sample_score,
            "status": "ok",
            "num_pairs": len(pair_scores),
            "total_pairs_considered": len(camera_pairs),
            "total_valid_matches": total_matches,
            "mean_confidence": mean_confidence,
            "pair_results": pair_results,
        }

    def _evaluate_loftr_pair(
        self,
        sample_id: str,
        cam_a: str,
        cam_b: str,
        path_a: str,
        path_b: str,
        side_a: str,
        side_b: str,
    ) -> dict[str, Any]:
        info_a = self._inspect_video(path_a)
        info_b = self._inspect_video(path_b)

        if not info_a.get("readable") or not info_b.get("readable"):
            return {
                "pair": f"{cam_a}|{cam_b}",
                "status": "skipped",
                "reason": "one or both videos are unreadable",
                "video_info_a": info_a,
                "video_info_b": info_b,
            }

        frame_count_a = int(info_a.get("frame_count") or 0)
        frame_count_b = int(info_b.get("frame_count") or 0)
        frame_count = min(frame_count_a, frame_count_b)

        if frame_count <= 0:
            return {
                "pair": f"{cam_a}|{cam_b}",
                "status": "skipped",
                "reason": "invalid frame count",
                "video_info_a": info_a,
                "video_info_b": info_b,
            }

        frame_indices = self._sample_frame_indices(frame_count)

        frame_results = []
        valid_frame_scores = []
        confidence_sum = 0.0
        valid_matches_total = 0

        for frame_idx in frame_indices:
            frame_result = self._evaluate_loftr_frame_pair(
                sample_id=sample_id,
                cam_a=cam_a,
                cam_b=cam_b,
                path_a=path_a,
                path_b=path_b,
                frame_idx=frame_idx,
                side_a=side_a,
                side_b=side_b,
            )
            frame_results.append(frame_result)

            if (
                frame_result.get("status") == "ok"
                and isinstance(frame_result.get("score"), (int, float))
                and math.isfinite(float(frame_result["score"]))
            ):
                valid_frame_scores.append(float(frame_result["score"]))
                confidence_sum += float(frame_result.get("confidence_sum", 0.0))
                valid_matches_total += int(frame_result.get("valid_matches", 0))

        if not valid_frame_scores:
            return {
                "pair": f"{cam_a}|{cam_b}",
                "status": "skipped",
                "reason": "No sampled frame produced valid LoFTR matches.",
                "num_sampled_frames": len(frame_indices),
                "frame_results": frame_results,
                "video_info_a": info_a,
                "video_info_b": info_b,
            }

        pair_score = float(sum(valid_frame_scores) / len(valid_frame_scores))
        mean_confidence = (
            confidence_sum / valid_matches_total if valid_matches_total > 0 else 0.0
        )
        mean_valid_matches = valid_matches_total / len(valid_frame_scores)

        return {
            "pair": f"{cam_a}|{cam_b}",
            "camera_a": cam_a,
            "camera_b": cam_b,
            "side_a": side_a,
            "side_b": side_b,
            "status": "ok",
            "score": pair_score,
            "num_valid_frames": len(valid_frame_scores),
            "num_sampled_frames": len(frame_indices),
            "valid_matches": valid_matches_total,
            "mean_valid_matches": mean_valid_matches,
            "confidence_sum": confidence_sum,
            "mean_confidence": mean_confidence,
            "frame_results": frame_results,
        }

    def _evaluate_loftr_frame_pair(
        self,
        sample_id: str,
        cam_a: str,
        cam_b: str,
        path_a: str,
        path_b: str,
        frame_idx: int,
        side_a: str,
        side_b: str,
    ) -> dict[str, Any]:
        frame_a = self._read_frame(path_a, frame_idx)
        frame_b = self._read_frame(path_b, frame_idx)

        if frame_a is None or frame_b is None:
            return {
                "frame_idx": frame_idx,
                "status": "skipped",
                "reason": "failed to read one or both frames",
                "score": None,
            }

        gray_a = self._preprocess_frame(frame_a)
        gray_b = self._preprocess_frame(frame_b)

        crop_a = self._crop_edge(gray_a, side_a)
        crop_b = self._crop_edge(gray_b, side_b)

        match_result = self._match_loftr(crop_a, crop_b)
        mkpts0 = match_result["mkpts0"]
        mkpts1 = match_result["mkpts1"]
        mconf = match_result["mconf"]

        if mconf is None or len(mconf) == 0:
            if self.save_visualizations:
                self._maybe_save_loftr_visualization(
                    sample_id=sample_id,
                    cam_a=cam_a,
                    cam_b=cam_b,
                    frame_idx=frame_idx,
                    crop_a=crop_a,
                    crop_b=crop_b,
                    mkpts0=[],
                    mkpts1=[],
                    mconf=[],
                )

            return {
                "frame_idx": frame_idx,
                "status": "ok",
                "score": 0.0,
                "raw_matches": 0,
                "valid_matches": 0,
                "mean_confidence": 0.0,
                "confidence_sum": 0.0,
                "match_count_score": 0.0,
                "reason": "no LoFTR matches",
            }

        valid_indices = [i for i, x in enumerate(mconf) if float(x) >= self.conf_threshold]
        valid_conf = [float(mconf[i]) for i in valid_indices]

        raw_matches = len(mconf)
        valid_matches = len(valid_conf)

        if self.save_visualizations:
            self._maybe_save_loftr_visualization(
                sample_id=sample_id,
                cam_a=cam_a,
                cam_b=cam_b,
                frame_idx=frame_idx,
                crop_a=crop_a,
                crop_b=crop_b,
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                mconf=mconf,
            )

        if valid_matches == 0:
            return {
                "frame_idx": frame_idx,
                "status": "ok",
                "score": 0.0,
                "raw_matches": raw_matches,
                "valid_matches": 0,
                "mean_confidence": 0.0,
                "confidence_sum": 0.0,
                "match_count_score": 0.0,
                "reason": "no matches passed confidence threshold",
            }

        confidence_sum = float(sum(valid_conf))
        mean_confidence = confidence_sum / valid_matches
        match_count_score = clamp01(valid_matches / max(1.0, self.target_matches))

        frame_score = weighted_average(
            [
                (clamp01(mean_confidence), self.loftr_conf_weight),
                (match_count_score, self.loftr_count_weight),
            ]
        )

        return {
            "frame_idx": frame_idx,
            "status": "ok",
            "score": frame_score,
            "raw_matches": raw_matches,
            "valid_matches": valid_matches,
            "mean_confidence": mean_confidence,
            "confidence_sum": confidence_sum,
            "match_count_score": match_count_score,
        }

    # ------------------------------------------------------------------
    # LoFTR utilities
    # ------------------------------------------------------------------

    def _loftr_is_configured(self) -> bool:
        return bool(self.loftr_repo_path or self.loftr_weight_path)

    def _ensure_loftr(self) -> str | None:
        """Initialize LoFTR lazily.

        Returns
        -------
        None if matcher is ready, otherwise a reason string.
        """
        if self._matcher is not None:
            return None

        try:
            import torch  # type: ignore

            self._torch = torch
        except Exception as exc:  # noqa: BLE001
            return f"torch is required for LoFTR mode: {type(exc).__name__}: {exc}"

        if self.device == "cuda" and not self._torch.cuda.is_available():
            self.device = "cpu"

        if self.loftr_repo_path:
            repo_path = str(Path(self.loftr_repo_path).expanduser().resolve())
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        try:
            from src.loftr import LoFTR, default_cfg  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return (
                "Failed to import official LoFTR. Provide config['loftr_repo_path'] "
                f"pointing to the LoFTR repo. Error: {type(exc).__name__}: {exc}"
            )

        try:
            model = LoFTR(config=default_cfg)

            if not self.loftr_weight_path:
                return "config['loftr_weight_path'] is required for official LoFTR mode."

            weight_path = Path(str(self.loftr_weight_path)).expanduser().resolve()
            if not weight_path.exists():
                return f"LoFTR checkpoint not found: {weight_path}"

            ckpt = self._torch.load(str(weight_path), map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict)
            model.eval().to(self.device)

            self._matcher = model
            return None

        except Exception as exc:  # noqa: BLE001
            self._matcher = None
            return f"Failed to initialize LoFTR: {type(exc).__name__}: {exc}"

    def _read_frame(self, video_path: str, frame_idx: int) -> Any | None:
        """Read one RGB frame from a video using cv2."""
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

    def _preprocess_frame(self, frame_rgb: Any) -> Any:
        """Convert RGB frame to resized grayscale image."""
        cv2 = self._get_cv2()

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.resize_width, self.resize_height))
        return gray

    def _crop_edge(self, img: Any, side: str) -> Any:
        """Crop the likely overlapping edge region."""
        width = img.shape[1]
        crop_width = max(1, int(width * self.crop_ratio))

        if side == "left":
            return img[:, :crop_width]
        if side == "right":
            return img[:, width - crop_width :]

        # Fallback: no crop.
        return img

    def _to_tensor(self, gray: Any) -> Any:
        torch = self._torch
        if torch is None:
            raise RuntimeError("torch is not initialized")

        tensor = torch.from_numpy(gray.astype("float32")) / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def _match_loftr(self, gray_a: Any, gray_b: Any) -> dict[str, Any]:
        """Run LoFTR and return matched points plus confidence."""
        if self._matcher is None:
            raise RuntimeError("LoFTR matcher is not initialized")

        torch = self._torch
        if torch is None:
            raise RuntimeError("torch is not initialized")

        data = {
            "image0": self._to_tensor(gray_a),
            "image1": self._to_tensor(gray_b),
        }

        with torch.no_grad():
            self._matcher(data)

        mkpts0 = data.get("mkpts0_f")
        mkpts1 = data.get("mkpts1_f")
        mconf = data.get("mconf")

        if mkpts0 is None or mkpts1 is None or mconf is None:
            return {"mkpts0": [], "mkpts1": [], "mconf": []}

        if getattr(mconf, "ndim", None) == 0:
            return {"mkpts0": [], "mkpts1": [], "mconf": []}

        return {
            "mkpts0": mkpts0.detach().cpu().numpy(),
            "mkpts1": mkpts1.detach().cpu().numpy(),
            "mconf": mconf.detach().cpu().numpy(),
        }

    # ------------------------------------------------------------------
    # Visualization utilities
    # ------------------------------------------------------------------

    def _maybe_save_loftr_visualization(
        self,
        sample_id: str,
        cam_a: str,
        cam_b: str,
        frame_idx: int,
        crop_a: Any,
        crop_b: Any,
        mkpts0: Any,
        mkpts1: Any,
        mconf: Any,
    ) -> None:
        """Save a LoFTR match visualization if enabled and under limit."""
        if not self.save_visualizations:
            return

        if self._visualization_count >= self.max_visualizations:
            return

        try:
            output_dir = Path(self.visualization_dir) / sanitize_filename(sample_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{cam_a}__{cam_b}__frame_{frame_idx:06d}.jpg"
            output_path = output_dir / filename

            self._save_match_image(
                output_path=output_path,
                crop_a=crop_a,
                crop_b=crop_b,
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                mconf=mconf,
                title=f"{sample_id} | {cam_a} - {cam_b} | frame {frame_idx}",
            )

            self._visualization_count += 1

        except Exception:
            # Visualization must never break evaluation.
            return

    def _save_match_image(
        self,
        output_path: Path,
        crop_a: Any,
        crop_b: Any,
        mkpts0: Any,
        mkpts1: Any,
        mconf: Any,
        title: str = "",
    ) -> None:
        """Save side-by-side match visualization using cv2."""
        cv2 = self._get_cv2()

        if len(crop_a.shape) == 2:
            img_a = cv2.cvtColor(crop_a, cv2.COLOR_GRAY2BGR)
        else:
            img_a = crop_a.copy()

        if len(crop_b.shape) == 2:
            img_b = cv2.cvtColor(crop_b, cv2.COLOR_GRAY2BGR)
        else:
            img_b = crop_b.copy()

        h = max(img_a.shape[0], img_b.shape[0])
        w_a = img_a.shape[1]
        w_b = img_b.shape[1]

        canvas = self._np_zeros((h, w_a + w_b, 3))
        canvas[: img_a.shape[0], :w_a] = img_a
        canvas[: img_b.shape[0], w_a : w_a + w_b] = img_b

        mkpts0_list, mkpts1_list, mconf_list = self._filter_visual_matches(
            mkpts0=mkpts0,
            mkpts1=mkpts1,
            mconf=mconf,
        )

        for p0, p1, conf in zip(mkpts0_list, mkpts1_list, mconf_list):
            x0, y0 = int(round(float(p0[0]))), int(round(float(p0[1])))
            x1, y1 = int(round(float(p1[0]) + w_a)), int(round(float(p1[1])))

            color = confidence_to_bgr(float(conf))
            cv2.circle(canvas, (x0, y0), 2, color, -1)
            cv2.circle(canvas, (x1, y1), 2, color, -1)
            cv2.line(canvas, (x0, y0), (x1, y1), color, 1)

        if title:
            cv2.putText(
                canvas,
                title[:180],
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                title[:180],
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    def _filter_visual_matches(
        self,
        mkpts0: Any,
        mkpts1: Any,
        mconf: Any,
    ) -> tuple[list[Any], list[Any], list[float]]:
        """Filter matches for visualization."""
        if mkpts0 is None or mkpts1 is None or mconf is None:
            return [], [], []

        if len(mconf) == 0:
            return [], [], []

        selected = []
        for i, conf in enumerate(mconf):
            conf_f = float(conf)
            if conf_f >= self.visualize_min_conf:
                selected.append((i, conf_f))

        selected = sorted(selected, key=lambda x: x[1], reverse=True)[: self.max_visual_matches]

        mkpts0_list = [mkpts0[i] for i, _ in selected]
        mkpts1_list = [mkpts1[i] for i, _ in selected]
        conf_list = [conf for _, conf in selected]

        return mkpts0_list, mkpts1_list, conf_list

    def _np_zeros(self, shape: tuple[int, int, int]) -> Any:
        if self._np is None:
            import numpy as np  # type: ignore

            self._np = np
        return self._np.zeros(shape, dtype=self._np.uint8)

    # ------------------------------------------------------------------
    # Mode 3: precomputed evidence aggregation
    # ------------------------------------------------------------------

    def _has_precomputed_evidence(self, metadata: dict[str, Any]) -> bool:
        return any(
            key in metadata
            for key in (
                self.cross_view_scores_key,
                self.cross_view_confidence_key,
                self.cross_view_matches_key,
                self.cross_view_features_key,
            )
        )

    def _evaluate_precomputed(self, sample_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        scores: list[tuple[float, float]] = []

        direct_score = self._aggregate_numeric_evidence(metadata.get(self.cross_view_scores_key))
        if direct_score is not None:
            scores.append((direct_score, self.score_weight))

        confidence_score = self._aggregate_numeric_evidence(
            metadata.get(self.cross_view_confidence_key)
        )
        if confidence_score is not None:
            scores.append((confidence_score, self.score_weight))

        match_score = self._score_matches(metadata.get(self.cross_view_matches_key))
        if match_score is not None:
            scores.append((match_score, self.match_weight))

        feature_score = self._score_features(metadata.get(self.cross_view_features_key))
        if feature_score is not None:
            scores.append((feature_score, self.feature_weight))

        if not scores:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "mode": "precomputed",
                "score": None,
                "status": "skipped",
                "reason": "No precomputed cross-view scores, matches, confidence, or features found.",
            }

        final_score = weighted_average(scores)

        return {
            "sample_id": sample_id,
            "metric": self.name,
            "mode": "precomputed",
            "score": final_score,
            "status": "ok",
            "component_scores": {
                "direct_score": direct_score,
                "confidence_score": confidence_score,
                "match_score": match_score,
                "feature_score": feature_score,
            },
        }

    def _aggregate_numeric_evidence(self, value: Any) -> float | None:
        nums = flatten_numeric(value)
        if not nums:
            return None
        return clamp01(sum(nums) / len(nums))

    def _score_matches(self, matches: Any) -> float | None:
        if matches is None:
            return None

        if isinstance(matches, dict):
            payloads = list(matches.values())
        elif isinstance(matches, list):
            payloads = matches
        else:
            nums = flatten_numeric(matches)
            return clamp01(sum(nums) / len(nums)) if nums else None

        scores = []
        for payload in payloads:
            if isinstance(payload, (int, float)):
                if math.isfinite(float(payload)):
                    scores.append(clamp01(float(payload)))
                continue

            if not isinstance(payload, dict):
                nums = flatten_numeric(payload)
                if nums:
                    scores.append(clamp01(sum(nums) / len(nums)))
                continue

            confidence = first_numeric(
                payload,
                keys=("confidence", "mean_confidence", "score", "match_score"),
            )
            inlier_ratio = first_numeric(
                payload,
                keys=("inlier_ratio", "inlier_rate", "valid_ratio"),
            )
            valid_match_count = first_numeric(
                payload,
                keys=("valid_match_count", "num_valid_matches", "match_count", "num_matches"),
            )

            parts = []
            if confidence is not None:
                parts.append(clamp01(confidence))
            if inlier_ratio is not None:
                parts.append(clamp01(inlier_ratio))
            if valid_match_count is not None:
                parts.append(clamp01(valid_match_count / 100.0))

            if parts:
                scores.append(sum(parts) / len(parts))

        if not scores:
            return None

        return clamp01(sum(scores) / len(scores))

    def _score_features(self, features: Any) -> float | None:
        if features is None:
            return None

        if isinstance(features, dict):
            values = list(features.values())
            if values and all(isinstance(v, (int, float)) for v in values):
                nums = flatten_numeric(values)
                return clamp01(sum(nums) / len(nums)) if nums else None

            vectors = []
            for value in values:
                vec = to_float_vector(value)
                if vec:
                    vectors.append(vec)
            return pairwise_cosine_score(vectors)

        if isinstance(features, list):
            vectors = []
            for value in features:
                vec = to_float_vector(value)
                if vec:
                    vectors.append(vec)
            return pairwise_cosine_score(vectors)

        nums = flatten_numeric(features)
        if nums:
            return clamp01(sum(nums) / len(nums))

        return None

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _filter_camera_videos(self, camera_videos: dict[Any, Any]) -> dict[str, str]:
        return {
            str(view): str(path)
            for view, path in camera_videos.items()
            if str(view) not in self.exclude_views
        }

    def _resolve_expected_views(
        self,
        metadata: dict[str, Any],
        camera_videos: dict[str, str],
    ) -> list[str]:
        """Resolve expected camera views for integrity scoring."""
        if self.expected_views:
            views = [str(v) for v in self.expected_views]
            return [v for v in views if v not in self.exclude_views]

        metadata_views = metadata.get(self.views_key)
        if isinstance(metadata_views, list) and metadata_views:
            views = [str(v) for v in metadata_views if str(v) not in self.exclude_views]
        else:
            views = sorted(camera_videos.keys())

        if self.expected_num_views is not None:
            expected_n = int(self.expected_num_views)
            if expected_n > len(views):
                missing_count = expected_n - len(views)
                views = views + [f"__missing_view_{i}" for i in range(missing_count)]
            elif expected_n < len(views):
                views = views[:expected_n]

        return views

    def _resolve_camera_pairs(
        self,
        metadata: dict[str, Any],
        camera_videos: dict[str, str],
    ) -> list[tuple[str, str, str, str]]:
        """Resolve valid camera pairs for LoFTR."""
        raw_pairs = (
            metadata.get(self.camera_pairs_key)
            or self.config_camera_pairs
            or DEFAULT_CAMERA_PAIRS
        )

        parsed_pairs: list[tuple[str, str, str, str]] = []

        for pair in raw_pairs:
            if not isinstance(pair, (list, tuple)):
                continue

            if len(pair) >= 4:
                cam_a, cam_b, side_a, side_b = pair[:4]
            elif len(pair) == 2:
                cam_a, cam_b = pair
                side_a, side_b = self._infer_pair_sides(str(cam_a), str(cam_b))
            else:
                continue

            cam_a = str(cam_a)
            cam_b = str(cam_b)
            side_a = str(side_a)
            side_b = str(side_b)

            if cam_a in self.exclude_views or cam_b in self.exclude_views:
                continue

            if cam_a not in camera_videos or cam_b not in camera_videos:
                continue

            parsed_pairs.append((cam_a, cam_b, side_a, side_b))

        return parsed_pairs

    def _infer_pair_sides(self, cam_a: str, cam_b: str) -> tuple[str, str]:
        """Infer crop sides for known adjacent camera pairs."""
        for a, b, side_a, side_b in DEFAULT_CAMERA_PAIRS:
            if a == cam_a and b == cam_b:
                return side_a, side_b
            if a == cam_b and b == cam_a:
                return side_b, side_a
        return "right", "left"

    def _inspect_video(self, path: str) -> dict[str, Any]:
        """Inspect a video file using cv2 if available, imageio fallback otherwise."""
        video_path = Path(path)

        info: dict[str, Any] = {
            "path": str(video_path),
            "exists": video_path.exists(),
            "is_file": video_path.is_file(),
            "readable": False,
            "frame_count": None,
            "fps": None,
            "width": None,
            "height": None,
            "duration": None,
            "backend": None,
            "reason": None,
        }

        if not video_path.exists():
            info["reason"] = "path does not exist"
            return info

        if not video_path.is_file():
            info["reason"] = "path is not a file"
            return info

        try:
            cv2 = self._get_cv2()
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                info["reason"] = "cv2.VideoCapture failed to open video"
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

            readable = frame_count > 0 and width > 0 and height > 0

            info.update(
                {
                    "readable": readable,
                    "frame_count": frame_count,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "backend": "cv2",
                    "reason": None if readable else "invalid cv2 video metadata",
                }
            )
            return info

        except Exception as exc:  # noqa: BLE001
            info["reason"] = f"cv2 unavailable or failed: {type(exc).__name__}: {exc}"

        try:
            import imageio.v2 as imageio  # type: ignore

            reader = imageio.get_reader(str(video_path))
            meta = reader.get_meta_data() or {}

            fps = meta.get("fps")
            size = meta.get("size")
            duration = meta.get("duration")

            frame_count = None
            try:
                nframes = reader.count_frames()
                if isinstance(nframes, int) and nframes > 0:
                    frame_count = nframes
            except Exception:
                pass

            width = None
            height = None
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                width = int(size[0])
                height = int(size[1])

            reader.close()

            readable = bool(width and height)

            info.update(
                {
                    "readable": readable,
                    "frame_count": frame_count,
                    "fps": float(fps) if fps else None,
                    "width": width,
                    "height": height,
                    "duration": float(duration) if duration else None,
                    "backend": "imageio",
                    "reason": None if readable else "invalid imageio video metadata",
                }
            )
            return info

        except Exception as exc:  # noqa: BLE001
            info["reason"] = f"imageio failed: {type(exc).__name__}: {exc}"
            return info

    def _numeric_consistency_score(
        self,
        values: list[Any],
        tolerance: float = 0.05,
    ) -> float:
        """Return 0-1 consistency score for numeric values."""
        nums = []
        for value in values:
            if value is None:
                continue
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and x > 0:
                nums.append(x)

        if len(nums) <= 1:
            return 1.0 if len(nums) == 1 else 0.0

        median = statistics.median(nums)
        if median <= 0:
            return 0.0

        rel_devs = [abs(x - median) / median for x in nums]
        max_rel_dev = max(rel_devs)

        if max_rel_dev <= tolerance:
            return 1.0

        return clamp01(1.0 - (max_rel_dev - tolerance) / max(1e-8, 1.0 - tolerance))

    def _resolution_consistency_score(self, sizes: list[tuple[Any, Any]]) -> float:
        """Return 0-1 consistency score for width/height pairs."""
        valid_sizes = []
        for width, height in sizes:
            try:
                w = int(width)
                h = int(height)
            except (TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                valid_sizes.append((w, h))

        if not valid_sizes:
            return 0.0
        if len(valid_sizes) == 1:
            return 1.0

        most_common_count = max(valid_sizes.count(s) for s in set(valid_sizes))
        return safe_div(most_common_count, len(valid_sizes))

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

        # Uniform positions in the open interval, avoiding exact first/last frames.
        indices = []
        for i in range(n):
            pos = (i + 1) / (n + 1)
            idx = int(round(pos * (frame_count - 1)))
            indices.append(idx)

        return sorted(set(indices))

    def _get_cv2(self) -> Any:
        if self._cv2 is not None:
            return self._cv2
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
            return cv2
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"cv2 is required but unavailable: {exc}") from exc

# Legacy alias kept for compatibility with older imports.
ViewConsistency = ViewConsistencyMetric

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a) / float(b)

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

def flatten_numeric(value: Any) -> list[float]:
    nums: list[float] = []

    def visit(x: Any) -> None:
        if x is None:
            return

        if isinstance(x, bool):
            nums.append(float(x))
            return

        if isinstance(x, (int, float)):
            xf = float(x)
            if math.isfinite(xf):
                nums.append(xf)
            return

        if isinstance(x, dict):
            for v in x.values():
                visit(v)
            return

        if isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
            return

    visit(value)
    return nums

def first_numeric(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in payload:
            value = payload[key]
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x):
                return x
    return None

def to_float_vector(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []

    vec = []
    for item in value:
        try:
            x = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(x):
            return []
        vec.append(x)

    return vec

def cosine_similarity(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na <= 0 or nb <= 0:
        return None

    # Cosine is [-1, 1]. Convert to [0, 1].
    return clamp01((dot / (na * nb) + 1.0) / 2.0)

def pairwise_cosine_score(vectors: list[list[float]]) -> float | None:
    if len(vectors) < 2:
        return None

    scores = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity(vectors[i], vectors[j])
            if sim is not None:
                scores.append(sim)

    if not scores:
        return None

    return clamp01(sum(scores) / len(scores))

def sanitize_filename(name: str) -> str:
    allowed = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            allowed.append(ch)
        else:
            allowed.append("_")
    text = "".join(allowed)
    return text[:180] if len(text) > 180 else text

def confidence_to_bgr(conf: float) -> tuple[int, int, int]:
    """Map confidence in [0, 1] to a simple BGR color.

    Low confidence: red.
    High confidence: green.
    """
    c = clamp01(conf)
    green = int(255 * c)
    red = int(255 * (1.0 - c))
    blue = 64
    return (blue, green, red)
