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
    """View consistency metric exposing separate video-integrity and LoFTR scores."""

    name = "view_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

        self.camera_videos_key = self.config.get("camera_videos_key", "camera_videos")
        self.views_key = self.config.get("views_key", "views")
        self.camera_pairs_key = self.config.get("camera_pairs_key", "camera_pairs")

        self.expected_num_views = self.config.get("expected_num_views")
        self.expected_views = self.config.get("expected_views")

        self.keep_front_tele = bool(self.config.get("keep_front_tele", False))
        self.exclude_views = set(self.config.get("exclude_views", []))
        if not self.keep_front_tele:
            self.exclude_views.add("camera_front_tele")

        self.frame_count_tolerance = float(
            self.config.get("frame_count_tolerance", 0.05)
        )
        self.fps_tolerance = float(self.config.get("fps_tolerance", 0.05))
        self.duration_tolerance = float(self.config.get("duration_tolerance", 0.05))

        self.device = self.config.get("device", "cuda")
        self.loftr_repo_path = self.config.get("repo_path") or self.config.get(
            "loftr_repo_path"
        )
        self.loftr_weight_path = (
            self.config.get("weight_path")
            or self.config.get("loftr_weight_path")
            or self.config.get("local_save_path")
        )
        self.loftr_config = self.config.get("loftr_config", "outdoor")
        self.num_frames = int(self.config.get("num_frames", 3))
        self.frame_positions = self.config.get("frame_positions")

        resize = self.config.get("resize")
        if isinstance(resize, (list, tuple)) and len(resize) >= 2:
            self.resize_width = int(resize[0])
            self.resize_height = int(resize[1])
        else:
            self.resize_width = int(self.config.get("resize_width", 640))
            self.resize_height = int(self.config.get("resize_height", 480))

        self.crop_ratio = float(self.config.get("crop_ratio", 1.0))
        self.conf_threshold = float(self.config.get("conf_threshold", 0.0))
        self.min_valid_matches = int(self.config.get("min_valid_matches", 20))
        self.min_mean_confidence = float(
            self.config.get("min_mean_confidence", 0.2)
        )
        self.min_pair_pass_rate = float(self.config.get("min_pair_pass_rate", 0.5))
        self.max_pairs_per_sample = self.config.get("max_pairs_per_sample")
        self.fail_if_loftr_unavailable = bool(
            self.config.get("fail_if_loftr_unavailable", False)
        )

        self.config_camera_pairs = self.config.get("camera_pairs")

        self.save_visualizations = bool(
            self.config.get("save_visualizations", False)
        )
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
        evaluated_samples: list[dict[str, Any]] = []
        valid_video_integrity_scores: list[float] = []
        valid_loftr_scores: list[float] = []
        valid_loftr_raw_scores: list[float] = []
        valid_loftr_coverage_scores: list[float] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        evaluated_pair_count = 0
        total_pairs_expected = 0
        skipped_pair_count = 0

        for sample in samples:
            sample_id = getattr(sample, "sample_id", None) or "unknown"
            try:
                sample_result = self._evaluate_sample(sample)
            except Exception as exc:  # noqa: BLE001
                sample_result = {
                    "sample_id": sample_id,
                    "metric": self.name,
                    "score": None,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "video_integrity_score": None,
                    "loftr_score": None,
                    "loftr_raw_score": None,
                    "loftr_coverage_score": None,
                    "video_integrity": {
                        "status": "failed",
                        "score": None,
                        "video_integrity_passed": False,
                        "failed_checks": [],
                    },
                    "loftr": {
                        "status": "failed",
                        "score": None,
                    },
                    "available_subscores": [],
                    "skipped_subscores": [],
                    "num_views": None,
                    "view_results": [],
                    "pair_results": [],
                    "evaluated_pair_count": 0,
                    "total_pairs_expected": 0,
                    "skipped_pair_count": 0,
                }

            evaluated_samples.append(sample_result)

            video_integrity_score = sample_result.get("video_integrity_score")
            loftr_score = sample_result.get("loftr_score")
            loftr_raw_score = sample_result.get("loftr_raw_score")
            loftr_coverage_score = sample_result.get("loftr_coverage_score")

            if is_finite_number(video_integrity_score):
                valid_video_integrity_scores.append(float(video_integrity_score))
            if is_finite_number(loftr_score):
                valid_loftr_scores.append(float(loftr_score))
            if is_finite_number(loftr_raw_score):
                valid_loftr_raw_scores.append(float(loftr_raw_score))
            if is_finite_number(loftr_coverage_score):
                valid_loftr_coverage_scores.append(float(loftr_coverage_score))

            evaluated_pair_count += int(sample_result.get("evaluated_pair_count") or 0)
            total_pairs_expected += int(sample_result.get("total_pairs_expected") or 0)
            skipped_pair_count += int(sample_result.get("skipped_pair_count") or 0)

            status = sample_result.get("status")
            if status == "skipped":
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

        video_integrity_score = mean_or_none(valid_video_integrity_scores)
        loftr_score = mean_or_none(valid_loftr_scores)
        loftr_raw_score = mean_or_none(valid_loftr_raw_scores)
        loftr_coverage_score = mean_or_none(valid_loftr_coverage_scores)

        if video_integrity_score is not None or loftr_score is not None:
            status = "success"
            reason = None
        else:
            status = "failed" if failed_samples else "skipped"
            reason = "No sample produced a valid view_consistency score."

        result = {
            "metric": self.name,
            "score": None,
            "status": status,
            "num_samples": len(samples),
            "video_integrity_score": video_integrity_score,
            "video_integrity_num_samples": len(valid_video_integrity_scores),
            "loftr_score": loftr_score,
            "loftr_raw_score": loftr_raw_score,
            "loftr_coverage_score": loftr_coverage_score,
            "loftr_num_samples": len(valid_loftr_scores),
            "evaluated_pair_count": evaluated_pair_count,
            "total_pairs_expected": total_pairs_expected,
            "skipped_pair_count": skipped_pair_count,
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

        video_integrity = self._evaluate_video_integrity(sample_id, metadata)
        if (
            video_integrity.get("video_integrity_passed") is True
            and video_integrity.get("score") == 1.0
        ):
            loftr = self._evaluate_loftr(sample_id, metadata)
        else:
            expected_pairs = self._resolve_expected_camera_pairs(metadata)
            if self.max_pairs_per_sample is not None:
                expected_pairs = expected_pairs[: int(self.max_pairs_per_sample)]
            loftr = {
                "sample_id": sample_id,
                "metric": self.name,
                "score": None,
                "status": "skipped",
                "reason": "Skipped because video_integrity did not pass.",
                "loftr_raw_score": None,
                "loftr_coverage_score": None,
                "evaluated_pair_count": 0,
                "total_pairs_expected": len(expected_pairs),
                "skipped_pair_count": len(expected_pairs),
                "pair_results": [],
            }
        return self._build_sample_result(sample_id, video_integrity, loftr)

    def _build_sample_result(
        self,
        sample_id: str,
        video_integrity: dict[str, Any],
        loftr: dict[str, Any],
    ) -> dict[str, Any]:
        video_integrity_score = numeric_or_none(video_integrity.get("score"))
        loftr_score = numeric_or_none(loftr.get("score"))
        loftr_raw_score = numeric_or_none(loftr.get("loftr_raw_score"))
        loftr_coverage_score = numeric_or_none(loftr.get("loftr_coverage_score"))

        available_subscores: list[str] = []
        skipped_subscores: list[dict[str, Any]] = []

        if video_integrity_score is not None:
            available_subscores.append("video_integrity")
        else:
            skipped_subscores.append(
                {
                    "name": "video_integrity",
                    "status": video_integrity.get("status", "skipped"),
                    "reason": video_integrity.get("reason"),
                }
            )

        if loftr_score is not None:
            available_subscores.append("loftr")
        else:
            skipped_subscores.append(
                {
                    "name": "loftr",
                    "status": loftr.get("status", "skipped"),
                    "reason": loftr.get("reason"),
                }
            )

        status = "success" if available_subscores else self._combine_missing_status(
            skipped_subscores
        )
        result = {
            "sample_id": sample_id,
            "metric": self.name,
            "score": None,
            "status": status,
            "video_integrity_score": video_integrity_score,
            "loftr_score": loftr_score,
            "loftr_raw_score": loftr_raw_score,
            "loftr_coverage_score": loftr_coverage_score,
            "video_integrity": self._strip_metric_identity(video_integrity),
            "loftr": self._strip_metric_identity(loftr),
            "available_subscores": available_subscores,
            "skipped_subscores": skipped_subscores,
            "num_views": video_integrity.get("num_views"),
            "view_results": video_integrity.get("view_results", []),
            "pair_results": loftr.get("pair_results", []),
            "evaluated_pair_count": int(loftr.get("evaluated_pair_count") or 0),
            "total_pairs_expected": int(loftr.get("total_pairs_expected") or 0),
            "skipped_pair_count": int(loftr.get("skipped_pair_count") or 0),
        }

        reasons = [str(item.get("reason")) for item in skipped_subscores if item.get("reason")]
        if reasons:
            result["reason"] = "; ".join(reasons)
        return result

    def _combine_missing_status(self, items: list[dict[str, Any]]) -> str:
        if any(str(item.get("status")) == "failed" for item in items):
            return "failed"
        return "skipped"

    def _strip_metric_identity(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in result.items()
            if key not in {"sample_id", "metric"}
        }

    def _evaluate_video_integrity(
        self, sample_id: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        camera_videos = metadata.get(self.camera_videos_key)
        if not isinstance(camera_videos, dict) or not camera_videos:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "score": None,
                "status": "skipped",
                "reason": "metadata['camera_videos'] is required for video_integrity.",
                "num_views": 0,
                "view_count": 0,
                "expected_num_views": 0,
                "existing_view_count": 0,
                "readable_view_count": 0,
                "component_scores": {},
                "video_infos": {},
                "view_results": [],
                "video_integrity_passed": False,
                "failed_checks": [
                    {
                        "check": "camera_videos",
                        "reason": "metadata['camera_videos'] is required for video_integrity.",
                    }
                ],
            }

        camera_videos = self._filter_camera_videos(camera_videos)
        if not camera_videos:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "score": None,
                "status": "skipped",
                "reason": "No usable camera videos after view filtering.",
                "num_views": 0,
                "view_count": 0,
                "expected_num_views": 0,
                "existing_view_count": 0,
                "readable_view_count": 0,
                "component_scores": {},
                "video_infos": {},
                "view_results": [],
                "video_integrity_passed": False,
                "failed_checks": [
                    {
                        "check": "camera_videos",
                        "reason": "No usable camera videos after view filtering.",
                    }
                ],
            }

        expected_views = self._resolve_expected_views(metadata, camera_videos)
        expected_num_views = len(expected_views)

        video_infos: dict[str, dict[str, Any]] = {}
        view_results: list[dict[str, Any]] = []
        for view in expected_views:
            path = camera_videos.get(view)
            if path is None:
                info = {
                    "path": None,
                    "exists": False,
                    "is_file": False,
                    "readable": False,
                    "frame_count": None,
                    "fps": None,
                    "width": None,
                    "height": None,
                    "duration": None,
                    "backend": None,
                    "reason": "missing from camera_videos",
                }
            else:
                info = self._inspect_video(str(path))
            video_infos[view] = info
            view_results.append(
                {
                    "view": view,
                    "exists": bool(info.get("exists")),
                    "readable": bool(info.get("readable")),
                    "frame_count": info.get("frame_count"),
                    "fps": info.get("fps"),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "duration": info.get("duration"),
                    "reason": info.get("reason"),
                }
            )

        existing_views = [
            view for view, info in video_infos.items() if bool(info.get("exists"))
        ]
        readable_views = [
            view for view, info in video_infos.items() if bool(info.get("readable"))
        ]
        readable_infos = [video_infos[view] for view in readable_views]

        failed_checks: list[dict[str, str]] = []

        presence_passed = expected_num_views > 0 and len(existing_views) == expected_num_views
        if not presence_passed:
            failed_checks.append(
                {
                    "check": "presence",
                    "reason": f"Expected {expected_num_views} views but found {len(existing_views)} existing views.",
                }
            )

        readability_passed = expected_num_views > 0 and len(readable_views) == expected_num_views
        if not readability_passed:
            failed_checks.append(
                {
                    "check": "readability",
                    "reason": f"Expected {expected_num_views} readable views but found {len(readable_views)} readable views.",
                }
            )

        frame_count_passed = self._numeric_consistency_pass(
            [info.get("frame_count") for info in readable_infos],
            tolerance=self.frame_count_tolerance,
        )
        if not frame_count_passed:
            failed_checks.append(
                {
                    "check": "frame_count_consistency",
                    "reason": "Frame counts are inconsistent across readable views.",
                }
            )

        fps_passed = self._numeric_consistency_pass(
            [info.get("fps") for info in readable_infos],
            tolerance=self.fps_tolerance,
        )
        if not fps_passed:
            failed_checks.append(
                {
                    "check": "fps_consistency",
                    "reason": "FPS values are inconsistent across readable views.",
                }
            )

        duration_passed = self._numeric_consistency_pass(
            [info.get("duration") for info in readable_infos],
            tolerance=self.duration_tolerance,
        )
        if not duration_passed:
            failed_checks.append(
                {
                    "check": "duration_consistency",
                    "reason": "Durations are inconsistent across readable views.",
                }
            )

        resolution_passed = self._resolution_consistency_pass(
            [(info.get("width"), info.get("height")) for info in readable_infos]
        )
        if not resolution_passed:
            failed_checks.append(
                {
                    "check": "resolution_consistency",
                    "reason": "Resolutions are inconsistent across readable views.",
                }
            )

        video_integrity_passed = not failed_checks
        score = 1.0 if video_integrity_passed else 0.0
        result = {
            "sample_id": sample_id,
            "metric": self.name,
            "score": score,
            "status": "success",
            "num_views": len(camera_videos),
            "view_count": len(camera_videos),
            "expected_num_views": expected_num_views,
            "existing_view_count": len(existing_views),
            "readable_view_count": len(readable_views),
            "component_scores": {
                "presence": 1.0 if presence_passed else 0.0,
                "readability": 1.0 if readability_passed else 0.0,
                "frame_count_consistency": 1.0 if frame_count_passed else 0.0,
                "resolution_consistency": 1.0 if resolution_passed else 0.0,
                "fps_consistency": 1.0 if fps_passed else 0.0,
                "duration_consistency": 1.0 if duration_passed else 0.0,
            },
            "video_infos": video_infos,
            "view_results": view_results,
            "video_integrity_passed": video_integrity_passed,
            "failed_checks": failed_checks,
        }
        if not video_integrity_passed:
            result["reason"] = "Video integrity checks did not pass."
        return result

    def _evaluate_loftr(self, sample_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        camera_pairs = self._resolve_expected_camera_pairs(metadata)
        if self.max_pairs_per_sample is not None:
            camera_pairs = camera_pairs[: int(self.max_pairs_per_sample)]
        total_pairs_expected = len(camera_pairs)

        camera_videos = metadata.get(self.camera_videos_key)
        if not isinstance(camera_videos, dict) or not camera_videos:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "score": None,
                "status": "skipped",
                "reason": "metadata['camera_videos'] is required for LoFTR evaluation.",
                "loftr_raw_score": None,
                "loftr_coverage_score": None,
                "evaluated_pair_count": 0,
                "total_pairs_expected": total_pairs_expected,
                "skipped_pair_count": 0,
                "pair_results": [],
            }

        camera_videos = self._filter_camera_videos(camera_videos)
        if not camera_videos:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "score": None,
                "status": "skipped",
                "reason": "No usable camera videos after view filtering.",
                "loftr_raw_score": None,
                "loftr_coverage_score": None,
                "evaluated_pair_count": 0,
                "total_pairs_expected": total_pairs_expected,
                "skipped_pair_count": 0,
                "pair_results": [],
            }

        if not self._loftr_is_configured():
            status = "failed" if self.fail_if_loftr_unavailable else "skipped"
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "score": None,
                "status": status,
                "reason": "LoFTR repo_path/weight_path is not configured.",
                "loftr_raw_score": None,
                "loftr_coverage_score": None,
                "evaluated_pair_count": 0,
                "total_pairs_expected": total_pairs_expected,
                "skipped_pair_count": 0,
                "pair_results": [],
            }

        matcher_status = self._ensure_loftr()
        if matcher_status is not None:
            status = "failed" if self.fail_if_loftr_unavailable else "skipped"
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "score": None,
                "status": status,
                "reason": matcher_status,
                "loftr_raw_score": None,
                "loftr_coverage_score": None,
                "evaluated_pair_count": 0,
                "total_pairs_expected": total_pairs_expected,
                "skipped_pair_count": 0,
                "pair_results": [],
            }

        if not camera_pairs:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "score": None,
                "status": "skipped",
                "reason": "No expected camera pairs available for LoFTR evaluation.",
                "loftr_raw_score": None,
                "loftr_coverage_score": None,
                "evaluated_pair_count": 0,
                "total_pairs_expected": 0,
                "skipped_pair_count": 0,
                "pair_results": [],
            }

        pair_results: list[dict[str, Any]] = []
        pair_scores: list[float] = []

        for cam_a, cam_b, side_a, side_b in camera_pairs:
            path_a = camera_videos.get(cam_a)
            path_b = camera_videos.get(cam_b)
            if not path_a or not path_b:
                pair_results.append(
                    {
                        "pair": f"{cam_a}|{cam_b}",
                        "camera_a": cam_a,
                        "camera_b": cam_b,
                        "side_a": side_a,
                        "side_b": side_b,
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
                pair_result.get("status") == "success"
                and is_finite_number(pair_result.get("score"))
            ):
                pair_scores.append(float(pair_result["score"]))

        evaluated_pair_count = len(pair_scores)
        skipped_pair_count = len(
            [item for item in pair_results if item.get("status") != "success"]
        )

        if not pair_scores:
            return {
                "sample_id": sample_id,
                "metric": self.name,
                "score": 0.0,
                "status": "success",
                "reason": "No camera pair produced a valid LoFTR score; assigned zero due to zero pair coverage.",
                "loftr_raw_score": None,
                "loftr_coverage_score": 0.0,
                "pair_results": pair_results,
                "num_pairs": 0,
                "evaluated_pair_count": 0,
                "total_pairs_expected": total_pairs_expected,
                "skipped_pair_count": skipped_pair_count,
                "total_valid_matches": 0,
                "mean_confidence": 0.0,
            }

        raw_score = float(sum(pair_scores) / len(pair_scores))
        coverage_score = safe_div(evaluated_pair_count, total_pairs_expected)
        loftr_score = raw_score * coverage_score
        total_conf = sum(float(item.get("confidence_sum", 0.0)) for item in pair_results)
        total_matches = sum(int(item.get("valid_matches", 0)) for item in pair_results)
        mean_confidence = total_conf / total_matches if total_matches > 0 else 0.0

        return {
            "sample_id": sample_id,
            "metric": self.name,
            "score": loftr_score,
            "loftr_raw_score": raw_score,
            "loftr_coverage_score": coverage_score,
            "status": "success",
            "num_pairs": evaluated_pair_count,
            "evaluated_pair_count": evaluated_pair_count,
            "total_pairs_expected": total_pairs_expected,
            "skipped_pair_count": skipped_pair_count,
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
                "camera_a": cam_a,
                "camera_b": cam_b,
                "side_a": side_a,
                "side_b": side_b,
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
                "camera_a": cam_a,
                "camera_b": cam_b,
                "side_a": side_a,
                "side_b": side_b,
                "status": "skipped",
                "reason": "invalid frame count",
                "video_info_a": info_a,
                "video_info_b": info_b,
            }

        frame_indices = self._sample_frame_indices(frame_count)
        frame_results: list[dict[str, Any]] = []
        evaluated_frames: list[dict[str, Any]] = []
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
                frame_result.get("status") == "success"
                and is_finite_number(frame_result.get("score"))
            ):
                evaluated_frames.append(frame_result)
                confidence_sum += float(frame_result.get("confidence_sum", 0.0))
                valid_matches_total += int(frame_result.get("valid_matches", 0))

        if not evaluated_frames:
            return {
                "pair": f"{cam_a}|{cam_b}",
                "camera_a": cam_a,
                "camera_b": cam_b,
                "side_a": side_a,
                "side_b": side_b,
                "status": "skipped",
                "score": None,
                "reason": "No sampled frame could be evaluated for LoFTR.",
                "num_sampled_frames": len(frame_indices),
                "frame_results": frame_results,
                "video_info_a": info_a,
                "video_info_b": info_b,
            }

        num_evaluated_frames = len(evaluated_frames)
        passed_frame_count = sum(
            1 for item in evaluated_frames if item.get("frame_passed") is True
        )
        pair_score = safe_div(passed_frame_count, num_evaluated_frames)
        pair_passed = pair_score >= self.min_pair_pass_rate
        mean_confidence = (
            confidence_sum / valid_matches_total if valid_matches_total > 0 else 0.0
        )
        mean_valid_matches = valid_matches_total / num_evaluated_frames

        return {
            "pair": f"{cam_a}|{cam_b}",
            "camera_a": cam_a,
            "camera_b": cam_b,
            "side_a": side_a,
            "side_b": side_b,
            "status": "success",
            "score": pair_score,
            "pair_passed": pair_passed,
            "passed_frame_count": passed_frame_count,
            "num_valid_frames": num_evaluated_frames,
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
                "score": None,
                "reason": "failed to read one or both frames",
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
                "status": "success",
                "score": 0.0,
                "raw_matches": 0,
                "valid_matches": 0,
                "mean_confidence": 0.0,
                "confidence_sum": 0.0,
                "frame_passed": False,
                "reason": "no LoFTR matches",
            }

        valid_indices = [
            i for i, value in enumerate(mconf) if float(value) >= self.conf_threshold
        ]
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
                "status": "success",
                "score": 0.0,
                "raw_matches": raw_matches,
                "valid_matches": 0,
                "mean_confidence": 0.0,
                "confidence_sum": 0.0,
                "frame_passed": False,
                "reason": "no matches passed confidence threshold",
            }

        confidence_sum = float(sum(valid_conf))
        mean_confidence = confidence_sum / valid_matches
        frame_passed = (
            valid_matches >= self.min_valid_matches
            and mean_confidence >= self.min_mean_confidence
        )
        frame_score = 1.0 if frame_passed else 0.0

        result = {
            "frame_idx": frame_idx,
            "status": "success",
            "score": frame_score,
            "raw_matches": raw_matches,
            "valid_matches": valid_matches,
            "mean_confidence": mean_confidence,
            "confidence_sum": confidence_sum,
            "frame_passed": frame_passed,
        }
        if not frame_passed:
            result["reason"] = (
                "frame did not meet LoFTR thresholds: "
                f"valid_matches={valid_matches} < {self.min_valid_matches} or "
                f"mean_confidence={mean_confidence:.4f} < {self.min_mean_confidence:.4f}"
            )
        return result

    def _loftr_is_configured(self) -> bool:
        return bool(self.loftr_repo_path and self.loftr_weight_path)

    def _ensure_loftr(self) -> str | None:
        if self._matcher is not None:
            return None

        try:
            import torch  # type: ignore

            self._torch = torch
        except Exception as exc:  # noqa: BLE001
            return f"torch is required for LoFTR evaluation: {type(exc).__name__}: {exc}"

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

            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def _preprocess_frame(self, frame_rgb: Any) -> Any:
        cv2 = self._get_cv2()
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.resize_width, self.resize_height))

    def _crop_edge(self, img: Any, side: str) -> Any:
        width = img.shape[1]
        crop_width = max(1, int(width * self.crop_ratio))
        if side == "left":
            return img[:, :crop_width]
        if side == "right":
            return img[:, width - crop_width :]
        return img

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
            return {"mkpts0": [], "mkpts1": [], "mconf": []}
        if getattr(mconf, "ndim", None) == 0:
            return {"mkpts0": [], "mkpts1": [], "mconf": []}

        return {
            "mkpts0": mkpts0.detach().cpu().numpy(),
            "mkpts1": mkpts1.detach().cpu().numpy(),
            "mconf": mconf.detach().cpu().numpy(),
        }

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
        if not self.save_visualizations:
            return
        if self._visualization_count >= self.max_visualizations:
            return

        try:
            output_dir = Path(self.visualization_dir) / sanitize_filename(sample_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{cam_a}__{cam_b}__frame_{frame_idx:06d}.jpg"
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
        if mkpts0 is None or mkpts1 is None or mconf is None:
            return [], [], []
        if len(mconf) == 0:
            return [], [], []

        selected = []
        for index, conf in enumerate(mconf):
            conf_f = float(conf)
            if conf_f >= self.visualize_min_conf:
                selected.append((index, conf_f))

        selected = sorted(selected, key=lambda item: item[1], reverse=True)[
            : self.max_visual_matches
        ]
        return (
            [mkpts0[index] for index, _ in selected],
            [mkpts1[index] for index, _ in selected],
            [conf for _, conf in selected],
        )

    def _np_zeros(self, shape: tuple[int, int, int]) -> Any:
        if self._np is None:
            import numpy as np  # type: ignore

            self._np = np
        return self._np.zeros(shape, dtype=self._np.uint8)

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
        if self.expected_views:
            views = [str(view) for view in self.expected_views]
            return [view for view in views if view not in self.exclude_views]

        metadata_views = metadata.get(self.views_key)
        if isinstance(metadata_views, list) and metadata_views:
            views = [
                str(view)
                for view in metadata_views
                if str(view) not in self.exclude_views
            ]
        else:
            views = sorted(camera_videos.keys())

        if self.expected_num_views is not None:
            expected_n = int(self.expected_num_views)
            if expected_n > len(views):
                missing_count = expected_n - len(views)
                views = views + [
                    f"__missing_view_{index}" for index in range(missing_count)
                ]
            elif expected_n < len(views):
                views = views[:expected_n]
        return views

    def _resolve_expected_camera_pairs(
        self,
        metadata: dict[str, Any],
    ) -> list[tuple[str, str, str, str]]:
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
            parsed_pairs.append((cam_a, cam_b, side_a, side_b))
        return parsed_pairs

    def _infer_pair_sides(self, cam_a: str, cam_b: str) -> tuple[str, str]:
        for view_a, view_b, side_a, side_b in DEFAULT_CAMERA_PAIRS:
            if view_a == cam_a and view_b == cam_b:
                return side_a, side_b
            if view_a == cam_b and view_b == cam_a:
                return side_b, side_a
        return "right", "left"

    def _inspect_video(self, path: str) -> dict[str, Any]:
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

    def _numeric_consistency_pass(
        self,
        values: list[Any],
        tolerance: float = 0.05,
    ) -> bool:
        nums = positive_finite_numbers(values)
        if len(nums) <= 1:
            return len(nums) == 1

        median = statistics.median(nums)
        if median <= 0:
            return False
        rel_devs = [abs(value - median) / median for value in nums]
        return max(rel_devs) <= tolerance

    def _resolution_consistency_pass(self, sizes: list[tuple[Any, Any]]) -> bool:
        valid_sizes = []
        for width, height in sizes:
            try:
                w = int(width)
                h = int(height)
            except (TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                valid_sizes.append((w, h))

        if len(valid_sizes) <= 1:
            return len(valid_sizes) == 1
        return len(set(valid_sizes)) == 1

    def _sample_frame_indices(self, frame_count: int) -> list[int]:
        if frame_count <= 0:
            return []

        if self.frame_positions:
            indices = []
            for pos in self.frame_positions:
                try:
                    normalized = float(pos)
                except (TypeError, ValueError):
                    continue
                normalized = max(0.0, min(1.0, normalized))
                indices.append(int(round(normalized * (frame_count - 1))))
            return sorted(set(indices))

        n = max(1, self.num_frames)
        if n == 1:
            return [frame_count // 2]

        indices = []
        for index in range(n):
            pos = (index + 1) / (n + 1)
            indices.append(int(round(pos * (frame_count - 1))))
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


ViewConsistency = ViewConsistencyMetric


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a) / float(b)


def clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def numeric_or_none(value: Any) -> float | None:
    if is_finite_number(value):
        return float(value)
    return None


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def positive_finite_numbers(values: list[Any]) -> list[float]:
    nums = []
    for value in values:
        if not is_finite_number(value):
            continue
        number = float(value)
        if number > 0:
            nums.append(number)
    return nums


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
    c = clamp01(conf)
    green = int(255 * c)
    red = int(255 * (1.0 - c))
    blue = 64
    return (blue, green, red)
