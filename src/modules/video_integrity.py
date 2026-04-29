from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

from modules.base import BaseModule

EXPECTED_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)

_CV2 = None


class VideoIntegrity(BaseModule):
    name = "video_integrity"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.camera_videos_key = self.config.get("camera_videos_key", "camera_videos")
        self.frame_count_tolerance = float(self.config.get("frame_count_tolerance", 0.05))
        self.fps_tolerance = float(self.config.get("fps_tolerance", 0.05))
        self.duration_tolerance = float(self.config.get("duration_tolerance", 0.05))

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        evaluated_samples: list[dict[str, Any]] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        valid_sample_count = 0
        invalid_sample_count = 0

        for sample in samples:
            sample_id = getattr(sample, "sample_id", None) or "unknown"
            try:
                sample_result = self._evaluate_sample(sample)
            except Exception as exc:
                failed_samples.append(
                    {
                        "sample_id": sample_id,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            evaluated_samples.append(sample_result)
            if sample_result.get("video_integrity_passed") is True:
                valid_sample_count += 1
            else:
                invalid_sample_count += 1

        evaluated_count = len(evaluated_samples)
        pass_rate = safe_div(valid_sample_count, evaluated_count) if evaluated_count > 0 else None

        if evaluated_count > 0:
            status = "success"
            reason = None
        elif failed_samples:
            status = "failed"
            reason = "No sample could be evaluated for video integrity."
        else:
            status = "skipped"
            reason = "No samples were provided."

        result: dict[str, Any] = {
            "metric": self.name,
            "status": status,
            "num_samples": len(samples),
            "valid_sample_count": valid_sample_count,
            "invalid_sample_count": invalid_sample_count,
            "skipped_sample_count": len(skipped_samples),
            "failed_sample_count": len(failed_samples),
            "pass_rate": pass_rate,
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
            return self._invalid_result(
                sample_id,
                "metadata['camera_videos'] is required and must be a non-empty dict.",
                [{"check": "camera_videos", "reason": "camera_videos is missing or empty."}],
            )

        normalized_videos = {
            str(view): str(path) for view, path in camera_videos.items() if path is not None
        }
        video_infos = {
            view: inspect_video(normalized_videos.get(view))
            if normalized_videos.get(view) is not None
            else missing_video_info()
            for view in EXPECTED_CAMERA_VIEWS
        }

        existing_views = [view for view, info in video_infos.items() if bool(info.get("exists"))]
        readable_views = [view for view, info in video_infos.items() if bool(info.get("readable"))]
        readable_infos = [video_infos[view] for view in EXPECTED_CAMERA_VIEWS]

        failed_checks: list[dict[str, str]] = []

        if len(existing_views) != len(EXPECTED_CAMERA_VIEWS):
            missing_views = [view for view in EXPECTED_CAMERA_VIEWS if view not in existing_views]
            failed_checks.append(
                {
                    "check": "presence",
                    "reason": f"Missing expected views: {', '.join(missing_views)}.",
                }
            )

        if len(readable_views) != len(EXPECTED_CAMERA_VIEWS):
            unreadable_views = [view for view in EXPECTED_CAMERA_VIEWS if view not in readable_views]
            failed_checks.append(
                {
                    "check": "readability",
                    "reason": f"Unreadable expected views: {', '.join(unreadable_views)}.",
                }
            )

        if not numeric_consistency_pass(
            [info.get("frame_count") for info in readable_infos],
            self.frame_count_tolerance,
            expected_count=len(EXPECTED_CAMERA_VIEWS),
        ):
            failed_checks.append(
                {
                    "check": "frame_count_consistency",
                    "reason": "Frame counts are inconsistent across expected views.",
                }
            )

        if not numeric_consistency_pass(
            [info.get("fps") for info in readable_infos],
            self.fps_tolerance,
            expected_count=len(EXPECTED_CAMERA_VIEWS),
        ):
            failed_checks.append(
                {
                    "check": "fps_consistency",
                    "reason": "FPS values are inconsistent across expected views.",
                }
            )

        if not numeric_consistency_pass(
            [info.get("duration") for info in readable_infos],
            self.duration_tolerance,
            expected_count=len(EXPECTED_CAMERA_VIEWS),
        ):
            failed_checks.append(
                {
                    "check": "duration_consistency",
                    "reason": "Durations are inconsistent across expected views.",
                }
            )

        if not resolution_consistency_pass(
            [(info.get("width"), info.get("height")) for info in readable_infos],
            expected_count=len(EXPECTED_CAMERA_VIEWS),
        ):
            failed_checks.append(
                {
                    "check": "resolution_consistency",
                    "reason": "Resolutions are inconsistent across expected views.",
                }
            )

        if failed_checks:
            return self._invalid_result(sample_id, "Video integrity checks failed.", failed_checks)

        return {
            "sample_id": sample_id,
            "status": "success",
            "video_integrity_passed": True,
        }

    def _invalid_result(self, sample_id: str, reason: str, failed_checks: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "status": "success",
            "video_integrity_passed": False,
            "reason": reason,
            "failed_checks": failed_checks,
        }


def missing_video_info() -> dict[str, Any]:
    return {
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


def inspect_video(path: str | None) -> dict[str, Any]:
    if not path:
        return missing_video_info()

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
        cv2 = get_cv2()
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
    except Exception as exc:
        info["reason"] = f"cv2 unavailable or failed: {type(exc).__name__}: {exc}"

    try:
        import imageio.v2 as imageio

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
    except Exception as exc:
        info["reason"] = f"imageio failed: {type(exc).__name__}: {exc}"
        return info


def get_cv2() -> Any:
    global _CV2
    if _CV2 is not None:
        return _CV2
    try:
        import cv2

        _CV2 = cv2
        return cv2
    except Exception as exc:
        raise RuntimeError(f"cv2 is required but unavailable: {exc}") from exc


def numeric_consistency_pass(values: list[Any], tolerance: float, expected_count: int) -> bool:
    nums = positive_finite_numbers(values)
    if len(nums) != expected_count:
        return False

    median = statistics.median(nums)
    if median <= 0:
        return False

    rel_devs = [abs(value - median) / median for value in nums]
    return max(rel_devs) <= tolerance


def resolution_consistency_pass(sizes: list[tuple[Any, Any]], expected_count: int) -> bool:
    valid_sizes = []
    for width, height in sizes:
        try:
            w = int(width)
            h = int(height)
        except (TypeError, ValueError):
            continue
        if w > 0 and h > 0:
            valid_sizes.append((w, h))

    if len(valid_sizes) != expected_count:
        return False
    return len(set(valid_sizes)) == 1


def positive_finite_numbers(values: list[Any]) -> list[float]:
    nums = []
    for value in values:
        if not is_finite_number(value):
            continue
        number = float(value)
        if number > 0:
            nums.append(number)
    return nums


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a) / float(b)
