from __future__ import annotations

from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def describe_video(path: str | Path) -> dict[str, object]:
    video_path = Path(path)
    return {
        "path": str(video_path),
        "exists": video_path.exists(),
        "suffix": video_path.suffix.lower(),
    }
