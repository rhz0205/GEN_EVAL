from __future__ import annotations

from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def ensure_visualization_layout(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    visualizations_dir = root / "visualizations"
    paths = {
        "visualizations_dir": visualizations_dir,
        "depth_raw": visualizations_dir / "depth_raw",
        "semantic_raw": visualizations_dir / "semantic_raw",
        "multiview_match_raw": visualizations_dir / "multiview_match_raw",
        "depth_6v_image": visualizations_dir / "depth_6v_image",
        "semantic_6v_image": visualizations_dir / "semantic_6v_image",
        "multiview_match_6v_image": visualizations_dir / "multiview_match_6v_image",
        "depth_6v_video": visualizations_dir / "depth_6v_video",
        "semantic_6v_video": visualizations_dir / "semantic_6v_video",
        "multiview_match_6v_video": visualizations_dir / "multiview_match_6v_video",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def collect_view_images(input_dir: str | Path, sample_id: str | None = None) -> list[Path]:
    raw_dir = Path(input_dir)
    if not raw_dir.exists() or not raw_dir.is_dir():
        return []
    matched = []
    sample_token = str(sample_id) if sample_id else None
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if sample_token and sample_token not in path.as_posix():
            continue
        matched.append(path)
    return sorted(matched)


def compose_6v_image(input_dir: str | Path, output_dir: str | Path, *, name: str) -> dict[str, Any]:
    raw_dir = Path(input_dir)
    target_dir = Path(output_dir)
    if not raw_dir.exists() or not raw_dir.is_dir():
        return {
            "name": name,
            "status": "skipped",
            "reason": f"Raw input directory is missing: {raw_dir}",
            "input_dir": str(raw_dir),
            "output_dir": str(target_dir),
        }
    source_files = collect_view_images(raw_dir)
    if not source_files:
        return {
            "name": name,
            "status": "skipped",
            "reason": f"No image files were found in {raw_dir}",
            "input_dir": str(raw_dir),
            "output_dir": str(target_dir),
        }
    target_dir.mkdir(parents=True, exist_ok=True)
    return {
        "name": name,
        "status": "skipped",
        "reason": "6-view image composition is not implemented for the current raw-data protocol.",
        "input_dir": str(raw_dir),
        "output_dir": str(target_dir),
        "num_source_files": len(source_files),
    }


def compose_6v_video(input_dir: str | Path, output_dir: str | Path, *, name: str) -> dict[str, Any]:
    raw_dir = Path(input_dir)
    target_dir = Path(output_dir)
    if not raw_dir.exists() or not raw_dir.is_dir():
        return {
            "name": name,
            "status": "skipped",
            "reason": f"Raw input directory is missing: {raw_dir}",
            "input_dir": str(raw_dir),
            "output_dir": str(target_dir),
        }
    source_files = sorted(
        path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not source_files:
        return {
            "name": name,
            "status": "skipped",
            "reason": f"No video files were found in {raw_dir}",
            "input_dir": str(raw_dir),
            "output_dir": str(target_dir),
        }
    target_dir.mkdir(parents=True, exist_ok=True)
    return {
        "name": name,
        "status": "skipped",
        "reason": "6-view video composition is not implemented for the current raw-data protocol.",
        "input_dir": str(raw_dir),
        "output_dir": str(target_dir),
        "num_source_files": len(source_files),
    }
