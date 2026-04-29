from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")
EXPECTED_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)


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
    sample_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    if not sample_dirs:
        return {
            "name": name,
            "status": "skipped",
            "reason": f"No sample directories were found in {raw_dir}",
            "input_dir": str(raw_dir),
            "output_dir": str(target_dir),
        }
    target_dir.mkdir(parents=True, exist_ok=True)
    composed_count = 0
    skipped_samples: list[dict[str, str]] = []

    for sample_dir in sample_dirs:
        view_paths = collect_sample_view_images(sample_dir)
        missing_views = [view for view in EXPECTED_CAMERA_VIEWS if view not in view_paths]
        if missing_views:
            skipped_samples.append(
                {
                    "sample_id": sample_dir.name,
                    "reason": f"missing raw images for views: {', '.join(missing_views)}",
                }
            )
            continue

        images = [Image.open(view_paths[view]).convert("RGB") for view in EXPECTED_CAMERA_VIEWS]
        try:
            tile_width, tile_height = images[0].size
            canvas = Image.new("RGB", (tile_width * 3, tile_height * 2))
            for index, image in enumerate(images):
                if image.size != (tile_width, tile_height):
                    image = image.resize((tile_width, tile_height))
                row = index // 3
                col = index % 3
                canvas.paste(image, (col * tile_width, row * tile_height))
            canvas.save(target_dir / f"{sample_dir.name}.png")
            composed_count += 1
        finally:
            for image in images:
                image.close()

    if composed_count == 0:
        return {
            "name": name,
            "status": "skipped",
            "reason": "No samples produced a valid 6-view image composition.",
            "input_dir": str(raw_dir),
            "output_dir": str(target_dir),
            "num_samples": len(sample_dirs),
            "num_composed": 0,
            "skipped_samples": skipped_samples,
        }

    payload = {
        "name": name,
        "status": "partial" if skipped_samples else "success",
        "input_dir": str(raw_dir),
        "output_dir": str(target_dir),
        "num_samples": len(sample_dirs),
        "num_composed": composed_count,
        "skipped_samples": skipped_samples,
    }
    if skipped_samples:
        payload["reason"] = skipped_samples[0]["reason"]
    return payload


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


def collect_sample_view_images(sample_dir: str | Path) -> dict[str, Path]:
    root = Path(sample_dir)
    matched: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        for view in EXPECTED_CAMERA_VIEWS:
            if view in path.stem and view not in matched:
                matched[view] = path
                break
    return matched
