from __future__ import annotations

from typing import Any

from .layout import FIXED_VIEW_ORDER, make_6v_montage_frame
from .video_io import inspect_video, read_first_frame

def build_integrity_overview_image(
    camera_videos: dict[str, str],
    *,
    tile_width: int,
    tile_height: int,
) -> Any:
    frames_by_view: dict[str, Any | None] = {}
    extra_lines: dict[str, list[str]] = {}

    for view_name in FIXED_VIEW_ORDER:
        path = camera_videos.get(view_name)
        if path is None:
            frames_by_view[view_name] = None
            extra_lines[view_name] = ["missing", "readable=no"]
            continue

        info = inspect_video(path)
        frame = read_first_frame(path) if info.get("readable") else None
        frames_by_view[view_name] = frame
        lines = [
            "readable=yes" if info.get("readable") else "readable=no",
            f"frames={info.get('frame_count') or 0}",
            f"fps={format_number(info.get('fps'))}",
            f"res={format_resolution(info)}",
        ]
        if info.get("reason") and not info.get("readable"):
            lines.append(str(info["reason"]))
        extra_lines[view_name] = lines

    return make_6v_montage_frame(
        frames_by_view,
        tile_width=tile_width,
        tile_height=tile_height,
        extra_lines_by_view=extra_lines,
    )

def format_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4g}"
    return "n/a"

def format_resolution(info: dict[str, Any]) -> str:
    width = info.get("width")
    height = info.get("height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return f"{width}x{height}"
    return "n/a"
