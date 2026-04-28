from __future__ import annotations

from typing import Any

from .video_io import blank_frame, draw_text_box, resize_frame

FIXED_VIEW_ORDER: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)

def make_labeled_tile(
    frame_rgb: Any | None,
    *,
    view_name: str,
    tile_width: int,
    tile_height: int,
    extra_lines: list[str] | None = None,
) -> Any:
    tile = (
        resize_frame(frame_rgb, tile_width, tile_height)
        if frame_rgb is not None
        else blank_frame(tile_width, tile_height)
    )
    lines = [view_name]
    if extra_lines:
        lines.extend(extra_lines)
    return draw_text_box(tile, lines)

def make_6v_montage_frame(
    tiles_by_view: dict[str, Any | None],
    *,
    tile_width: int,
    tile_height: int,
    extra_lines_by_view: dict[str, list[str]] | None = None,
) -> Any:
    import numpy as np  # type: ignore

    rows: list[Any] = []
    for row_index in range(2):
        row_tiles = []
        for col_index in range(3):
            view_name = FIXED_VIEW_ORDER[row_index * 3 + col_index]
            frame_rgb = tiles_by_view.get(view_name)
            row_tiles.append(
                make_labeled_tile(
                    frame_rgb,
                    view_name=view_name,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    extra_lines=(extra_lines_by_view or {}).get(view_name),
                )
            )
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)
