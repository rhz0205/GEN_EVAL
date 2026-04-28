from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .layout import FIXED_VIEW_ORDER, make_6v_montage_frame
from .video_io import read_all_frames, resize_frame, write_video

def build_semantic_outputs(
    sample_id: str,
    metadata: dict[str, Any],
    output_dir: str | Path,
    *,
    tile_width: int,
    tile_height: int,
) -> tuple[bool, str]:
    camera_videos = metadata.get("camera_videos")
    if not isinstance(camera_videos, dict) or not camera_videos:
        return False, "metadata['camera_videos'] is required."

    output_root = Path(output_dir)
    per_view_frames: dict[str, list[Any]] = {}
    any_success = False

    for view_name in FIXED_VIEW_ORDER:
        try:
            semantic_masks = load_view_semantic_masks(metadata, view_name)
        except SkipSemantic as exc:
            continue

        video_path = camera_videos.get(view_name)
        if not video_path:
            continue
        rgb_frames = read_all_frames(video_path)
        if not rgb_frames or len(semantic_masks) == 0:
            continue

        frame_count = min(len(rgb_frames), len(semantic_masks))
        palette = resolve_palette(metadata, semantic_masks)
        overlay_frames = [
            overlay_semantic_mask(
                resize_frame(rgb_frames[index], tile_width, tile_height),
                resize_label_mask(semantic_masks[index], tile_width, tile_height),
                palette,
            )
            for index in range(frame_count)
        ]
        if not overlay_frames:
            continue
        any_success = True
        per_view_frames[view_name] = overlay_frames
        write_video(
            output_root / f"{sample_id}_{view_name}_semantic_overlay.mp4",
            overlay_frames,
            fps=12.0,
        )

    if not any_success:
        return False, "No semantic data was available for the fixed expected views."

    montage_frames = make_sequence_montage(
        per_view_frames,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    if montage_frames:
        write_video(
            output_root / f"{sample_id}_6v_semantic_overlay.mp4",
            montage_frames,
            fps=12.0,
        )
    return True, "ok"

def load_view_semantic_masks(metadata: dict[str, Any], view_name: str) -> list[Any]:
    semantic_masks = metadata.get("semantic_masks")
    if isinstance(semantic_masks, dict):
        value = semantic_masks.get(view_name)
        if value is not None:
            return load_semantic_masks(value)

    segmentation_frames = metadata.get("segmentation_frames")
    if isinstance(segmentation_frames, dict):
        value = segmentation_frames.get(view_name)
        if value is not None:
            return load_segmentation_frames(value)

    segmentation_video = metadata.get("segmentation_video")
    if isinstance(segmentation_video, dict):
        value = segmentation_video.get(view_name)
        if value is not None:
            return load_segmentation_video(value)

    if semantic_masks is not None and not isinstance(semantic_masks, dict):
        return load_semantic_masks(semantic_masks)
    if segmentation_frames is not None and not isinstance(segmentation_frames, dict):
        return load_segmentation_frames(segmentation_frames)
    if segmentation_video is not None and not isinstance(segmentation_video, dict):
        return load_segmentation_video(segmentation_video)
    raise SkipSemantic(f"No semantic data for view {view_name}.")

def load_semantic_masks(raw_value: Any) -> list[Any]:
    import numpy as np  # type: ignore

    if raw_value is None:
        raise SkipSemantic("semantic_masks entry is missing.")
    if isinstance(raw_value, str):
        path = Path(raw_value)
        if not path.exists():
            raise SkipSemantic(f"semantic_masks path does not exist: {raw_value}")
        if path.suffix.lower() == ".npy":
            data = np.load(path)
        elif path.suffix.lower() == ".json":
            data = np.asarray(json.loads(path.read_text(encoding="utf-8")))
        else:
            raise SkipSemantic(f"Unsupported semantic_masks file format: {path.suffix}")
    else:
        data = np.asarray(raw_value)
    if data.ndim != 3:
        raise SkipSemantic("semantic_masks must have shape [T, H, W].")
    return [data[index] for index in range(data.shape[0])]

def load_segmentation_frames(raw_value: Any) -> list[Any]:
    if raw_value is None:
        raise SkipSemantic("segmentation_frames entry is missing.")
    if isinstance(raw_value, list):
        return [read_image_rgb(item) for item in raw_value if read_image_rgb(item) is not None]
    if isinstance(raw_value, str):
        path = Path(raw_value)
        if not path.exists():
            raise SkipSemantic(f"segmentation_frames path does not exist: {raw_value}")
        if path.suffix.lower() == ".json":
            items = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(items, list):
                raise SkipSemantic("segmentation_frames json must be a list.")
            return [read_image_rgb(item) for item in items if read_image_rgb(item) is not None]
        return [read_image_rgb(raw_value)] if read_image_rgb(raw_value) is not None else []
    raise SkipSemantic("Unsupported segmentation_frames value.")

def load_segmentation_video(raw_value: Any) -> list[Any]:
    if raw_value is None or not isinstance(raw_value, str):
        raise SkipSemantic("segmentation_video must be a file path.")
    path = Path(raw_value)
    if not path.exists():
        raise SkipSemantic(f"segmentation_video path does not exist: {raw_value}")
    return read_all_frames(path)

def read_image_rgb(path_value: Any) -> Any | None:
    if not isinstance(path_value, str):
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    cv2 = _get_cv2()
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def resolve_palette(metadata: dict[str, Any], semantic_masks: list[Any]) -> list[list[int]]:
    palette_value = metadata.get("segmentation_palette")
    if isinstance(palette_value, list) and palette_value:
        return [
            [int(color[0]), int(color[1]), int(color[2])]
            for color in palette_value
            if isinstance(color, (list, tuple)) and len(color) >= 3
        ]
    max_label = 0
    for mask in semantic_masks:
        try:
            max_label = max(max_label, int(mask.max()))
        except Exception:
            continue
    return make_palette(max_label + 1)

def make_palette(num_colors: int) -> list[list[int]]:
    palette: list[list[int]] = []
    for index in range(max(1, num_colors)):
        palette.append(
            [
                (37 * index + 23) % 256,
                (67 * index + 59) % 256,
                (97 * index + 101) % 256,
            ]
        )
    return palette

def overlay_semantic_mask(frame_rgb: Any, label_mask: Any, palette: list[list[int]]) -> Any:
    import numpy as np  # type: ignore

    height, width = frame_rgb.shape[:2]
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    max_label = int(label_mask.max()) if label_mask.size > 0 else 0
    for label in range(max_label + 1):
        color = palette[label % len(palette)]
        color_mask[label_mask == label] = color
    return (0.6 * frame_rgb + 0.4 * color_mask).astype("uint8")

def resize_label_mask(mask: Any, width: int, height: int) -> Any:
    cv2 = _get_cv2()
    return cv2.resize(mask.astype("uint8"), (width, height), interpolation=cv2.INTER_NEAREST)

def make_sequence_montage(
    frames_by_view: dict[str, list[Any]],
    *,
    tile_width: int,
    tile_height: int,
) -> list[Any]:
    if not frames_by_view:
        return []
    frame_count = min(len(items) for items in frames_by_view.values() if items)
    if frame_count <= 0:
        return []
    montage_frames: list[Any] = []
    for frame_index in range(frame_count):
        tiles = {
            view_name: frames_by_view.get(view_name, [None] * frame_count)[frame_index]
            if view_name in frames_by_view and len(frames_by_view[view_name]) > frame_index
            else None
            for view_name in FIXED_VIEW_ORDER
        }
        montage_frames.append(
            make_6v_montage_frame(
                tiles,
                tile_width=tile_width,
                tile_height=tile_height,
            )
        )
    return montage_frames

class SkipSemantic(RuntimeError):
    pass

def _get_cv2() -> Any:
    import cv2  # type: ignore

    return cv2
