from __future__ import annotations

from pathlib import Path
from typing import Any

from .layout import FIXED_VIEW_ORDER, make_6v_montage_frame
from .video_io import read_all_frames, resize_frame, write_video


def build_instance_outputs(
    sample_id: str,
    metadata: dict[str, Any],
    output_dir: str | Path,
    *,
    tile_width: int,
    tile_height: int,
) -> tuple[bool, str]:
    camera_videos = metadata.get("camera_videos")
    instance_tracks = metadata.get("instance_tracks")
    if not isinstance(camera_videos, dict) or not camera_videos:
        return False, "metadata['camera_videos'] is required."
    if not isinstance(instance_tracks, dict) or not instance_tracks:
        return False, "metadata['instance_tracks'] is missing."

    output_root = Path(output_dir)
    per_view_frames: dict[str, list[Any]] = {}
    any_success = False

    for view_name in FIXED_VIEW_ORDER:
        video_path = camera_videos.get(view_name)
        raw_tracks = instance_tracks.get(view_name)
        if not video_path or not isinstance(raw_tracks, list):
            continue
        frames = read_all_frames(video_path)
        if not frames:
            continue
        overlays = render_track_overlay_sequence(
            frames,
            raw_tracks,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        if not overlays:
            continue
        any_success = True
        per_view_frames[view_name] = overlays
        write_video(
            output_root / f"{sample_id}_{view_name}_tracks.mp4",
            overlays,
            fps=12.0,
        )

    if not any_success:
        return False, "No per-view instance tracks were available for visualization."

    montage_frames = make_sequence_montage(
        per_view_frames,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    if montage_frames:
        write_video(
            output_root / f"{sample_id}_6v_tracks.mp4",
            montage_frames,
            fps=12.0,
        )
    return True, "ok"


def render_track_overlay_sequence(
    frames_rgb: list[Any],
    raw_tracks: list[dict[str, Any]],
    *,
    tile_width: int,
    tile_height: int,
) -> list[Any]:
    overlays = [resize_frame(frame, tile_width, tile_height) for frame in frames_rgb]
    normalized_tracks = normalize_instance_tracks(raw_tracks)
    cv2 = _get_cv2()
    trail_length = 8

    for track in normalized_tracks:
        color = color_for_key(str(track.get("object_id") or "unknown"))
        box_items = track.get("boxes", [])
        trail_points: list[tuple[int, int]] = []
        for box_item in box_items:
            if not isinstance(box_item, dict):
                continue
            frame_index = box_item.get("frame_index")
            bbox = extract_bbox(box_item)
            if not isinstance(frame_index, int) or bbox is None:
                continue
            if frame_index < 0 or frame_index >= len(overlays):
                continue
            frame = overlays[frame_index]
            draw_track_box(
                frame,
                bbox,
                label=build_track_label(track, box_item),
                color=color,
            )
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            trail_points.append(center)
            recent_points = trail_points[-trail_length:]
            for idx in range(1, len(recent_points)):
                cv2.line(
                    frame,
                    recent_points[idx - 1],
                    recent_points[idx],
                    tuple(int(v) for v in color[::-1]),
                    2,
                    cv2.LINE_AA,
                )
    return overlays


def normalize_instance_tracks(raw_tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_track in raw_tracks:
        if not isinstance(raw_track, dict):
            continue
        track = dict(raw_track)
        if "boxes_2d" in track and "boxes" not in track:
            track["boxes"] = track["boxes_2d"]
        normalized.append(track)
    return normalized


def extract_bbox(box_item: dict[str, Any]) -> list[float] | None:
    for key in ("bbox", "box", "boxes_2d"):
        value = box_item.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    return None


def build_track_label(track: dict[str, Any], box_item: dict[str, Any]) -> str:
    object_id = str(track.get("object_id") or "track")
    category = str(track.get("category") or "object")
    confidence = box_item.get("confidence")
    if isinstance(confidence, (int, float)):
        return f"{object_id} | {category} | {float(confidence):.2f}"
    return f"{object_id} | {category}"


def draw_track_box(
    frame_rgb: Any,
    bbox: list[float],
    *,
    label: str,
    color: tuple[int, int, int],
) -> None:
    cv2 = _get_cv2()
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), tuple(int(v) for v in color[::-1]), 2)
    cv2.putText(
        frame_bgr,
        label[:80],
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        tuple(int(v) for v in color[::-1]),
        1,
        cv2.LINE_AA,
    )
    frame_rgb[:, :] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def color_for_key(key: str) -> tuple[int, int, int]:
    seed = abs(hash(key))
    return (
        64 + (seed * 37) % 192,
        64 + (seed * 67) % 192,
        64 + (seed * 97) % 192,
    )


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
    result: list[Any] = []
    for frame_index in range(frame_count):
        result.append(
            make_6v_montage_frame(
                {
                    view_name: frames_by_view.get(view_name, [None] * frame_count)[frame_index]
                    if view_name in frames_by_view and len(frames_by_view[view_name]) > frame_index
                    else None
                    for view_name in FIXED_VIEW_ORDER
                },
                tile_width=tile_width,
                tile_height=tile_height,
            )
        )
    return result


def _get_cv2() -> Any:
    import cv2  # type: ignore

    return cv2

