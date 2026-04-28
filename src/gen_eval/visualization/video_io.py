from __future__ import annotations

from pathlib import Path
from typing import Any

def get_cv2() -> Any:
    import cv2  # type: ignore

    return cv2

def inspect_video(path: str | Path) -> dict[str, Any]:
    video_path = Path(path)
    info: dict[str, Any] = {
        "path": str(video_path),
        "exists": video_path.exists(),
        "is_file": video_path.is_file(),
        "readable": False,
        "frame_count": 0,
        "fps": None,
        "width": None,
        "height": None,
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
        info.update(
            {
                "readable": frame_count > 0 and width > 0 and height > 0,
                "frame_count": frame_count,
                "fps": fps if fps > 0 else None,
                "width": width if width > 0 else None,
                "height": height if height > 0 else None,
                "reason": None if frame_count > 0 and width > 0 and height > 0 else "invalid video metadata",
            }
        )
        return info
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return info

def read_all_frames(video_path: str | Path, max_frames: int | None = None) -> list[Any]:
    cv2 = get_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []

    frames: list[Any] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames

def read_selected_frames(video_path: str | Path, frame_indices: list[int]) -> dict[int, Any]:
    cv2 = get_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return {}

    result: dict[int, Any] = {}
    for frame_index in sorted(set(index for index in frame_indices if index >= 0)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = cap.read()
        if ok and frame_bgr is not None:
            result[frame_index] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    return result

def read_first_frame(video_path: str | Path) -> Any | None:
    frames = read_all_frames(video_path, max_frames=1)
    return frames[0] if frames else None

def resize_frame(frame_rgb: Any, width: int, height: int) -> Any:
    cv2 = get_cv2()
    return cv2.resize(frame_rgb, (int(width), int(height)))

def rgb_to_bgr(frame_rgb: Any) -> Any:
    cv2 = get_cv2()
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(frame_bgr: Any) -> Any:
    cv2 = get_cv2()
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def write_video(
    output_path: str | Path,
    frames_rgb: list[Any],
    fps: float | None,
) -> None:
    if not frames_rgb:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2 = get_cv2()
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps or 12.0),
        (int(width), int(height)),
    )
    try:
        for frame_rgb in frames_rgb:
            writer.write(rgb_to_bgr(frame_rgb))
    finally:
        writer.release()

def write_image(output_path: str | Path, frame_rgb: Any) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2 = get_cv2()
    cv2.imwrite(str(output), rgb_to_bgr(frame_rgb))

def blank_frame(width: int, height: int, color: tuple[int, int, int] = (24, 24, 24)) -> Any:
    import numpy as np  # type: ignore

    frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    frame[:, :] = color
    return frame

def draw_text_box(
    frame_rgb: Any,
    lines: list[str],
    *,
    origin: tuple[int, int] = (12, 20),
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> Any:
    cv2 = get_cv2()
    frame_bgr = rgb_to_bgr(frame_rgb.copy())
    x, y = origin
    line_height = 22
    max_width = 0
    valid_lines = [str(line) for line in lines if str(line)]
    for line in valid_lines:
        (width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        max_width = max(max_width, width)
    box_height = max(line_height * len(valid_lines) + 10, 10)
    cv2.rectangle(
        frame_bgr,
        (x - 8, y - 18),
        (x + max_width + 8, y - 18 + box_height),
        tuple(int(v) for v in bg_color[::-1]),
        -1,
    )
    for index, line in enumerate(valid_lines):
        baseline_y = y + index * line_height
        cv2.putText(
            frame_bgr,
            line[:120],
            (x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            tuple(int(v) for v in text_color[::-1]),
            1,
            cv2.LINE_AA,
        )
    return bgr_to_rgb(frame_bgr)
