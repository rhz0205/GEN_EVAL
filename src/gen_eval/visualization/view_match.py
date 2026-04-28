from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .video_io import draw_text_box, get_cv2, read_selected_frames, resize_frame, write_image, write_video

ADJACENT_CAMERA_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("camera_front", "camera_cross_left", "left", "right"),
    ("camera_front", "camera_cross_right", "right", "left"),
    ("camera_cross_left", "camera_rear_left", "left", "right"),
    ("camera_cross_right", "camera_rear_right", "right", "left"),
    ("camera_rear_left", "camera_rear", "right", "left"),
    ("camera_rear_right", "camera_rear", "left", "right"),
)

def build_view_match_outputs(
    sample_id: str,
    camera_videos: dict[str, str],
    output_dir: str | Path,
    *,
    loftr_repo_path: str | None,
    loftr_weight_path: str | None,
    frame_indices: list[int] | None,
    max_frames: int | None,
    tile_width: int,
    tile_height: int,
) -> tuple[bool, str]:
    if not loftr_repo_path or not loftr_weight_path:
        return False, "LoFTR repo/weight paths are required for view_match visualization."

    runtime = ensure_loftr_runtime(loftr_repo_path, loftr_weight_path)
    if runtime.get("error"):
        return False, str(runtime["error"])

    output_root = Path(output_dir)
    any_success = False

    for cam_a, cam_b, side_a, side_b in ADJACENT_CAMERA_PAIRS:
        path_a = camera_videos.get(cam_a)
        path_b = camera_videos.get(cam_b)
        if not path_a or not path_b:
            continue
        frames_a = select_pair_frames(path_a, frame_indices, max_frames)
        frames_b = select_pair_frames(path_b, frame_indices, max_frames)
        shared_indices = sorted(set(frames_a) & set(frames_b))
        if not shared_indices:
            continue

        pair_name = f"{cam_a}__{cam_b}"
        pair_frames: list[Any] = []
        for frame_index in shared_indices:
            frame_a = resize_frame(frames_a[frame_index], tile_width, tile_height)
            frame_b = resize_frame(frames_b[frame_index], tile_width, tile_height)
            crop_a = crop_edge(frame_a, side_a)
            crop_b = crop_edge(frame_b, side_b)
            match_data = match_loftr(runtime, crop_a, crop_b)
            rendered = draw_match_image(
                crop_a,
                crop_b,
                match_data.get("mkpts0", []),
                match_data.get("mkpts1", []),
                match_data.get("mconf", []),
                title_lines=[
                    f"{cam_a} <-> {cam_b}",
                    f"frame={frame_index}",
                    f"matches={len(match_data.get('mconf', []))}",
                    f"mean_conf={mean_confidence(match_data.get('mconf', [])):.4g}",
                ],
            )
            pair_frames.append(rendered)
            write_image(
                output_root / f"{sample_id}_{pair_name}_frame_{frame_index:06d}.jpg",
                rendered,
            )
            any_success = True

        if pair_frames:
            write_video(output_root / f"{sample_id}_{pair_name}.mp4", pair_frames, fps=6.0)

    if not any_success:
        return False, "No adjacent camera pairs produced match visualizations."
    return True, "ok"

def ensure_loftr_runtime(repo_path: str, weight_path: str) -> dict[str, Any]:
    runtime: dict[str, Any] = {"error": None}
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        runtime["error"] = f"torch is required for LoFTR visualization: {exc}"
        return runtime

    repo = str(Path(repo_path).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        from src.loftr import LoFTR, default_cfg  # type: ignore
    except Exception as exc:  # noqa: BLE001
        runtime["error"] = f"Failed to import LoFTR from local repo: {exc}"
        return runtime

    ckpt_path = Path(weight_path).expanduser().resolve()
    if not ckpt_path.exists():
        runtime["error"] = f"LoFTR checkpoint not found: {ckpt_path}"
        return runtime

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = LoFTR(config=default_cfg)
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        model.eval().to(device)
    except Exception as exc:  # noqa: BLE001
        runtime["error"] = f"Failed to initialize LoFTR: {exc}"
        return runtime

    runtime["torch"] = torch
    runtime["model"] = model
    runtime["device"] = device
    return runtime

def select_pair_frames(
    video_path: str,
    frame_indices: list[int] | None,
    max_frames: int | None,
) -> dict[int, Any]:
    if frame_indices:
        return read_selected_frames(video_path, frame_indices)
    if max_frames is None:
        max_frames = 1
    return {index: frame for index, frame in enumerate(read_selected_frames(video_path, list(range(max_frames))).values())}

def crop_edge(frame_rgb: Any, side: str) -> Any:
    width = frame_rgb.shape[1]
    crop_width = max(1, int(width / 3))
    if side == "left":
        return frame_rgb[:, :crop_width]
    return frame_rgb[:, width - crop_width :]

def match_loftr(runtime: dict[str, Any], crop_a: Any, crop_b: Any) -> dict[str, Any]:
    torch = runtime["torch"]
    model = runtime["model"]
    device = runtime["device"]
    cv2 = get_cv2()
    gray_a = cv2.cvtColor(crop_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(crop_b, cv2.COLOR_RGB2GRAY)
    tensor_a = torch.from_numpy(gray_a.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    tensor_b = torch.from_numpy(gray_b.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    data = {"image0": tensor_a, "image1": tensor_b}
    with torch.no_grad():
        model(data)
    mkpts0 = data.get("mkpts0_f")
    mkpts1 = data.get("mkpts1_f")
    mconf = data.get("mconf")
    if mkpts0 is None or mkpts1 is None or mconf is None:
        return {"mkpts0": [], "mkpts1": [], "mconf": []}
    return {
        "mkpts0": mkpts0.detach().cpu().numpy(),
        "mkpts1": mkpts1.detach().cpu().numpy(),
        "mconf": mconf.detach().cpu().numpy(),
    }

def draw_match_image(
    crop_a: Any,
    crop_b: Any,
    mkpts0: Any,
    mkpts1: Any,
    mconf: Any,
    *,
    title_lines: list[str],
) -> Any:
    import numpy as np  # type: ignore

    cv2 = get_cv2()
    img_a = cv2.cvtColor(crop_a, cv2.COLOR_RGB2BGR)
    img_b = cv2.cvtColor(crop_b, cv2.COLOR_RGB2BGR)
    height = max(img_a.shape[0], img_b.shape[0])
    canvas = np.zeros((height, img_a.shape[1] + img_b.shape[1], 3), dtype=np.uint8)
    canvas[: img_a.shape[0], : img_a.shape[1]] = img_a
    canvas[: img_b.shape[0], img_a.shape[1] :] = img_b
    width_offset = img_a.shape[1]
    for point_a, point_b, conf in zip(mkpts0, mkpts1, mconf):
        color = confidence_color(float(conf))
        x0, y0 = int(round(float(point_a[0]))), int(round(float(point_a[1])))
        x1, y1 = int(round(float(point_b[0]) + width_offset)), int(round(float(point_b[1])))
        cv2.circle(canvas, (x0, y0), 2, color, -1)
        cv2.circle(canvas, (x1, y1), 2, color, -1)
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1)
    rendered = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return draw_text_box(rendered, title_lines)

def confidence_color(confidence: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, float(confidence)))
    return (64, int(255 * value), int(255 * (1.0 - value)))

def mean_confidence(values: Any) -> float:
    items = [float(value) for value in values] if values is not None else []
    return sum(items) / len(items) if items else 0.0
