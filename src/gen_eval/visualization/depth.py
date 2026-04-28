from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from .layout import FIXED_VIEW_ORDER, make_6v_montage_frame
from .video_io import read_all_frames, resize_frame, write_video

MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


def build_depth_outputs(
    sample_id: str,
    camera_videos: dict[str, str],
    output_dir: str | Path,
    *,
    repo_path: str | None,
    checkpoint_path: str | None,
    model_dir: str | None,
    encoder: str,
    max_res: int,
    input_size: int,
    target_fps: int,
    tile_width: int,
    tile_height: int,
    video_depth_module: str = "gen_eval.third_party.video_depth_anything.video_depth",
    video_depth_class: str = "VideoDepthAnything",
) -> tuple[bool, str]:
    runtime = ensure_depth_runtime(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        model_dir=model_dir,
        encoder=encoder,
        video_depth_module=video_depth_module,
        video_depth_class=video_depth_class,
    )
    if runtime.get("error"):
        return False, str(runtime["error"])

    output_root = Path(output_dir)
    per_view_depth_frames: dict[str, list[Any]] = {}
    any_success = False

    for view_name in FIXED_VIEW_ORDER:
        video_path = camera_videos.get(view_name)
        if not video_path:
            continue
        rgb_frames = [
            downscale_max_res(frame, max_res=max_res)
            for frame in read_all_frames(video_path)
        ]
        if len(rgb_frames) < 2:
            continue
        depths = infer_depth(runtime, rgb_frames, input_size=input_size, target_fps=target_fps)
        if depths is None or len(depths) < 2:
            continue
        depth_rgb = render_depth_rgb(depths)
        rgb_depth_frames = [
            side_by_side(
                resize_frame(rgb_frames[index], tile_width, tile_height),
                resize_frame(depth_rgb[index], tile_width, tile_height),
            )
            for index in range(min(len(rgb_frames), len(depth_rgb)))
        ]
        depth_only_frames = [
            resize_frame(frame, tile_width, tile_height) for frame in depth_rgb
        ]
        per_view_depth_frames[view_name] = depth_only_frames
        write_video(output_root / f"{sample_id}_{view_name}_depth.mp4", depth_only_frames, fps=float(target_fps))
        write_video(output_root / f"{sample_id}_{view_name}_rgb_depth.mp4", rgb_depth_frames, fps=float(target_fps))
        any_success = True

    if not any_success:
        return False, "No view produced depth visualizations."

    montage_frames = make_sequence_montage(
        per_view_depth_frames,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    if montage_frames:
        write_video(output_root / f"{sample_id}_6v_depth.mp4", montage_frames, fps=float(target_fps))
    return True, "ok"


def ensure_depth_runtime(
    *,
    repo_path: str | None,
    checkpoint_path: str | None,
    model_dir: str | None,
    encoder: str,
    video_depth_module: str = "gen_eval.third_party.video_depth_anything.video_depth",
    video_depth_class: str = "VideoDepthAnything",
) -> dict[str, Any]:
    runtime: dict[str, Any] = {"error": None}
    if not repo_path:
        runtime["error"] = "Depth repo path is required for depth visualization."
        return runtime
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        runtime["error"] = f"torch is required for depth visualization: {exc}"
        return runtime

    if encoder not in MODEL_CONFIGS:
        runtime["error"] = f"Unsupported depth encoder: {encoder}"
        return runtime

    repo = str(Path(repo_path).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        module = importlib.import_module(video_depth_module)
        depth_cls = getattr(module, video_depth_class)
    except Exception as exc:  # noqa: BLE001
        runtime["error"] = (
            "Failed to import VideoDepthAnything from local runtime "
            f"{video_depth_module}.{video_depth_class}: {exc}"
        )
        return runtime

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
    elif model_dir:
        ckpt_path = Path(model_dir).expanduser().resolve() / f"metric_video_depth_anything_{encoder}.pth"
    else:
        runtime["error"] = "Depth checkpoint path or depth model dir is required."
        return runtime

    if not ckpt_path.exists():
        runtime["error"] = f"Depth checkpoint not found: {ckpt_path}"
        return runtime

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        engine = depth_cls(**MODEL_CONFIGS[encoder])
        state = torch.load(str(ckpt_path), map_location="cpu")
        engine.load_state_dict(state, strict=True)
        engine = engine.to(device).eval()
    except Exception as exc:  # noqa: BLE001
        runtime["error"] = f"Failed to initialize depth engine: {exc}"
        return runtime

    runtime["torch"] = torch
    runtime["engine"] = engine
    runtime["device"] = device
    return runtime


def infer_depth(runtime: dict[str, Any], frames_rgb: list[Any], *, input_size: int, target_fps: int) -> Any | None:
    try:
        depths, _ = runtime["engine"].infer_video_depth(
            frames_rgb,
            target_fps=target_fps,
            input_size=input_size,
            device=runtime["device"],
            fp32=False,
        )
        return depths
    except Exception:
        return None


def render_depth_rgb(depths: Any) -> list[Any]:
    import matplotlib.cm as cm  # type: ignore
    import numpy as np  # type: ignore

    frames = np.asarray(depths)
    if frames.ndim == 4 and frames.shape[-1] == 1:
        frames = frames[..., 0]
    mn = float(frames.min())
    mx = float(frames.max())
    if mx > mn:
        depth_norm = (frames - mn) / (mx - mn)
    else:
        depth_norm = np.zeros_like(frames, dtype=np.float32)
    colored = (cm.get_cmap("turbo")(depth_norm)[..., :3] * 255).astype(np.uint8)
    return [colored[index] for index in range(colored.shape[0])]


def downscale_max_res(frame_rgb: Any, *, max_res: int) -> Any:
    height, width = frame_rgb.shape[:2]
    if max(height, width) <= max_res:
        return frame_rgb
    scale = float(max_res) / float(max(height, width))
    return resize_frame(frame_rgb, max(1, int(width * scale)), max(1, int(height * scale)))


def side_by_side(left_rgb: Any, right_rgb: Any) -> Any:
    import numpy as np  # type: ignore

    return np.concatenate([left_rgb, right_rgb], axis=1)


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
