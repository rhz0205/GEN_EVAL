#!/usr/bin/env python3
"""Generate lightweight 6-view visualizations for GEN_EVAL samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gen_eval.visualization.depth import build_depth_outputs
from gen_eval.visualization.instance import build_instance_outputs
from gen_eval.visualization.integrity import build_integrity_overview_image
from gen_eval.visualization.layout import FIXED_VIEW_ORDER, make_6v_montage_frame
from gen_eval.visualization.semantic import build_semantic_outputs
from gen_eval.visualization.video_io import inspect_video, read_all_frames, read_first_frame, write_image, write_video
from gen_eval.visualization.view_match import build_view_match_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize GEN_EVAL 6-view results.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSON path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--sample-id", default=None, help="Optional single sample_id to visualize.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process.")
    parser.add_argument(
        "--types",
        default="all",
        help="Comma-separated visualization types: rgb,integrity,view_match,depth,semantic,instance,all",
    )
    parser.add_argument("--tile-width", type=int, default=480, help="Common tile width for montage and overlay outputs.")
    parser.add_argument("--loftr-repo-path", default=None, help="Local LoFTR repo path for view_match.")
    parser.add_argument("--loftr-weight-path", default=None, help="Local LoFTR checkpoint path for view_match.")
    parser.add_argument("--view-match-frame-indices", default=None, help="Comma-separated frame indices for view_match.")
    parser.add_argument("--view-match-max-frames", type=int, default=3, help="Maximum frames to visualize per adjacent pair when indices are not provided.")
    parser.add_argument("--depth-repo-path", default=None, help="Local Video-Depth-Anything repo path.")
    parser.add_argument("--depth-checkpoint-path", default=None, help="Local Video-Depth-Anything checkpoint path.")
    parser.add_argument("--depth-model-dir", default=None, help="Local depth model directory used to resolve checkpoints.")
    parser.add_argument("--depth-encoder", default="vits", help="Depth encoder name.")
    parser.add_argument(
        "--depth-video-module",
        default="gen_eval.third_party.video_depth_anything.video_depth",
        help="Import path for the local Video-Depth-Anything module.",
    )
    parser.add_argument(
        "--depth-video-class",
        default="VideoDepthAnything",
        help="Class name inside the local Video-Depth-Anything module.",
    )
    parser.add_argument("--depth-max-res", type=int, default=400, help="Maximum frame resolution for depth inference.")
    parser.add_argument("--depth-input-size", type=int, default=400, help="Depth input size.")
    parser.add_argument("--depth-target-fps", type=int, default=12, help="Depth target fps.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    manifest_payload = load_manifest_payload(args.manifest)
    samples = extract_samples(manifest_payload)
    selected_samples = select_samples(
        samples,
        sample_id=args.sample_id,
        max_samples=args.max_samples if args.max_samples is not None else (None if args.sample_id else 1),
    )
    if not selected_samples:
        print("No samples selected.")
        return 1

    output_root = Path(args.output_dir)
    types = normalize_types(args.types)
    frame_indices = parse_frame_indices(args.view_match_frame_indices)
    tile_width = max(64, int(args.tile_width))
    tile_height = int(round(tile_width * 9 / 16))

    for sample in selected_samples:
        sample_id = str(sample.get("sample_id") or "unknown")
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        camera_videos = normalize_camera_videos(metadata.get("camera_videos"))
        sample_root = output_root / sample_id
        print(f"[sample] {sample_id}")

        for vis_type in types:
            try:
                ok, message = run_visualization_type(
                    vis_type,
                    sample_id=sample_id,
                    metadata=metadata,
                    camera_videos=camera_videos,
                    sample_root=sample_root,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    loftr_repo_path=args.loftr_repo_path,
                    loftr_weight_path=args.loftr_weight_path,
                    view_match_frame_indices=frame_indices,
                    view_match_max_frames=args.view_match_max_frames,
                    depth_repo_path=args.depth_repo_path,
                    depth_checkpoint_path=args.depth_checkpoint_path,
                    depth_model_dir=args.depth_model_dir,
                    depth_encoder=args.depth_encoder,
                    depth_video_module=args.depth_video_module,
                    depth_video_class=args.depth_video_class,
                    depth_max_res=args.depth_max_res,
                    depth_input_size=args.depth_input_size,
                    depth_target_fps=args.depth_target_fps,
                )
            except Exception as exc:  # noqa: BLE001
                ok = False
                message = f"{type(exc).__name__}: {exc}"

            status = "success" if ok else "skipped"
            print(f"  [{vis_type}] {status}: {message}")
    return 0


def normalize_types(raw_value: str) -> list[str]:
    requested = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not requested or "all" in requested:
        return ["rgb", "integrity", "view_match", "depth", "semantic", "instance"]
    return requested


def parse_frame_indices(raw_value: str | None) -> list[int] | None:
    if not raw_value:
        return None
    indices: list[int] = []
    for item in raw_value.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            indices.append(int(text))
        except ValueError:
            continue
    return sorted(set(index for index in indices if index >= 0))


def load_manifest_payload(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_samples(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        samples = payload.get("samples")
        if isinstance(samples, list):
            return [item for item in samples if isinstance(item, dict)]
    raise ValueError("Manifest JSON must be a list or an object with a 'samples' list.")


def select_samples(
    samples: list[dict[str, Any]],
    *,
    sample_id: str | None,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    if sample_id:
        return [sample for sample in samples if str(sample.get("sample_id")) == sample_id][:1]
    if max_samples is None:
        return samples
    return samples[: max(0, int(max_samples))]


def normalize_camera_videos(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(view): str(path)
        for view, path in value.items()
        if path is not None and str(view) in FIXED_VIEW_ORDER
    }


def run_visualization_type(
    vis_type: str,
    *,
    sample_id: str,
    metadata: dict[str, Any],
    camera_videos: dict[str, str],
    sample_root: Path,
    tile_width: int,
    tile_height: int,
    loftr_repo_path: str | None,
    loftr_weight_path: str | None,
    view_match_frame_indices: list[int] | None,
    view_match_max_frames: int | None,
    depth_repo_path: str | None,
    depth_checkpoint_path: str | None,
    depth_model_dir: str | None,
    depth_encoder: str,
    depth_video_module: str,
    depth_video_class: str,
    depth_max_res: int,
    depth_input_size: int,
    depth_target_fps: int,
) -> tuple[bool, str]:
    if vis_type == "rgb":
        return build_rgb_montage(
            sample_id,
            camera_videos,
            sample_root / "rgb",
            tile_width=tile_width,
            tile_height=tile_height,
        )
    if vis_type == "integrity":
        return build_integrity_image(
            sample_id,
            camera_videos,
            sample_root / "integrity",
            tile_width=tile_width,
            tile_height=tile_height,
        )
    if vis_type == "view_match":
        return build_view_match_outputs(
            sample_id,
            camera_videos,
            sample_root / "view_match",
            loftr_repo_path=loftr_repo_path,
            loftr_weight_path=loftr_weight_path,
            frame_indices=view_match_frame_indices,
            max_frames=view_match_max_frames,
            tile_width=tile_width,
            tile_height=tile_height,
        )
    if vis_type == "depth":
        return build_depth_outputs(
            sample_id,
            camera_videos,
            sample_root / "depth",
            repo_path=depth_repo_path,
            checkpoint_path=depth_checkpoint_path,
            model_dir=depth_model_dir,
            encoder=depth_encoder,
            video_depth_module=depth_video_module,
            video_depth_class=depth_video_class,
            max_res=depth_max_res,
            input_size=depth_input_size,
            target_fps=depth_target_fps,
            tile_width=tile_width,
            tile_height=tile_height,
        )
    if vis_type == "semantic":
        return build_semantic_outputs(
            sample_id,
            metadata,
            sample_root / "semantic",
            tile_width=tile_width,
            tile_height=tile_height,
        )
    if vis_type == "instance":
        return build_instance_outputs(
            sample_id,
            metadata,
            sample_root / "instance",
            tile_width=tile_width,
            tile_height=tile_height,
        )
    return False, f"Unknown visualization type: {vis_type}"


def build_rgb_montage(
    sample_id: str,
    camera_videos: dict[str, str],
    output_dir: Path,
    *,
    tile_width: int,
    tile_height: int,
) -> tuple[bool, str]:
    frames_by_view: dict[str, list[Any]] = {}
    available_counts: list[int] = []
    for view_name in FIXED_VIEW_ORDER:
        video_path = camera_videos.get(view_name)
        if not video_path:
            continue
        info = inspect_video(video_path)
        if not info.get("readable"):
            continue
        frames = read_all_frames(video_path)
        if not frames:
            continue
        frames_by_view[view_name] = frames
        available_counts.append(len(frames))
    if not available_counts:
        return False, "No readable camera views were available."
    frame_count = min(available_counts)
    montage_frames: list[Any] = []
    for frame_index in range(frame_count):
        montage_frames.append(
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
    write_video(output_dir / f"{sample_id}_6v_rgb.mp4", montage_frames, fps=12.0)
    return True, f"wrote {len(montage_frames)} frames"


def build_integrity_image(
    sample_id: str,
    camera_videos: dict[str, str],
    output_dir: Path,
    *,
    tile_width: int,
    tile_height: int,
) -> tuple[bool, str]:
    if not camera_videos:
        return False, "No camera_videos metadata was available."
    overview = build_integrity_overview_image(
        camera_videos,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    write_image(output_dir / f"{sample_id}_video_integrity_overview.jpg", overview)
    return True, "wrote overview image"


if __name__ == "__main__":
    raise SystemExit(main())
