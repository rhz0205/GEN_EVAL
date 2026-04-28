#!/usr/bin/env python3
"""Convert a category-indexed pickle file into a GEN_EVAL manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gen_eval.manifest_builder import build_manifest_from_pkl, summarize_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert category-indexed pkl to GEN_EVAL manifest."
    )
    parser.add_argument("--pkl", required=True, help="Input pkl path.")
    parser.add_argument("--output", default=None, help="Output manifest JSON path.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset group name, e.g. sample_data, geely_data, cosmos_data, real_data.",
    )
    parser.add_argument(
        "--dataset-split",
        default=None,
        help=(
            "Optional dataset split name, e.g. sample, geely, cosmos, real. "
            "If omitted, a simple split is inferred from dataset_name when possible."
        ),
    )
    parser.add_argument("--video-field", default="video", help="Field name for video path.")
    parser.add_argument("--hdmap-field", default="hdmap", help="Field name for HD map path.")
    parser.add_argument(
        "--keys",
        default=None,
        help="Comma-separated top-level keys to include. Default: all keys.",
    )
    parser.add_argument(
        "--limit-per-key",
        type=int,
        default=None,
        help="Keep only the first N raw items per top-level key before random sampling.",
    )
    parser.add_argument(
        "--limit-total",
        type=int,
        default=None,
        help="Keep only the first N sampled candidate items after random sampling.",
    )
    parser.add_argument(
        "--sample-per-key",
        type=int,
        default=None,
        help="Randomly sample this many valid items per top-level key before global sampling.",
    )
    parser.add_argument(
        "--sample-total",
        type=int,
        default=None,
        help="Randomly sample this many valid items from all selected keys.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not merge duplicate video paths across different top-level keys.",
    )
    parser.add_argument(
        "--check-exists",
        action="store_true",
        help="Skip records whose generated_video path does not exist.",
    )
    parser.add_argument(
        "--drop-extra-fields",
        action="store_true",
        help="Do not copy extra item fields into metadata['raw_item'].",
    )
    parser.add_argument(
        "--detect-camera-videos",
        action="store_true",
        help="If item video path is a directory, detect camera videos inside it.",
    )
    parser.add_argument(
        "--primary-camera",
        default="camera_front",
        help="Primary camera name to use as generated_video when detecting camera videos.",
    )
    parser.add_argument(
        "--camera-ext",
        default=".mp4",
        help="Camera video file extension for directory scanning. Default: .mp4",
    )
    parser.add_argument(
        "--keep-front-tele",
        action="store_true",
        help=(
            "Keep camera_front_tele.mp4 in metadata['camera_videos'] and metadata['views']. "
            "By default, camera_front_tele is excluded."
        ),
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="If set, write multiple manifest shards with this many samples each.",
    )
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Optional stats JSON output path.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = build_manifest_from_pkl(
            pkl_path=args.pkl,
            output=args.output,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            video_field=args.video_field,
            hdmap_field=args.hdmap_field,
            keys=args.keys,
            limit_per_key=args.limit_per_key,
            limit_total=args.limit_total,
            sample_per_key=args.sample_per_key,
            sample_total=args.sample_total,
            seed=args.seed,
            dedupe=not args.no_dedupe,
            check_exists=args.check_exists,
            keep_extra_fields=not args.drop_extra_fields,
            detect_camera_videos_dirs=args.detect_camera_videos,
            primary_camera=args.primary_camera,
            camera_ext=args.camera_ext,
            keep_front_tele=args.keep_front_tele,
            shard_size=args.shard_size,
            stats_output=args.stats_output,
        )
    except ValueError as exc:
        parser.error(str(exc))

    stats = result["stats"]
    output_path = result["output_path"]
    stats_path = result["stats_path"]

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote stats to: {stats_path}")
    if result["sharded"]:
        print(f"Wrote {len(result['shard_paths'])} manifest shards.")
    else:
        print(f"Wrote manifest to: {output_path}")

    summarize_records(
        records=result["records"],
        output_manifest_path=None if result["sharded"] else output_path,
        stats_path=stats_path,
        dataset_name=result["dataset_name"],
        dataset_split=result["effective_dataset_split"],
        keep_front_tele=result["keep_front_tele"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
