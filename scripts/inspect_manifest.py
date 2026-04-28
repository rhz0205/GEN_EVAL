#!/usr/bin/env python3
"""Inspect a GEN_EVAL manifest file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        return [item for item in data["samples"] if isinstance(item, dict)]
    return []


def _path_kind(value: Any) -> str:
    if value is None or value == "":
        return "empty/null"
    path = Path(str(value))
    if path.is_file():
        return "file"
    if path.is_dir():
        return "dir"
    return "missing"


def _truncate_json(value: Any, limit: int = 800) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) <= limit:
        return text
    return text[: limit - 15] + "\n...<truncated>"


def inspect_manifest(manifest_path: Path) -> int:
    print(f"manifest_path: {manifest_path}")
    print(f"exists: {manifest_path.exists()}")
    if not manifest_path.exists():
        return 1

    samples = _load_manifest(manifest_path)
    print(f"num_samples: {len(samples)}")

    first_sample = samples[0] if samples else None
    print(f"first_sample_id: {first_sample.get('sample_id') if first_sample else None}")

    generated_video_counts: Counter[str] = Counter()
    reference_video_counts: Counter[str] = Counter()
    camera_videos_presence = 0
    views_distribution: Counter[int] = Counter()
    camera_name_frequency: Counter[str] = Counter()
    camera_front_tele_present = False

    for sample in samples:
        generated_video_counts[_path_kind(sample.get("generated_video"))] += 1

        reference_video = sample.get("reference_video")
        if reference_video is None or reference_video == "":
            reference_video_counts["missing/null"] += 1
        else:
            reference_video_counts["present"] += 1

        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            continue

        camera_videos = metadata.get("camera_videos")
        if isinstance(camera_videos, dict):
            camera_videos_presence += 1
            for camera_name in camera_videos.keys():
                camera_name = str(camera_name)
                camera_name_frequency[camera_name] += 1
                if camera_name == "camera_front_tele":
                    camera_front_tele_present = True

        views = metadata.get("views")
        if isinstance(views, list):
            views_distribution[len(views)] += 1

    print("generated_video_path_type_counts:")
    for key in ("file", "dir", "missing", "empty/null"):
        print(f"  {key}: {generated_video_counts.get(key, 0)}")

    print("reference_video_counts:")
    print(f"  present: {reference_video_counts.get('present', 0)}")
    print(f"  missing/null: {reference_video_counts.get('missing/null', 0)}")

    print(f"camera_videos_presence_count: {camera_videos_presence}")
    print(f"views_count_distribution: {dict(sorted(views_distribution.items()))}")
    print(f"camera_name_frequency: {dict(sorted(camera_name_frequency.items()))}")
    print(f"camera_front_tele_appears: {camera_front_tele_present}")

    print("first_sample_preview:")
    print(_truncate_json(first_sample))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a GEN_EVAL manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file.")
    args = parser.parse_args()
    return inspect_manifest(Path(args.manifest))


if __name__ == "__main__":
    raise SystemExit(main())
