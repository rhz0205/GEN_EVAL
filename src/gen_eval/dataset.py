from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from gen_eval.schemas import GenerationSample


def load_manifest(path: str | Path) -> list[GenerationSample]:
    payload = load_manifest_payload(path)
    samples_data = _extract_samples(payload)
    return [GenerationSample.from_dict(item) for item in samples_data]


def load_manifest_payload(path: str | Path) -> Any:
    manifest_path = Path(path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    payload = load_manifest_payload(path)
    return [item for item in _extract_samples(payload) if isinstance(item, dict)]


def _extract_samples(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "samples" in payload and isinstance(payload["samples"], list):
            return payload["samples"]
    raise ValueError("Manifest JSON must be a list or an object with a 'samples' list.")


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


def format_manifest_summary(manifest_path: str | Path) -> list[str]:
    path = Path(manifest_path)
    lines = [f"manifest_path: {path}", f"exists: {path.exists()}"]
    if not path.exists():
        return lines

    samples = load_manifest_records(path)
    lines.append(f"num_samples: {len(samples)}")

    first_sample = samples[0] if samples else None
    lines.append(f"first_sample_id: {first_sample.get('sample_id') if first_sample else None}")

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
                camera_name_text = str(camera_name)
                camera_name_frequency[camera_name_text] += 1
                if camera_name_text == "camera_front_tele":
                    camera_front_tele_present = True

        views = metadata.get("views")
        if isinstance(views, list):
            views_distribution[len(views)] += 1

    lines.append("generated_video_path_type_counts:")
    for key in ("file", "dir", "missing", "empty/null"):
        lines.append(f"  {key}: {generated_video_counts.get(key, 0)}")

    lines.append("reference_video_counts:")
    lines.append(f"  present: {reference_video_counts.get('present', 0)}")
    lines.append(f"  missing/null: {reference_video_counts.get('missing/null', 0)}")
    lines.append(f"camera_videos_presence_count: {camera_videos_presence}")
    lines.append(f"views_count_distribution: {dict(sorted(views_distribution.items()))}")
    lines.append(f"camera_name_frequency: {dict(sorted(camera_name_frequency.items()))}")
    lines.append(f"camera_front_tele_appears: {camera_front_tele_present}")
    lines.append("first_sample_preview:")
    lines.append(_truncate_json(first_sample))
    return lines
