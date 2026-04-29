from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PKL_PATH = Path("/di/group/lishun/outputs/cosmos2/guojiaxiangmu_0424_with_hdmap.pkl")
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data"
DEFAULT_DATASET_NAME = "geely"
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_SEED = 42
DEFAULT_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
EXCLUDED_VIEW_ALIASES = ("tele", "long", "zoom")
VIEW_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("camera_cross_left", ("camera_cross_left", "cross_left", "front_left", "left_front")),
    ("camera_cross_right", ("camera_cross_right", "cross_right", "front_right", "right_front")),
    ("camera_rear_left", ("camera_rear_left", "rear_left", "back_left", "left_rear")),
    ("camera_rear_right", ("camera_rear_right", "rear_right", "back_right", "right_rear")),
    ("camera_front", ("camera_front", "front", "cam_front")),
    ("camera_rear", ("camera_rear", "rear", "back")),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select random multi-view samples from a pickle index.")
    parser.add_argument("--path", default=str(DEFAULT_PKL_PATH), help="Path to the pickle file.")
    parser.add_argument(
        "--sample-size",
        "-k",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of valid multi-view samples to export.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for sampled json outputs.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name used in output filenames.",
    )
    return parser


def load_pickle(path: str | Path) -> Any:
    pkl_path = Path(path)
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Pickle file does not exist: {pkl_path}")
    with pkl_path.open("rb") as file:
        return pickle.load(file)


def require_top_level_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Pickle payload must be a top-level dict of tag lists.")
    return payload


def build_tag_summary(payload: dict[str, Any], dataset_name: str, source_path: Path) -> dict[str, Any]:
    tags: list[dict[str, Any]] = []
    total_items = 0
    for tag, value in payload.items():
        if isinstance(value, (list, tuple)):
            sample_count = len(value)
            total_items += sample_count
        else:
            sample_count = None
        tags.append({"tag": str(tag), "sample_count": sample_count})
    tags.sort(
        key=lambda item: (
            item["sample_count"] is None,
            -(item["sample_count"] or 0),
            item["tag"],
        )
    )
    return {
        "dataset_name": dataset_name,
        "source_pkl": str(source_path),
        "tag_count": len(payload),
        "total_tagged_items": total_items,
        "tags": tags,
    }


def build_video_key(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    video = item.get("video")
    if video is None:
        return None
    text = str(video).strip()
    return text or None


def collect_unique_samples(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for tag, value in payload.items():
        if not isinstance(value, (list, tuple)):
            continue
        for item in value:
            video_key = build_video_key(item)
            if video_key is None:
                continue
            record = deduped.get(video_key)
            if record is None:
                deduped[video_key] = {
                    "video": video_key,
                    "tags": [str(tag)],
                }
                continue
            record["tags"].append(str(tag))
    for record in deduped.values():
        record["tags"] = dedupe_preserve_order(record["tags"])
    return deduped


def resolve_candidate_files(video_path: Path) -> list[Path]:
    if video_path.is_file():
        return [video_path]
    if not video_path.exists():
        return []
    if not video_path.is_dir():
        return []

    files = [path for path in video_path.rglob("*") if path.is_file()]
    video_files = [path for path in files if path.suffix.lower() in VIDEO_SUFFIXES]
    if video_files:
        return sorted(video_files)
    return sorted(files)


def normalize_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.as_posix().lower()).strip("_")


def infer_view_name(path: Path) -> str | None:
    normalized = normalize_name(path)
    if any(alias in normalized for alias in EXCLUDED_VIEW_ALIASES):
        return None
    for canonical_name, aliases in VIEW_ALIASES:
        for alias in aliases:
            if alias in normalized:
                return canonical_name
    return None


def resolve_camera_videos(video_path: str) -> tuple[dict[str, str] | None, str | None]:
    base_path = Path(video_path)
    candidate_files = resolve_candidate_files(base_path)
    if not candidate_files:
        return None, f"video path cannot be expanded: {base_path}"

    matched: dict[str, Path] = {}
    for candidate in candidate_files:
        view_name = infer_view_name(candidate)
        if view_name is None:
            continue
        if view_name in matched:
            return None, f"multiple files matched view '{view_name}': {base_path}"
        matched[view_name] = candidate

    missing_views = [view for view in DEFAULT_CAMERA_VIEWS if view not in matched]
    if missing_views:
        return None, f"missing expected views: {', '.join(missing_views)}"

    return {view: str(matched[view]) for view in DEFAULT_CAMERA_VIEWS}, None


def build_sample_id(dataset_name: str, video_path: str) -> str:
    name = Path(video_path).name.strip() or "sample"
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "sample"
    digest = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:12]
    return f"{dataset_name}_{safe_name}_{digest}"


def build_sample_payload(dataset_name: str, video_path: str, tags: list[str], camera_videos: dict[str, str]) -> dict[str, Any]:
    return {
        "sample_id": build_sample_id(dataset_name, video_path),
        "generated_video": camera_videos["camera_front"],
        "reference_video": None,
        "prompt": "",
        "objects": [],
        "metadata": {
            "source_video_dir": video_path,
            "camera_videos": camera_videos,
            "tags": list(tags),
        },
    }


def select_samples(
    unique_samples: dict[str, dict[str, Any]],
    *,
    dataset_name: str,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = list(unique_samples.values())
    random.Random(seed).shuffle(records)

    selected: list[dict[str, Any]] = []
    invalid_reasons: dict[str, int] = {}
    scanned_count = 0

    for record in records:
        if len(selected) >= sample_size:
            break
        scanned_count += 1
        camera_videos, error = resolve_camera_videos(record["video"])
        if camera_videos is None:
            invalid_reasons[error or "unknown error"] = invalid_reasons.get(error or "unknown error", 0) + 1
            continue
        selected.append(
            build_sample_payload(
                dataset_name=dataset_name,
                video_path=record["video"],
                tags=record["tags"],
                camera_videos=camera_videos,
            )
        )

    stats = {
        "unique_candidate_count": len(records),
        "scanned_candidate_count": scanned_count,
        "selected_count": len(selected),
        "invalid_reason_counts": invalid_reasons,
    }
    return selected, stats


def build_output_payload(
    *,
    dataset_name: str,
    source_path: Path,
    sample_size: int,
    seed: int,
    timestamp: str,
    samples: list[dict[str, Any]],
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "source_pkl": str(source_path),
        "sample_size": sample_size,
        "seed": seed,
        "timestamp": timestamp,
        "selected_count": len(samples),
        "unique_candidate_count": stats["unique_candidate_count"],
        "scanned_candidate_count": stats["scanned_candidate_count"],
        "invalid_reason_counts": stats["invalid_reason_counts"],
        "samples": samples,
    }


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def build_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source_path = Path(args.path)
    output_dir = Path(args.output_dir)
    dataset_name = str(args.dataset_name).strip() or DEFAULT_DATASET_NAME
    sample_size = max(0, int(args.sample_size))
    seed = int(args.seed)
    timestamp = build_timestamp()

    payload = require_top_level_dict(load_pickle(source_path))
    tag_summary = build_tag_summary(payload, dataset_name=dataset_name, source_path=source_path)
    unique_samples = collect_unique_samples(payload)
    selected_samples, stats = select_samples(
        unique_samples,
        dataset_name=dataset_name,
        sample_size=sample_size,
        seed=seed,
    )
    output_payload = build_output_payload(
        dataset_name=dataset_name,
        source_path=source_path,
        sample_size=sample_size,
        seed=seed,
        timestamp=timestamp,
        samples=selected_samples,
        stats=stats,
    )

    write_json(output_dir / f"{dataset_name}_tag_summary.json", tag_summary)
    write_json(output_dir / f"{dataset_name}_{sample_size}_{timestamp}.json", output_payload)

    print(f"tag_count={tag_summary['tag_count']}")
    print(f"total_tagged_items={tag_summary['total_tagged_items']}")
    print(f"unique_candidate_count={stats['unique_candidate_count']}")
    print(f"scanned_candidate_count={stats['scanned_candidate_count']}")
    print(f"selected_count={len(selected_samples)}")
    print(f"tag_summary_path={output_dir / f'{dataset_name}_tag_summary.json'}")
    print(f"sample_output_path={output_dir / f'{dataset_name}_{sample_size}_{timestamp}.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"FileNotFoundError: {exc}")
        raise SystemExit(1)
    except ValueError as exc:
        print(f"ValueError: {exc}")
        raise SystemExit(1)
