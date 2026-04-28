from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any


def load_pkl(path: Path) -> Any:
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def make_sample_id(video_path: str) -> str:
    path = Path(video_path)
    name = path.stem if path.suffix else path.name
    return name.replace(" ", "_")


def normalize_path(path_value: Any) -> str | None:
    if path_value is None:
        return None
    return str(path_value)


def path_exists(path_value: str | None) -> bool:
    if not path_value:
        return False
    return Path(path_value).exists()


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


def detect_camera_videos(
    video_path: str,
    primary_camera: str = "camera_front",
    camera_ext: str = ".mp4",
    keep_front_tele: bool = False,
) -> dict[str, Any]:
    path = Path(video_path)
    result: dict[str, Any] = {
        "generated_video": video_path,
        "camera_videos": {},
        "views": [],
        "multi_view_video_dir": None,
        "primary_camera": None,
        "fallback_primary_camera": False,
    }

    if not path.exists() or not path.is_dir():
        return result

    ext = camera_ext if camera_ext.startswith(".") else f".{camera_ext}"
    camera_files = sorted(path.glob(f"*{ext}"))

    camera_videos: dict[str, str] = {}
    for camera_file in camera_files:
        if not camera_file.is_file():
            continue
        camera_name = camera_file.stem
        if camera_name == "camera_front_tele" and not keep_front_tele:
            continue
        camera_videos[camera_name] = str(camera_file.resolve())

    if not camera_videos:
        return result

    views = sorted(camera_videos.keys())
    selected_camera = None
    selected_video = None
    fallback = False

    if primary_camera in camera_videos:
        selected_camera = primary_camera
        selected_video = camera_videos[primary_camera]
    else:
        selected_camera = views[0]
        selected_video = camera_videos[selected_camera]
        fallback = True

    result.update(
        {
            "generated_video": selected_video,
            "camera_videos": camera_videos,
            "views": views,
            "multi_view_video_dir": str(path.resolve()),
            "primary_camera": selected_camera,
            "fallback_primary_camera": fallback,
        }
    )
    return result


def convert_pkl_to_manifest(
    data: dict[str, list[dict[str, Any]]],
    source_pkl: str,
    dataset_name: str | None = None,
    dataset_split: str | None = None,
    video_field: str = "video",
    hdmap_field: str = "hdmap",
    selected_keys: set[str] | None = None,
    dedupe: bool = True,
    check_exists: bool = False,
    limit_per_key: int | None = None,
    limit_total: int | None = None,
    sample_per_key: int | None = None,
    sample_total: int | None = None,
    seed: int = 42,
    keep_extra_fields: bool = True,
    detect_camera_video_dirs: bool = False,
    primary_camera: str = "camera_front",
    camera_ext: str = ".mp4",
    keep_front_tele: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    stats: dict[str, Any] = {
        "source_pkl": source_pkl,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "top_level_keys": len(data),
        "selected_keys": [],
        "raw_items_seen": 0,
        "candidate_items": 0,
        "sampled_items": 0,
        "raw_items_used": 0,
        "missing_video": 0,
        "missing_path": 0,
        "duplicates_merged": 0,
        "output_samples": 0,
        "dedupe": dedupe,
        "check_exists": check_exists,
        "seed": seed,
        "limit_per_key": limit_per_key,
        "limit_total": limit_total,
        "sample_per_key": sample_per_key,
        "sample_total": sample_total,
        "video_field": video_field,
        "hdmap_field": hdmap_field,
        "keep_extra_fields": keep_extra_fields,
        "detect_camera_video_dirs": detect_camera_video_dirs,
        "primary_camera": primary_camera,
        "camera_ext": camera_ext,
        "keep_front_tele": keep_front_tele,
        "multi_view_dirs_detected": 0,
        "camera_video_records": 0,
        "primary_camera_missing": 0,
        "per_key": {},
    }

    candidates: list[tuple[str, dict[str, Any]]] = []

    for key, items in data.items():
        if selected_keys is not None and key not in selected_keys:
            continue

        if not isinstance(items, list):
            stats["per_key"][str(key)] = {
                "total_in_pkl": None,
                "seen": 0,
                "valid_candidates": 0,
                "sampled": 0,
                "missing_video": 0,
                "missing_path": 0,
                "skipped_reason": f"expected list, got {type(items).__name__}",
            }
            continue

        key_text = str(key)
        stats["selected_keys"].append(key_text)
        valid_items_for_key: list[tuple[str, dict[str, Any]]] = []
        seen_this_key = 0
        missing_video_this_key = 0
        missing_path_this_key = 0

        iterable = items[:limit_per_key] if limit_per_key is not None else items
        for item in iterable:
            stats["raw_items_seen"] += 1
            seen_this_key += 1

            if not isinstance(item, dict):
                continue

            video = normalize_path(item.get(video_field))
            if not video:
                stats["missing_video"] += 1
                missing_video_this_key += 1
                continue

            if check_exists and not path_exists(video):
                stats["missing_path"] += 1
                missing_path_this_key += 1
                continue

            valid_items_for_key.append((key_text, item))

        stats["candidate_items"] += len(valid_items_for_key)

        if sample_per_key is not None and len(valid_items_for_key) > sample_per_key:
            selected_for_key = rng.sample(valid_items_for_key, sample_per_key)
        else:
            selected_for_key = valid_items_for_key

        candidates.extend(selected_for_key)
        stats["per_key"][key_text] = {
            "total_in_pkl": len(items),
            "seen": seen_this_key,
            "valid_candidates": len(valid_items_for_key),
            "sampled": len(selected_for_key),
            "missing_video": missing_video_this_key,
            "missing_path": missing_path_this_key,
        }

    if sample_total is not None and len(candidates) > sample_total:
        candidates = rng.sample(candidates, sample_total)
    if limit_total is not None:
        candidates = candidates[:limit_total]

    stats["sampled_items"] = len(candidates)

    records_by_key: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for key, item in candidates:
        original_video = normalize_path(item.get(video_field))
        hdmap = normalize_path(item.get(hdmap_field))
        if not original_video:
            continue

        camera_info = {
            "generated_video": original_video,
            "camera_videos": {},
            "views": [],
            "multi_view_video_dir": None,
            "primary_camera": None,
            "fallback_primary_camera": False,
        }
        if detect_camera_video_dirs:
            camera_info = detect_camera_videos(
                original_video,
                primary_camera=primary_camera,
                camera_ext=camera_ext,
                keep_front_tele=keep_front_tele,
            )

        generated_video = str(camera_info["generated_video"])
        if check_exists and not path_exists(generated_video):
            stats["missing_path"] += 1
            continue

        if camera_info.get("multi_view_video_dir"):
            stats["multi_view_dirs_detected"] += 1
            stats["camera_video_records"] += len(camera_info.get("camera_videos", {}))
            if camera_info.get("fallback_primary_camera"):
                stats["primary_camera_missing"] += 1

        sample_id = make_sample_id(original_video)
        dedupe_key = original_video

        if dedupe and dedupe_key in records_by_key:
            existing = records_by_key[dedupe_key]
            metadata = existing.setdefault("metadata", {})
            tags = metadata.setdefault("tags", [])
            if key not in tags:
                tags.append(key)
            source_keys = metadata.setdefault("source_keys", [])
            if key not in source_keys:
                source_keys.append(key)
            if hdmap and not metadata.get("hdmap"):
                metadata["hdmap"] = hdmap
            stats["duplicates_merged"] += 1
            stats["raw_items_used"] += 1
            continue

        metadata: dict[str, Any] = {
            "hdmap": hdmap,
            "tags": [key],
            "source_pkl": source_pkl,
            "source_key": key,
            "source_keys": [key],
            "original_video_path": original_video,
        }
        if dataset_name:
            metadata["dataset_name"] = dataset_name
        if dataset_split:
            metadata["dataset_split"] = dataset_split
        if camera_info.get("multi_view_video_dir"):
            metadata["multi_view_video_dir"] = camera_info["multi_view_video_dir"]
            metadata["camera_videos"] = camera_info["camera_videos"]
            metadata["views"] = camera_info["views"]
            metadata["primary_camera"] = camera_info["primary_camera"]
            metadata["fallback_primary_camera"] = camera_info["fallback_primary_camera"]
        if keep_extra_fields:
            raw_item = {
                str(item_key): to_jsonable(item_value)
                for item_key, item_value in item.items()
                if item_key not in {video_field, hdmap_field}
            }
            if raw_item:
                metadata["raw_item"] = raw_item

        record = {
            "sample_id": sample_id,
            "generated_video": generated_video,
            "reference_video": None,
            "prompt": "",
            "objects": [],
            "metadata": metadata,
        }

        if dedupe:
            records_by_key[dedupe_key] = record
        else:
            record["sample_id"] = f"{sample_id}__{key}"

        records.append(record)
        stats["raw_items_used"] += 1

    stats["output_samples"] = len(records)
    return records, stats


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_shards(output: Path, records: list[dict[str, Any]], shard_size: int) -> list[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.stem
    suffix = output.suffix or ".json"
    shard_paths: list[str] = []
    for start in range(0, len(records), shard_size):
        shard_idx = start // shard_size
        shard_records = records[start : start + shard_size]
        shard_path = output.parent / f"{stem}_part_{shard_idx:05d}{suffix}"
        write_json(shard_path, shard_records)
        shard_paths.append(str(shard_path))
    return shard_paths


def parse_keys(keys: str | None) -> set[str] | None:
    if not keys:
        return None
    parsed = {item.strip() for item in keys.split(",") if item.strip()}
    return parsed or None


def infer_dataset_split(dataset_name: str | None) -> str | None:
    mapping = {
        "sample_data": "sample",
        "geely_data": "geely",
        "cosmos_data": "cosmos",
        "real_data": "real",
    }
    if not dataset_name:
        return None
    return mapping.get(dataset_name)


def infer_output_path(dataset_name: str | None, dataset_split: str | None) -> Path | None:
    effective_split = dataset_split or infer_dataset_split(dataset_name)
    if not dataset_name or not effective_split:
        return None
    return Path("manifests") / f"{effective_split}.json"


def infer_stats_path(dataset_name: str | None, dataset_split: str | None) -> Path | None:
    effective_split = dataset_split or infer_dataset_split(dataset_name)
    if not dataset_name or not effective_split:
        return None
    return Path("outputs") / dataset_name / f"{effective_split}_manifest_stats.json"


def get_path_type_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"file": 0, "dir": 0, "missing": 0, "empty/null": 0}
    for record in records:
        value = record.get("generated_video")
        if value is None or value == "":
            counts["empty/null"] += 1
            continue
        path = Path(str(value))
        if path.is_file():
            counts["file"] += 1
        elif path.is_dir():
            counts["dir"] += 1
        else:
            counts["missing"] += 1
    return counts


def summarize_records(
    records: list[dict[str, Any]],
    output_manifest_path: Path | None,
    stats_path: Path | None,
    dataset_name: str | None,
    dataset_split: str | None,
    keep_front_tele: bool,
) -> None:
    path_type_counts = get_path_type_counts(records)
    camera_videos_present = 0
    view_count_distribution: dict[int, int] = {}

    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        camera_videos = metadata.get("camera_videos")
        if isinstance(camera_videos, dict) and camera_videos:
            camera_videos_present += 1
        views = metadata.get("views")
        if isinstance(views, list):
            view_count = len(views)
            view_count_distribution[view_count] = view_count_distribution.get(view_count, 0) + 1

    print("\nSummary")
    print(f"output_manifest_path: {output_manifest_path}")
    print(f"num_samples: {len(records)}")
    if dataset_name:
        print(f"dataset_name: {dataset_name}")
    if dataset_split:
        print(f"dataset_split: {dataset_split}")
    print(f"camera_front_tele: {'kept' if keep_front_tele else 'excluded'}")
    print("generated_video_path_type_counts:")
    for key in ("file", "dir", "missing", "empty/null"):
        print(f"  {key}: {path_type_counts.get(key, 0)}")
    print(f"samples_with_camera_videos: {camera_videos_present}")
    print(f"view_count_distribution: {dict(sorted(view_count_distribution.items()))}")
    if stats_path is not None:
        print(f"stats_output_path: {stats_path}")


def build_manifest_from_pkl(
    pkl_path: str | Path,
    output: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_split: str | None = None,
    video_field: str = "video",
    hdmap_field: str = "hdmap",
    keys: str | None = None,
    limit_per_key: int | None = None,
    limit_total: int | None = None,
    sample_per_key: int | None = None,
    sample_total: int | None = None,
    seed: int = 42,
    dedupe: bool = True,
    check_exists: bool = False,
    keep_extra_fields: bool = True,
    detect_camera_videos_dirs: bool = False,
    primary_camera: str = "camera_front",
    camera_ext: str = ".mp4",
    keep_front_tele: bool = False,
    shard_size: int | None = None,
    stats_output: str | Path | None = None,
) -> dict[str, Any]:
    pkl_path = Path(pkl_path).expanduser().resolve()
    effective_dataset_split = dataset_split or infer_dataset_split(dataset_name)

    if output:
        output_path = Path(output).expanduser().resolve()
    else:
        inferred_output = infer_output_path(dataset_name, effective_dataset_split)
        if inferred_output is None:
            raise ValueError(
                "Either output or a dataset name that can infer a default split must be provided."
            )
        output_path = inferred_output.expanduser().resolve()

    data = load_pkl(pkl_path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected pkl top-level type dict, got {type(data).__name__}")

    selected_keys = parse_keys(keys)
    records, stats = convert_pkl_to_manifest(
        data=data,
        source_pkl=str(pkl_path),
        dataset_name=dataset_name,
        dataset_split=effective_dataset_split,
        video_field=video_field,
        hdmap_field=hdmap_field,
        selected_keys=selected_keys,
        dedupe=dedupe,
        check_exists=check_exists,
        limit_per_key=limit_per_key,
        limit_total=limit_total,
        sample_per_key=sample_per_key,
        sample_total=sample_total,
        seed=seed,
        keep_extra_fields=keep_extra_fields,
        detect_camera_video_dirs=detect_camera_videos_dirs,
        primary_camera=primary_camera,
        camera_ext=camera_ext,
        keep_front_tele=keep_front_tele,
    )

    if shard_size:
        shard_paths = write_shards(output_path, records, shard_size)
        stats["manifest_shards"] = shard_paths
        stats["output_manifest"] = None
    else:
        write_json(output_path, records)
        shard_paths = []
        stats["output_manifest"] = str(output_path)

    if stats_output:
        stats_path = Path(stats_output).expanduser().resolve()
    elif dataset_name and effective_dataset_split:
        stats_path = infer_stats_path(dataset_name, effective_dataset_split)
        assert stats_path is not None
        stats_path = stats_path.expanduser().resolve()
    else:
        stats_path = output_path.with_suffix(".stats.json")

    write_json(stats_path, stats)

    return {
        "records": records,
        "stats": stats,
        "output_path": output_path,
        "stats_path": stats_path,
        "effective_dataset_split": effective_dataset_split,
        "shard_paths": shard_paths,
        "sharded": bool(shard_size),
        "dataset_name": dataset_name,
        "keep_front_tele": keep_front_tele,
    }
