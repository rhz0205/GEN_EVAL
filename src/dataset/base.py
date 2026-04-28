from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas import GenerationSample


DEFAULT_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)


def load_json(path: str | Path) -> Any:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_samples(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return payload["samples"]
    raise ValueError("Data JSON must be a top-level list or an object containing a 'samples' list.")


def load_data_records(path: str | Path) -> list[dict[str, Any]]:
    records = extract_samples(load_json(path))
    return [item for item in records if isinstance(item, dict)]


class BaseDataset:
    name = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        configured_name = self.config.get("name")
        self.name = str(configured_name).strip() if configured_name else self.name
        self.data_file = Path(str(self.config.get("data_file", "")).strip())
        self.camera_videos_key = str(
            self.config.get("camera_videos_key", "camera_videos")
        ).strip() or "camera_videos"
        expected_camera_views = self.config.get("expected_camera_views")
        if isinstance(expected_camera_views, list) and expected_camera_views:
            self.expected_camera_views = tuple(str(item) for item in expected_camera_views)
        else:
            self.expected_camera_views = DEFAULT_CAMERA_VIEWS

    def load(self) -> list[GenerationSample]:
        records = extract_samples(self.load_payload())
        return [self.normalize_sample(record, index=index) for index, record in enumerate(records)]

    def inspect(self, *, check_paths: bool = False) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "dataset_name": self.name,
            "data_file": str(self.data_file),
            "camera_videos_key": self.camera_videos_key,
            "expected_camera_views": list(self.expected_camera_views),
            "file_exists": self.data_file.is_file(),
            "samples_format_valid": False,
            "num_samples": 0,
            "num_valid_samples": 0,
            "num_invalid_samples": 0,
            "check_paths": check_paths,
            "invalid_samples": [],
        }
        if not self.data_file.is_file():
            summary["error"] = f"Data file does not exist: {self.data_file}"
            return summary

        payload = load_json(self.data_file)
        records = extract_samples(payload)
        summary["samples_format_valid"] = True
        summary["num_samples"] = len(records)

        invalid_samples: list[dict[str, Any]] = []
        valid_count = 0
        for index, record in enumerate(records):
            sample_id, reasons = self.inspect_sample(
                record,
                index=index,
                check_paths=check_paths,
            )
            if reasons:
                invalid_samples.append({"sample_id": sample_id, "reasons": reasons})
            else:
                valid_count += 1

        summary["num_valid_samples"] = valid_count
        summary["num_invalid_samples"] = len(invalid_samples)
        summary["invalid_samples"] = invalid_samples
        return summary

    def load_valid_samples(self, *, check_paths: bool = False) -> list[GenerationSample]:
        records = extract_samples(self.load_payload())
        valid_samples: list[GenerationSample] = []
        for index, record in enumerate(records):
            _, reasons = self.inspect_sample(record, index=index, check_paths=check_paths)
            if reasons:
                continue
            valid_samples.append(self.normalize_sample(record, index=index))
        return valid_samples

    def load_payload(self) -> Any:
        if not str(self.data_file):
            raise ValueError("Dataset config must define a non-empty data_file.")
        if not self.data_file.is_file():
            raise FileNotFoundError(f"Data file does not exist: {self.data_file}")
        return load_json(self.data_file)

    def normalize_sample(self, record: Any, *, index: int) -> GenerationSample:
        if not isinstance(record, dict):
            raise ValueError(f"Sample at index {index} must be an object.")

        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"Sample at index {index} must define a non-empty sample_id.")

        metadata = record.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Sample '{sample_id}' metadata must be an object.")

        camera_videos = metadata.get(self.camera_videos_key)
        if camera_videos is not None and not isinstance(camera_videos, dict):
            raise ValueError(
                f"Sample '{sample_id}' metadata['{self.camera_videos_key}'] must be a dict when provided."
            )

        normalized = dict(record)
        normalized["sample_id"] = sample_id
        normalized["generated_video"] = self._normalize_generated_video(record.get("generated_video"))
        normalized["metadata"] = dict(metadata)
        return GenerationSample.from_dict(normalized)

    def inspect_sample(
        self,
        record: Any,
        *,
        index: int,
        check_paths: bool,
    ) -> tuple[str, list[str]]:
        fallback_sample_id = f"index_{index}"
        reasons: list[str] = []

        if not isinstance(record, dict):
            return fallback_sample_id, ["sample must be an object"]

        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            sample_id = fallback_sample_id
            reasons.append("sample_id is required and must be non-empty")

        metadata = record.get("metadata")
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            return sample_id, reasons + ["metadata must be an object when provided"]

        camera_videos = metadata.get(self.camera_videos_key)
        if camera_videos is None:
            return sample_id, reasons
        if not isinstance(camera_videos, dict):
            return sample_id, reasons + [
                f"metadata['{self.camera_videos_key}'] must be a dict when provided"
            ]

        missing_views = [view for view in self.expected_camera_views if view not in camera_videos]
        if missing_views:
            reasons.append(f"missing expected camera views: {', '.join(missing_views)}")

        for view in self.expected_camera_views:
            if view not in camera_videos:
                continue
            path_value = camera_videos.get(view)
            if path_value is None or not str(path_value).strip():
                reasons.append(f"{view} path is required and must be non-empty")
                continue
            if check_paths:
                path = Path(str(path_value))
                if not path.exists():
                    reasons.append(f"{view} path does not exist: {path}")
                elif not path.is_file():
                    reasons.append(f"{view} path is not a file: {path}")

        return sample_id, dedupe_preserve_order(reasons)

    @staticmethod
    def _normalize_generated_video(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        return text if text else ""


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
