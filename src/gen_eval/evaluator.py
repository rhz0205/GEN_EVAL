from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gen_eval.config import load_yaml
from gen_eval.dataset import load_manifest
from gen_eval.metrics import register_builtin_metrics
from gen_eval.registry import registry
from gen_eval.result_writer import write_results

VIDEO_INTEGRITY_METRIC = "video_integrity"
VIDEO_INTEGRITY_GATE_REASON = "Skipped because video_integrity did not pass."

QUALITY_METRIC_SCORE_FIELDS: dict[str, str] = {
    "view_consistency": "view_consistency_score",
    "temporal_consistency": "temporal_consistency_score",
    "appearance_consistency": "appearance_consistency_score",
    "depth_consistency": "depth_consistency_score",
    "semantic_consistency": "semantic_consistency_score",
    "instance_consistency": "instance_consistency_score",
}

QUALITY_METRIC_MEAN_FIELDS: dict[str, str] = {
    "view_consistency": "mean_view_consistency_score",
    "temporal_consistency": "mean_temporal_consistency_score",
    "appearance_consistency": "mean_appearance_consistency_score",
    "depth_consistency": "mean_depth_consistency_score",
    "semantic_consistency": "mean_semantic_consistency_score",
    "instance_consistency": "mean_instance_consistency_score",
}

QUALITY_METRIC_MEAN_SOURCE_FIELDS: dict[str, str] = {
    "view_consistency": "view_consistency_score",
    "temporal_consistency": "mean_temporal_consistency_score",
    "appearance_consistency": "mean_appearance_consistency_score",
    "depth_consistency": "mean_depth_consistency_score",
    "semantic_consistency": "mean_semantic_consistency_score",
    "instance_consistency": "mean_instance_consistency_score",
}

class Evaluator:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        register_builtin_metrics()

    @classmethod
    def from_config_path(cls, path: str | Path) -> "Evaluator":
        config_path = Path(path)
        return cls(_load_config(config_path))

    def run(self) -> dict[str, Any]:
        manifest_path = self.config["manifest_path"]
        samples = load_manifest(manifest_path)

        results = []
        enabled_metrics = iter_enabled_metrics(self.config.get("metrics", {}))
        video_integrity_gate: dict[str, Any] | None = None
        total_samples = len(samples)

        for metric_name, metric_config in enabled_metrics:
            if metric_name == VIDEO_INTEGRITY_METRIC:
                metric_result = run_metric(metric_name, metric_config, samples)
                results.append(metric_result)
                video_integrity_gate = build_video_integrity_gate(samples, metric_result)
                continue

            metric_samples = samples
            if video_integrity_gate is not None and is_quality_metric(metric_name):
                metric_samples = list(video_integrity_gate["valid_samples"])

            if (
                video_integrity_gate is not None
                and is_quality_metric(metric_name)
                and not metric_samples
            ):
                metric_result = build_gated_metric_result(
                    metric_name,
                    total_samples=total_samples,
                    invalid_sample_ids=list(video_integrity_gate["invalid_sample_ids"]),
                )
            else:
                metric_result = run_metric(metric_name, metric_config, metric_samples)
                if video_integrity_gate is not None and is_quality_metric(metric_name):
                    metric_result = apply_video_integrity_gate(
                        metric_name,
                        metric_result,
                        total_samples=total_samples,
                        invalid_sample_ids=list(video_integrity_gate["invalid_sample_ids"]),
                    )

            results.append(metric_result)

        payload = {
            "manifest_path": str(manifest_path),
            "num_samples": len(samples),
            "results": results,
        }

        output_path = self.config.get("output_path")
        if output_path:
            write_results(output_path, payload)
        return payload

def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return load_yaml(path)

def iter_enabled_metrics(metrics_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    enabled = []
    for metric_name, raw_config in metrics_config.items():
        metric_config = raw_config or {}
        if metric_config.get("enabled", False):
            enabled.append((metric_name, metric_config))
    video_integrity_items = [
        item for item in enabled if item[0] == VIDEO_INTEGRITY_METRIC
    ]
    other_items = [item for item in enabled if item[0] != VIDEO_INTEGRITY_METRIC]
    return video_integrity_items + other_items

def run_metric(
    metric_name: str,
    metric_config: dict[str, Any],
    samples: list[Any],
) -> dict[str, Any]:
    metric_class = registry.get(metric_name)
    metric = metric_class(metric_config)
    result = metric.evaluate(samples)
    return normalize_metric_result(metric_name, result)

def normalize_metric_result(metric_name: str, result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError(f"Metric '{metric_name}' must return a dict, got {type(result).__name__}.")

    normalized = dict(result)
    normalized.setdefault("metric", metric_name)
    normalized.setdefault("num_samples", 0)

    details = normalized.get("details")
    normalized["details"] = details if isinstance(details, dict) else {}

    status = normalized.get("status")
    normalized_status = str(status) if status is not None else "failed"
    if normalized_status == "ok":
        normalized_status = "success"
    normalized["status"] = normalized_status
    return normalized

def is_quality_metric(metric_name: str) -> bool:
    return metric_name in QUALITY_METRIC_SCORE_FIELDS

def get_quality_metric_score_field(metric_name: str) -> str | None:
    return QUALITY_METRIC_SCORE_FIELDS.get(metric_name)

def get_quality_metric_mean_field(metric_name: str) -> str | None:
    return QUALITY_METRIC_MEAN_FIELDS.get(metric_name)

def get_quality_metric_mean_source_field(metric_name: str) -> str | None:
    return QUALITY_METRIC_MEAN_SOURCE_FIELDS.get(metric_name)

def get_sample_id(sample: Any) -> str:
    return str(getattr(sample, "sample_id", None) or "unknown")

def build_video_integrity_gate(
    samples: list[Any],
    video_integrity_result: dict[str, Any],
) -> dict[str, Any]:
    details = video_integrity_result.get("details")
    details = details if isinstance(details, dict) else {}
    evaluated_samples = details.get("evaluated_samples")
    evaluated_samples = evaluated_samples if isinstance(evaluated_samples, list) else []

    passed_sample_ids: set[str] = set()
    for item in evaluated_samples:
        if not isinstance(item, dict):
            continue
        sample_id = item.get("sample_id")
        if sample_id is None:
            continue
        if item.get("video_integrity_passed") is True:
            passed_sample_ids.add(str(sample_id))

    valid_samples = [sample for sample in samples if get_sample_id(sample) in passed_sample_ids]
    invalid_sample_ids = [
        get_sample_id(sample) for sample in samples if get_sample_id(sample) not in passed_sample_ids
    ]

    return {
        "valid_sample_ids": passed_sample_ids,
        "invalid_sample_ids": invalid_sample_ids,
        "valid_samples": valid_samples,
    }

def build_gated_metric_placeholder(metric_name: str, sample_id: str) -> dict[str, Any]:
    score_field = get_quality_metric_score_field(metric_name)
    if score_field is None:
        raise KeyError(f"Unsupported gated quality metric: {metric_name}")
    return {
        "sample_id": sample_id,
        score_field: None,
        "gated_by_video_integrity": True,
        "reason": VIDEO_INTEGRITY_GATE_REASON,
    }

def build_gated_metric_result(
    metric_name: str,
    *,
    total_samples: int,
    invalid_sample_ids: list[str],
) -> dict[str, Any]:
    mean_field = get_quality_metric_mean_field(metric_name)
    if mean_field is None:
        raise KeyError(f"Unsupported gated quality metric: {metric_name}")

    placeholders = [
        build_gated_metric_placeholder(metric_name, sample_id)
        for sample_id in invalid_sample_ids
    ]
    return {
        "metric": metric_name,
        "status": "skipped",
        "num_samples": total_samples,
        "valid_sample_count": 0,
        "invalid_sample_count": len(invalid_sample_ids),
        mean_field: None,
        "details": {
            "evaluated_samples": placeholders,
            "skipped_samples": [],
            "failed_samples": [],
        },
        "reason": "No samples passed video_integrity.",
    }

def apply_video_integrity_gate(
    metric_name: str,
    metric_result: dict[str, Any],
    *,
    total_samples: int,
    invalid_sample_ids: list[str],
) -> dict[str, Any]:
    if not is_quality_metric(metric_name):
        return metric_result

    normalized = normalize_metric_result(metric_name, metric_result)
    details = normalized.get("details")
    if not isinstance(details, dict):
        details = {}
        normalized["details"] = details

    evaluated_samples = details.get("evaluated_samples")
    if not isinstance(evaluated_samples, list):
        evaluated_samples = []
        details["evaluated_samples"] = evaluated_samples

    skipped_samples = details.get("skipped_samples")
    if not isinstance(skipped_samples, list):
        details["skipped_samples"] = []

    failed_samples = details.get("failed_samples")
    if not isinstance(failed_samples, list):
        details["failed_samples"] = []

    placeholders = [
        build_gated_metric_placeholder(metric_name, sample_id)
        for sample_id in invalid_sample_ids
    ]
    evaluated_samples.extend(placeholders)

    valid_sample_count = _get_quality_metric_valid_count(metric_name, normalized)
    mean_field = get_quality_metric_mean_field(metric_name)
    mean_source_field = get_quality_metric_mean_source_field(metric_name)
    if mean_field is not None:
        normalized[mean_field] = normalized.get(mean_source_field)

    normalized["num_samples"] = total_samples
    normalized["valid_sample_count"] = valid_sample_count
    normalized["invalid_sample_count"] = len(invalid_sample_ids)
    return normalized

def _get_quality_metric_valid_count(metric_name: str, metric_result: dict[str, Any]) -> int:
    valid_sample_count = metric_result.get("valid_sample_count")
    if isinstance(valid_sample_count, int) and valid_sample_count >= 0:
        return valid_sample_count
    if metric_name == "view_consistency":
        valid_evaluated_count = metric_result.get("valid_evaluated_count")
        if isinstance(valid_evaluated_count, int) and valid_evaluated_count >= 0:
            return valid_evaluated_count
    return 0
