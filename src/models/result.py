from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_output_layout(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    results_dir = root / "results"
    logs_dir = root / "logs"
    visualizations_dir = root / "visualizations"

    paths = {
        "output_dir": root,
        "results_dir": results_dir,
        "logs_dir": logs_dir,
        "visualizations_dir": visualizations_dir,
        "depth_raw": visualizations_dir / "depth_raw",
        "semantic_raw": visualizations_dir / "semantic_raw",
        "multiview_match_raw": visualizations_dir / "multiview_match_raw",
        "depth_6v_image": visualizations_dir / "depth_6v_image",
        "semantic_6v_image": visualizations_dir / "semantic_6v_image",
        "multiview_match_6v_image": visualizations_dir / "multiview_match_6v_image",
        "depth_6v_video": visualizations_dir / "depth_6v_video",
        "semantic_6v_video": visualizations_dir / "semantic_6v_video",
        "multiview_match_6v_video": visualizations_dir / "multiview_match_6v_video",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(payload: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def build_summary(metrics_result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    failed_metrics: dict[str, Any] = {}

    for metric_name, result in metrics_result.items():
        if not isinstance(result, dict):
            summary[metric_name] = {"status": "failed", "reason": "metric result is not an object"}
            failed_metrics[metric_name] = summary[metric_name]
            continue

        item: dict[str, Any] = {"status": result.get("status", "unknown")}
        for key in (
            "pass_rate",
            "view_consistency_score",
            "mean_view_consistency_score",
            "mean_temporal_consistency_score",
            "mean_appearance_consistency_score",
            "mean_depth_consistency_score",
            "mean_semantic_consistency_score",
            "mean_instance_consistency_score",
            "valid_sample_count",
            "invalid_sample_count",
        ):
            if key in result:
                item[key] = result.get(key)
        if "reason" in result:
            item["reason"] = result.get("reason")
        if "error" in result:
            item["error"] = result.get("error")

        summary[metric_name] = item
        if item["status"] == "failed":
            failed_metrics[metric_name] = item

    summary["_failed_metrics"] = failed_metrics
    return summary


def collect_failed_samples(metrics_result: dict[str, Any]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for metric_name, result in metrics_result.items():
        if not isinstance(result, dict):
            continue
        details = result.get("details")
        if not isinstance(details, dict):
            continue
        failed_samples = details.get("failed_samples")
        skipped_samples = details.get("skipped_samples")
        item: dict[str, Any] = {}
        if isinstance(failed_samples, list) and failed_samples:
            item["failed_samples"] = failed_samples
        if isinstance(skipped_samples, list) and skipped_samples:
            item["skipped_samples"] = skipped_samples
        if item:
            aggregated[metric_name] = item
    return aggregated


def write_metrics_result(
    *,
    metrics_result: dict[str, Any],
    path: str | Path,
    dataset_name: str,
    data_count: int,
    timestamp: str,
    data_file: str | Path,
) -> dict[str, Any]:
    payload = {
        "dataset_name": dataset_name,
        "data_count": data_count,
        "timestamp": timestamp,
        "data_file": str(data_file),
        "num_samples": _infer_num_samples(metrics_result),
        "results": metrics_result,
    }
    write_json(payload, path)
    return payload


def write_summary_result(
    *,
    metrics_result: dict[str, Any],
    summary_path: str | Path,
    failed_samples_path: str | Path,
    dataset_name: str,
    data_count: int,
    timestamp: str,
    data_file: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_payload = {
        "dataset_name": dataset_name,
        "data_count": data_count,
        "timestamp": timestamp,
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "num_samples": _infer_num_samples(metrics_result),
        "summary": build_summary(metrics_result),
    }
    failed_samples_payload = {
        "dataset_name": dataset_name,
        "data_count": data_count,
        "timestamp": timestamp,
        "failed_samples": collect_failed_samples(metrics_result),
    }
    write_json(summary_payload, summary_path)
    write_json(failed_samples_payload, failed_samples_path)
    return summary_payload, failed_samples_payload


def _infer_num_samples(metrics_result: dict[str, Any]) -> int | None:
    for result in metrics_result.values():
        if isinstance(result, dict):
            value = result.get("num_samples")
            if isinstance(value, int):
                return value
    return None
