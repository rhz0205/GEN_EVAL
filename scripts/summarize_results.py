#!/usr/bin/env python3
"""Summarize a saved GEN_EVAL result JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Result JSON root must be an object.")
    return data


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _format_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def _get_list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    return value if isinstance(value, list) else []


def summarize_view_data(evaluated_samples: list[dict[str, Any]]) -> list[str]:
    num_views_counter: Counter[int] = Counter()
    per_view_scores: dict[str, list[float]] = defaultdict(list)
    avg_l2_distances: list[float] = []

    for sample in evaluated_samples:
        num_views = sample.get("num_views")
        if isinstance(num_views, int):
            num_views_counter[num_views] += 1

        if isinstance(sample.get("avg_l2_distance"), (int, float)):
            avg_l2_distances.append(float(sample["avg_l2_distance"]))

        for view_item in _get_list(sample, "view_results"):
            if not isinstance(view_item, dict):
                continue
            view_name = str(
                view_item.get("view")
                or view_item.get("camera")
                or view_item.get("name")
                or "unknown"
            )
            score = view_item.get("score")
            if isinstance(score, (int, float)):
                per_view_scores[view_name].append(float(score))
            if isinstance(view_item.get("avg_l2_distance"), (int, float)):
                avg_l2_distances.append(float(view_item["avg_l2_distance"]))

    lines: list[str] = []
    if num_views_counter:
        items = ", ".join(f"{k}:{v}" for k, v in sorted(num_views_counter.items()))
        lines.append(f"  num_views_distribution: {items}")
    if per_view_scores:
        lines.append("  per_view_mean_score:")
        for view_name in sorted(per_view_scores):
            lines.append(
                f"    {view_name}: {_format_number(_safe_mean(per_view_scores[view_name]))}"
            )
    if avg_l2_distances:
        lines.append(f"  mean_avg_l2_distance: {_format_number(_safe_mean(avg_l2_distances))}")
    return lines


def summarize_pair_data(evaluated_samples: list[dict[str, Any]]) -> list[str]:
    pair_status_counts: Counter[str] = Counter()
    pair_scores: dict[str, list[float]] = defaultdict(list)
    pair_valid_matches: dict[str, list[float]] = defaultdict(list)
    pair_mean_confidence: dict[str, list[float]] = defaultdict(list)

    for sample in evaluated_samples:
        for pair_item in _get_list(sample, "pair_results"):
            if not isinstance(pair_item, dict):
                continue
            pair_name = str(
                pair_item.get("pair")
                or pair_item.get("pair_name")
                or pair_item.get("name")
                or "unknown"
            )
            status = str(pair_item.get("status", "unknown"))
            pair_status_counts[status] += 1

            score = pair_item.get("score")
            if isinstance(score, (int, float)):
                pair_scores[pair_name].append(float(score))

            valid_matches = pair_item.get("valid_matches")
            if isinstance(valid_matches, (int, float)):
                pair_valid_matches[pair_name].append(float(valid_matches))

            mean_confidence = pair_item.get("mean_confidence")
            if isinstance(mean_confidence, (int, float)):
                pair_mean_confidence[pair_name].append(float(mean_confidence))

    lines: list[str] = []
    if pair_status_counts:
        items = ", ".join(f"{k}:{v}" for k, v in sorted(pair_status_counts.items()))
        lines.append(f"  pair_status_counts: {items}")
    if pair_scores:
        lines.append("  per_pair_mean_score:")
        for pair_name in sorted(pair_scores):
            lines.append(f"    {pair_name}: {_format_number(_safe_mean(pair_scores[pair_name]))}")
    if pair_valid_matches:
        lines.append("  per_pair_mean_valid_matches:")
        for pair_name in sorted(pair_valid_matches):
            lines.append(
                f"    {pair_name}: {_format_number(_safe_mean(pair_valid_matches[pair_name]))}"
            )
    if pair_mean_confidence:
        lines.append("  per_pair_mean_mean_confidence:")
        for pair_name in sorted(pair_mean_confidence):
            lines.append(
                f"    {pair_name}: {_format_number(_safe_mean(pair_mean_confidence[pair_name]))}"
            )
    return lines


def summarize_metric(metric_result: dict[str, Any]) -> list[str]:
    metric = metric_result.get("metric")
    status = metric_result.get("status")
    score = metric_result.get("score")
    num_samples = metric_result.get("num_samples")
    reason = metric_result.get("reason")
    details = metric_result.get("details")
    details = details if isinstance(details, dict) else {}

    evaluated_samples = [item for item in _get_list(details, "evaluated_samples") if isinstance(item, dict)]
    skipped_samples = _get_list(details, "skipped_samples")
    failed_samples = _get_list(details, "failed_samples")

    lines = [
        f"- metric={metric} status={status} score={_format_number(score)} "
        f"num_samples={_format_number(num_samples)} skipped={len(skipped_samples)} failed={len(failed_samples)}"
    ]
    if reason:
        lines.append(f"  reason: {reason}")

    view_lines = summarize_view_data(evaluated_samples)
    pair_lines = summarize_pair_data(evaluated_samples)
    lines.extend(view_lines)
    lines.extend(pair_lines)
    return lines


def summarize_results(result_path: Path) -> int:
    payload = load_json(result_path)

    print("Run summary")
    print(f"result_path: {result_path}")
    if payload.get("run_name") is not None:
        print(f"run_name: {payload.get('run_name')}")
    if payload.get("dataset_name") is not None:
        print(f"dataset_name: {payload.get('dataset_name')}")
    if payload.get("manifest_path") is not None:
        print(f"manifest_path: {payload.get('manifest_path')}")
    if payload.get("output_dir") is not None:
        print(f"output_dir: {payload.get('output_dir')}")

    runtime = payload.get("runtime")
    if isinstance(runtime, dict) and runtime.get("device") is not None:
        print(f"runtime.device: {runtime.get('device')}")

    metric_results = payload.get("results")
    metric_results = metric_results if isinstance(metric_results, list) else []
    print(f"num_metric_results: {len(metric_results)}")

    print("\nMetric summary")
    for item in metric_results:
        if not isinstance(item, dict):
            continue
        for line in summarize_metric(item):
            print(line)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a GEN_EVAL result JSON.")
    parser.add_argument("--result", required=True, help="Path to saved result JSON.")
    args = parser.parse_args()
    return summarize_results(Path(args.result))


if __name__ == "__main__":
    raise SystemExit(main())
