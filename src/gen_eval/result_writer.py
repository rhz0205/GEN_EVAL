from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def write_results(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def default_output_dir(dataset: str, run_name: str) -> str:
    return str(Path("outputs") / dataset / run_name)


def safe_filename(text: str) -> str:
    allowed = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            allowed.append(ch)
        else:
            allowed.append("_")
    name = "".join(allowed).strip("_")
    return name or "eval"


def make_timestamp(created_at: datetime | None = None) -> str:
    created_time = created_at or datetime.now()
    return created_time.strftime("%Y%m%d_%H%M%S")


def resolve_output_paths(
    output_dir: str | Path,
    timestamp: str,
    explicit_output: str | None = None,
) -> dict[str, Path]:
    if explicit_output:
        result_path = Path(explicit_output)
        run_dir = result_path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_dir": run_dir,
            "timestamp_dir": run_dir,
            "result_path": result_path,
            "summary_path": run_dir / "summary.txt",
        }

    run_dir = Path(output_dir)
    timestamp_dir = run_dir / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "timestamp_dir": timestamp_dir,
        "result_path": timestamp_dir / "result.json",
        "summary_path": timestamp_dir / "summary.txt",
    }


def get_command_string() -> str:
    return " ".join(sys.argv)


def build_result_payload(
    resolved_config: dict[str, Any],
    evaluation_result: dict[str, Any],
    *,
    command: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    created_time = created_at or datetime.now()
    timestamp = make_timestamp(created_time)
    manifest_path = resolved_config.get("manifest_path")
    metric_results = evaluation_result.get("results", [])

    return {
        "run": {
            "run_name": resolved_config.get("run_name"),
            "dataset": resolved_config.get("dataset"),
            "dataset_name": resolved_config.get("dataset_name"),
            "created_at": created_time.isoformat(timespec="seconds"),
            "timestamp": timestamp,
            "command": command,
        },
        "runtime": resolved_config.get("runtime", {}),
        "manifest": {
            "path": manifest_path,
            "num_samples": evaluation_result.get("num_samples"),
        },
        "config": {
            "run_config_path": resolved_config.get("config_path"),
            "dataset_config_path": resolved_config.get("dataset_config_path"),
            "metric_config_path": resolved_config.get("metric_config_path"),
            "selected_metrics": resolved_config.get("selected_metrics", []),
            "output_root": resolved_config.get("output_dir"),
        },
        "metrics": metric_results if isinstance(metric_results, list) else [],
    }


def save_result_bundle(
    payload: dict[str, Any],
    output_dir: str | Path,
    explicit_output: str | None = None,
) -> dict[str, Path]:
    from gen_eval.result_summary import format_result_summary

    run_info = payload.get("run")
    timestamp = None
    if isinstance(run_info, dict):
        raw_timestamp = run_info.get("timestamp")
        if isinstance(raw_timestamp, str) and raw_timestamp:
            timestamp = raw_timestamp
    if not timestamp:
        timestamp = make_timestamp()

    output_paths = resolve_output_paths(output_dir, timestamp, explicit_output)
    write_results(output_paths["result_path"], payload)

    summary_lines = format_result_summary(payload, output_paths["result_path"])
    output_paths["summary_path"].write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    return output_paths
