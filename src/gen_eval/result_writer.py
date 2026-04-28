from __future__ import annotations

import json
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


def resolve_output_path(
    output_dir: str | Path,
    config_path: str | Path,
    explicit_output: str | None = None,
) -> Path:
    if explicit_output:
        output_path = Path(explicit_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_stem = safe_filename(Path(config_path).stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{config_stem}_results_{timestamp}.json"


def build_result_payload(
    resolved_config: dict[str, Any],
    evaluation_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_name": resolved_config.get("run_name"),
        "dataset": resolved_config.get("dataset"),
        "dataset_name": resolved_config.get("dataset_name"),
        "manifest_path": resolved_config.get("manifest_path"),
        "config_path": resolved_config.get("config_path"),
        "output_dir": resolved_config.get("output_dir"),
        "runtime": resolved_config.get("runtime", {}),
        "results": evaluation_result.get("results", []),
        "num_samples": evaluation_result.get("num_samples"),
    }


def save_result_payload(
    payload: dict[str, Any],
    output_dir: str | Path,
    config_path: str | Path,
    explicit_output: str | None = None,
) -> Path:
    output_path = resolve_output_path(output_dir, config_path, explicit_output)
    write_results(output_path, payload)
    return output_path
