#!/usr/bin/env python3
"""Run GEN_EVAL evaluation from a run config."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gen_eval.evaluator import Evaluator


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read YAML configs. Install pyyaml in the evaluation environment."
        ) from exc

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object: {config_path}")
    return data


def resolve_run_config(config_path: str | Path) -> dict[str, Any]:
    run_config_path = Path(config_path)
    run_config = load_yaml(run_config_path)

    run_name = run_config_path.stem

    dataset = run_config.get("dataset")
    if not dataset or not isinstance(dataset, str):
        raise ValueError("Run config must define a string 'dataset'.")

    selected_metrics = run_config.get("metrics")
    if not isinstance(selected_metrics, list) or not selected_metrics:
        raise ValueError("Run config must define a non-empty 'metrics' list.")

    dataset_config_path = Path("configs") / "datasets" / f"{dataset}.yaml"
    metric_config_path = Path("configs") / "metrics.yaml"

    dataset_config = load_yaml(dataset_config_path)
    metric_file = load_yaml(metric_config_path)
    runtime = run_config.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise ValueError("Run config field 'runtime' must be an object.")

    metrics: dict[str, dict[str, Any]] = {}
    for metric_name in selected_metrics:
        if not isinstance(metric_name, str):
            raise ValueError("Run config field 'metrics' must contain only strings.")
        if metric_name not in metric_file:
            raise ValueError(
                f"Selected metric '{metric_name}' not found in metric config: "
                f"{metric_config_path}"
            )
        raw_metric_config = metric_file.get(metric_name)
        metric_config = dict(raw_metric_config or {})
        for key, value in runtime.items():
            metric_config.setdefault(key, value)
        metrics[metric_name] = metric_config

    dataset_name = dataset_config.get("dataset_name") or dataset
    output_dir = str(Path("outputs") / dataset / run_name)

    return {
        "run_name": run_name,
        "dataset": dataset,
        "dataset_name": dataset_name,
        "manifest_path": dataset_config.get("manifest_path"),
        "output_dir": output_dir,
        "runtime": runtime,
        "metrics": metrics,
        "save_details": True,
        "config_path": str(run_config_path.resolve()),
        "dataset_config_path": str(Path(dataset_config_path)),
        "metric_config_path": str(Path(metric_config_path)),
        "selected_metrics": list(selected_metrics),
    }


def safe_filename(text: str) -> str:
    allowed = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            allowed.append(ch)
        else:
            allowed.append("_")
    name = "".join(allowed).strip("_")
    return name or "eval"


def save_results(
    payload: dict[str, Any],
    output_dir: str,
    config_path: str | Path,
    explicit_output: str | None = None,
) -> Path:
    if explicit_output:
        output_path = Path(explicit_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        config_stem = safe_filename(Path(config_path).stem)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = out_dir / f"{config_stem}_results_{timestamp}.json"

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def print_summary(payload: dict[str, Any], output_path: Path) -> None:
    print("\nEvaluation finished.")
    print(f"Run: {payload.get('run_name')}")
    print(f"Dataset: {payload.get('dataset_name')}")
    print(f"Manifest: {payload.get('manifest_path')}")
    print(f"Output dir: {payload.get('output_dir')}")
    print(f"Saved results: {output_path}")

    metric_results = payload.get("results", [])
    if not isinstance(metric_results, list):
        return

    print("\nMetric summary:")
    for item in metric_results:
        if not isinstance(item, dict):
            continue
        print(
            f"- {item.get('metric')}: "
            f"status={item.get('status')}, "
            f"score={item.get('score')}, "
            f"num_samples={item.get('num_samples')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GEN_EVAL evaluation.")
    parser.add_argument("--config", required=True, help="Path to run config JSON file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output JSON path. If not set, save under output_dir.",
    )
    args = parser.parse_args()

    resolved_config = resolve_run_config(args.config)
    evaluator = Evaluator(resolved_config)
    results = evaluator.run()

    payload = {
        "run_name": resolved_config.get("run_name"),
        "dataset": resolved_config.get("dataset"),
        "dataset_name": resolved_config.get("dataset_name"),
        "manifest_path": resolved_config.get("manifest_path"),
        "config_path": resolved_config.get("config_path"),
        "output_dir": resolved_config.get("output_dir"),
        "runtime": resolved_config.get("runtime", {}),
        "results": results.get("results", []),
        "num_samples": results.get("num_samples"),
    }

    output_path = save_results(
        payload=payload,
        output_dir=str(resolved_config.get("output_dir") or "outputs"),
        config_path=args.config,
        explicit_output=args.output,
    )
    print_summary(payload, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
