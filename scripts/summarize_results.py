from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.result import write_summary_result

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG_PATH = Path("configs/run.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize worldbench metrics.json outputs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to run config.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--metrics-path", default=None, help="Direct path to metrics.json.")
    parser.add_argument("--summary-path", default=None, help="Direct path to summary.json.")
    parser.add_argument("--failed-samples-path", default=None, help="Direct path to failed_samples.json.")
    parser.add_argument("--print-summary", action="store_true", help="Print compact terminal summary.")
    return parser


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load worldbench config files.")
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Run config must be a YAML mapping: {path}")
    return payload


def normalize_run_config(payload: dict[str, Any]) -> dict[str, Any]:
    run_config = payload.get("run") if isinstance(payload.get("run"), dict) else payload
    if not isinstance(run_config, dict):
        raise ValueError("Run config must be a mapping.")
    return dict(run_config)


def resolve_output_dir(args: argparse.Namespace, run_config: dict[str, Any]) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    paths = run_config.get("paths")
    if isinstance(paths, dict):
        output_dir = paths.get("output_dir")
        if isinstance(output_dir, str) and output_dir.strip():
            return Path(output_dir.strip())
    dataset_name = run_config.get("dataset_name")
    data_count = run_config.get("data_count")
    timestamp = run_config.get("timestamp")
    if isinstance(dataset_name, str) and dataset_name and isinstance(data_count, int) and isinstance(timestamp, str):
        return Path("outputs") / dataset_name / f"{data_count}_{timestamp}"
    raise ValueError("Unable to resolve output_dir. Provide --output-dir or a usable run config.")


def load_metrics_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metrics.json must contain a JSON object: {path}")
    return payload


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("summarize_results")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def print_summary(summary_payload: dict[str, Any]) -> None:
    print(f"output_dir: {summary_payload.get('output_dir')}")
    print(f"num_samples: {summary_payload.get('num_samples')}")
    summary = summary_payload.get("summary")
    if not isinstance(summary, dict):
        return
    for metric_name, item in summary.items():
        if metric_name == "_failed_metrics" or not isinstance(item, dict):
            continue
        parts = [f"{metric_name}: status={item.get('status', 'unknown')}"]
        for key in (
            "pass_rate",
            "mean_view_consistency_score",
            "view_consistency_score",
            "mean_temporal_consistency_score",
            "mean_appearance_consistency_score",
            "mean_depth_consistency_score",
            "mean_semantic_consistency_score",
            "mean_instance_consistency_score",
            "valid_sample_count",
        ):
            if key in item:
                parts.append(f"{key}={item[key]}")
        reason = item.get("reason") or item.get("error")
        if reason:
            parts.append(f"reason={reason}")
        print(" ".join(parts))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_config = normalize_run_config(load_yaml(args.config))
    output_dir = resolve_output_dir(args, run_config)
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logger(logs_dir / "summarize.log")

    metrics_path = Path(args.metrics_path) if args.metrics_path else results_dir / "metrics.json"
    summary_path = Path(args.summary_path) if args.summary_path else results_dir / "summary.json"
    failed_samples_path = Path(args.failed_samples_path) if args.failed_samples_path else results_dir / "failed_samples.json"

    logger.info("Resolved output directory: %s", output_dir)
    logger.info("Resolved metrics path: %s", metrics_path)
    logger.info("Resolved summary path: %s", summary_path)
    logger.info("Resolved failed samples path: %s", failed_samples_path)

    metrics_payload = load_metrics_payload(metrics_path)
    metric_results = metrics_payload.get("results")
    if not isinstance(metric_results, dict):
        raise ValueError("metrics.json must contain a 'results' object.")

    summary_payload, _ = write_summary_result(
        metrics_result=metric_results,
        summary_path=summary_path,
        failed_samples_path=failed_samples_path,
        dataset_name=str(metrics_payload.get("dataset_name", run_config.get("dataset_name", ""))),
        data_count=int(metrics_payload.get("data_count", run_config.get("data_count", 0))),
        timestamp=str(metrics_payload.get("timestamp", run_config.get("timestamp", ""))),
        data_file=metrics_payload.get("data_file", ""),
        output_dir=output_dir,
    )

    logger.info("Wrote summary.json")
    logger.info("Wrote failed_samples.json")

    if args.print_summary:
        print_summary(summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
