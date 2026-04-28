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

from models.result import write_json
from visualization import VISUALIZER_REGISTRY, build_visualizer
from visualization.composer import ensure_visualization_layout

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG_PATH = Path("configs/run.yaml")
VALID_TARGETS = ("depth", "semantic", "multiview", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize worldbench results.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to run config.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--target", action="append", default=None, help="Optional repeated or comma-separated visualization targets.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and targets without rendering.")
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


def resolve_output_dir(run_config: dict[str, Any], output_dir_override: str | None) -> Path:
    if output_dir_override:
        return Path(output_dir_override)
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


def parse_targets(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return ["depth", "semantic", "multiview"]
    names: list[str] = []
    for raw_value in raw_values:
        for item in str(raw_value).split(","):
            target = item.strip()
            if target:
                names.append(target)
    invalid = [name for name in names if name not in VALID_TARGETS]
    if invalid:
        raise ValueError(f"Unsupported target values: {', '.join(invalid)}")
    if "all" in names:
        return ["depth", "semantic", "multiview"]
    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("visualize_results")
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


def summarize_status(target_results: dict[str, dict[str, Any]]) -> str:
    statuses = {result.get("status", "skipped") for result in target_results.values()}
    if "failed" in statuses:
        return "failed"
    if "success" in statuses and "skipped" in statuses:
        return "partial"
    if "partial" in statuses:
        return "partial"
    if "success" in statuses:
        return "success"
    return "skipped"


def print_summary(payload: dict[str, Any]) -> None:
    print(f"status: {payload.get('status')}")
    print(f"output_dir: {payload.get('output_dir')}")
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        return
    for name, result in targets.items():
        if not isinstance(result, dict):
            continue
        parts = [f"{name}: status={result.get('status', 'unknown')}"]
        reason = result.get("reason")
        if reason:
            parts.append(f"reason={reason}")
        print(" ".join(parts))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_config = normalize_run_config(load_yaml(args.config))
    output_dir = resolve_output_dir(run_config, args.output_dir)
    targets = parse_targets(args.target)
    layout = ensure_visualization_layout(output_dir)

    if args.dry_run:
        payload = {
            "status": "skipped",
            "output_dir": str(output_dir),
            "targets": targets,
            "layout": {key: str(value) for key, value in layout.items()},
        }
        if args.print_summary:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    logs_dir = output_dir / "logs"
    results_dir = output_dir / "results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(logs_dir / "visualize.log")

    target_results: dict[str, dict[str, Any]] = {}
    input_mapping = {
        "depth": layout["depth_raw"],
        "semantic": layout["semantic_raw"],
        "multiview": layout["multiview_match_raw"],
    }

    for target in targets:
        visualizer = build_visualizer(target, {})
        input_dir = input_mapping[target]
        logger.info("Rendering target=%s input_dir=%s output_dir=%s", target, input_dir, output_dir)
        try:
            target_results[target] = visualizer.render(input_dir, output_dir)
        except Exception as exc:
            target_results[target] = {
                "name": target,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
            }

    payload = {
        "status": summarize_status(target_results),
        "output_dir": str(output_dir),
        "targets": target_results,
    }
    summary_path = results_dir / "visualization_summary.json"
    write_json(payload, summary_path)
    logger.info("Wrote visualization summary: %s", summary_path)

    if args.print_summary:
        print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
