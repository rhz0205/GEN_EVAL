from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models import GenEval


DEFAULT_RUN_CONFIG = Path("configs/run.yaml")
DEFAULT_DATASET_CONFIG = Path("configs/dataset.yaml")
DEFAULT_METRICS_CONFIG = Path("configs/metrics.yaml")
DEFAULT_REFERENCE_CONFIG = Path("configs/reference.yaml")
VALID_STAGES = ("inspect_data", "prepare_reference", "evaluate", "summarize", "visualize", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run worldbench evaluation.")
    parser.add_argument("--config", default=str(DEFAULT_RUN_CONFIG), help="Path to run config YAML file.")
    parser.add_argument("--dataset-config", default=str(DEFAULT_DATASET_CONFIG), help="Path to dataset config YAML file.")
    parser.add_argument("--metrics-config", default=str(DEFAULT_METRICS_CONFIG), help="Path to metrics config YAML file.")
    parser.add_argument("--reference-config", default=str(DEFAULT_REFERENCE_CONFIG), help="Path to reference config YAML file.")
    parser.add_argument("--profile", default=None, help="Optional run profile override, for example debug or eval.")
    parser.add_argument("--stage", action="append", default=None, help="Optional repeated or comma-separated stage override.")
    parser.add_argument("--skip-reference", action="store_true", help="Force prepare_reference=false for this run.")
    parser.add_argument("--no-visualize", action="store_true", help="Force visualize=false for this run.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and stages without running.")
    parser.add_argument("--print-config", action="store_true", help="Print resolved description payload.")
    return parser


def parse_stage_overrides(raw_values: list[str] | None) -> dict[str, bool] | None:
    if not raw_values:
        return None
    names: list[str] = []
    for raw_value in raw_values:
        for item in str(raw_value).split(","):
            stage = item.strip()
            if stage:
                names.append(stage)
    invalid = [name for name in names if name not in VALID_STAGES]
    if invalid:
        raise ValueError(f"Unsupported stage values: {', '.join(invalid)}")
    if "all" in names:
        overrides = {stage: True for stage in VALID_STAGES if stage != "all"}
    else:
        overrides = {stage: False for stage in VALID_STAGES if stage != "all"}
        for name in names:
            overrides[name] = True
    return overrides


def print_terminal_summary(result: dict[str, object]) -> None:
    print(f"status={result.get('status')}")
    stages = result.get("stages")
    if isinstance(stages, dict):
        for stage_name, stage_result in stages.items():
            if not isinstance(stage_result, dict):
                continue
            line = [f"{stage_name}: status={stage_result.get('status', 'unknown')}"]
            reason = stage_result.get("reason")
            if reason:
                line.append(f"reason={reason}")
            print(" ".join(line))
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        metric_results = metrics.get("results")
        if isinstance(metric_results, dict):
            for metric_name, metric_result in metric_results.items():
                if not isinstance(metric_result, dict):
                    continue
                line = [f"{metric_name}: status={metric_result.get('status', 'unknown')}"]
                for key in (
                    "pass_rate",
                    "mean_view_consistency_score",
                    "view_consistency_score",
                    "mean_temporal_consistency_score",
                    "mean_instance_coherence_score",
                    "mean_depth_consistency_score",
                    "mean_semantic_consistency_score",
                    "mean_instance_consistency_score",
                    "valid_sample_count",
                ):
                    if key in metric_result:
                        line.append(f"{key}={metric_result.get(key)}")
                reason = metric_result.get("reason") or metric_result.get("error")
                if reason:
                    line.append(f"reason={reason}")
                print(" ".join(line))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    evaluator = GenEval(
        run_config_path=args.config,
        dataset_config_path=args.dataset_config,
        metrics_config_path=args.metrics_config,
        reference_config_path=args.reference_config,
    )
    if args.profile:
        evaluator.run_config["profile"] = str(args.profile)

    stage_overrides = parse_stage_overrides(args.stage)
    if stage_overrides is None:
        stage_overrides = {}
    if args.skip_reference:
        stage_overrides["prepare_reference"] = False
    if args.no_visualize:
        stage_overrides["visualize"] = False

    description = evaluator.describe()
    description["effective_stages"] = evaluator.effective_stages(stage_overrides or None)

    if args.print_config or args.dry_run:
        print(json.dumps(description, indent=2, ensure_ascii=False))

    if args.dry_run:
        return 0

    try:
        result = evaluator.run(stage_overrides=stage_overrides or None)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print_terminal_summary(result)
    if result.get("status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
