#!/usr/bin/env python3
"""Run GEN_EVAL evaluation from a run config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gen_eval.config import resolve_run_config
from gen_eval.execution import run_evaluation
from gen_eval.result_summary import print_evaluation_summary
from gen_eval.result_writer import build_result_payload, save_result_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GEN_EVAL evaluation.")
    parser.add_argument("--config", required=True, help="Path to run config YAML file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output JSON path. If not set, save under output_dir.",
    )
    args = parser.parse_args()

    resolved_config = resolve_run_config(args.config)
    results = run_evaluation(resolved_config)

    payload = build_result_payload(resolved_config, results)

    output_path = save_result_payload(
        payload=payload,
        output_dir=str(resolved_config.get("output_dir") or "outputs"),
        config_path=args.config,
        explicit_output=args.output,
    )
    print_evaluation_summary(payload, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
