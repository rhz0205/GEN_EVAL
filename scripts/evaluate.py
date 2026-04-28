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
from gen_eval.result_writer import (
    build_result_payload,
    get_command_string,
    save_result_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GEN_EVAL evaluation.")
    parser.add_argument("--config", required=True, help="Path to run config YAML file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output JSON path. If not set, save under output_dir.",
    )
    args = parser.parse_args()

    # 1. 导入并解析配置文件
    resolved_config = resolve_run_config(args.config)
    # 2. 执行评估并获取结果
    results = run_evaluation(resolved_config)

    # 3. 构建结果负载，记录配置、指标输出和运行命令等信息
    payload = build_result_payload(
        resolved_config,
        results,
        command=get_command_string(),
    )

    # 4. 保存结果到指定目录，并获取输出路径
    output_paths = save_result_bundle(
        payload=payload,
        output_dir=str(resolved_config.get("output_dir") or "outputs"),
        explicit_output=args.output,
    )

    # 5. 打印评估摘要信息
    print_evaluation_summary(payload, output_paths["result_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
