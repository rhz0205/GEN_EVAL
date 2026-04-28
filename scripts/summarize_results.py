#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gen_eval.result_summary import summarize_result_file

def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a GEN_EVAL result JSON.")
    parser.add_argument("--result", required=True, help="Path to saved result JSON.")
    args = parser.parse_args()

    for line in summarize_result_file(args.result):
        print(line)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
