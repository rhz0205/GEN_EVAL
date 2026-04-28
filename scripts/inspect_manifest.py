#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gen_eval.dataset import format_manifest_summary

def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a GEN_EVAL manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file.")
    args = parser.parse_args()

    lines = format_manifest_summary(args.manifest)
    for line in lines:
        print(line)

    manifest_path = Path(args.manifest)
    return 0 if manifest_path.exists() else 1

if __name__ == "__main__":
    raise SystemExit(main())
