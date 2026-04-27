from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gen_eval.evaluator import Evaluator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GEN_EVAL metrics from config.")
    parser.add_argument("--config", required=True, help="Path to evaluation config.")
    args = parser.parse_args()

    evaluator = Evaluator.from_config_path(args.config)
    results = evaluator.run()
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
