from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gen_eval.schemas import GenerationSample


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a simple GEN_EVAL manifest.")
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Generated video path. Reuse as sample_id stem when not otherwise available.",
    )
    parser.add_argument("--output", required=True, help="Manifest JSON output path.")
    args = parser.parse_args()

    samples = []
    for sample_path in args.sample:
        path = Path(sample_path)
        sample = GenerationSample(
            sample_id=path.stem,
            generated_video=str(path),
            reference_video=None,
            prompt="",
            objects=[],
            metadata={},
        )
        samples.append(sample.to_dict())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
