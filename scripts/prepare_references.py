from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from gen_eval.reference import ReferencePreparer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reference files from a GEN_EVAL manifest and write an "
            "enriched manifest for downstream metric evaluation."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Input manifest path. The manifest can be a list or a dict with samples.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Reference generation config path, for example configs/reference.yaml.",
    )
    parser.add_argument(
        "--output_manifest",
        required=True,
        help="Output enriched manifest path.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for generated reference files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_yaml(args.config)
    preparer = ReferencePreparer(config)
    preparer.prepare_manifest(
        manifest_path=args.manifest,
        output_manifest_path=args.output_manifest,
        output_dir=args.output_dir,
    )

    print(f"Reference files written to: {args.output_dir}")
    print(f"Enriched manifest written to: {args.output_manifest}")
    return 0


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Reference config must be a YAML dict: {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
