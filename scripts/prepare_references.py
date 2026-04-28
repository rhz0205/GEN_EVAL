from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reference import ReferencePreparer

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_REFERENCE_CONFIG = Path("configs/reference.yaml")
DEFAULT_RUN_CONFIG = Path("configs/run.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare reference data and write enriched data outputs.")
    parser.add_argument("--config", default=str(DEFAULT_REFERENCE_CONFIG), help="Path to reference config YAML.")
    parser.add_argument("--run-config", default=str(DEFAULT_RUN_CONFIG), help="Path to run config YAML.")
    parser.add_argument("--data-path", default=None, help="Input data JSON path override.")
    parser.add_argument("--manifest", default=None, help="Alias for --data-path.")
    parser.add_argument("--output-path", default=None, help="Enriched data output path override.")
    parser.add_argument("--output-dir", default=None, help="Output root override.")
    parser.add_argument("--summary-path", default=None, help="Reference summary path override.")
    return parser


def require_yaml() -> None:
    if yaml is None:
        print("PyYAML is required to load YAML config files for scripts/prepare_references.py.", file=sys.stderr)
        raise SystemExit(2)


def load_yaml(path: str | Path) -> dict[str, Any]:
    require_yaml()
    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML config file does not exist: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must load as an object: {yaml_path}")
    return payload


def get_run_config(payload: dict[str, Any]) -> dict[str, Any]:
    run_config = payload.get("run")
    if isinstance(run_config, dict):
        return run_config
    return payload


def require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field '{key}' must be a non-empty string.")
    return value.strip()


def require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Config field '{key}' must be an integer.")
    return value


def resolve_input_data(args: argparse.Namespace, run_config: dict[str, Any]) -> Path:
    if args.data_path:
        return Path(args.data_path)
    if args.manifest:
        return Path(args.manifest)
    paths = run_config.get("paths")
    if isinstance(paths, dict):
        data_file = paths.get("data_file")
        if isinstance(data_file, str) and data_file.strip():
            return Path(data_file)
    dataset_name = require_string(run_config, "dataset_name")
    data_count = require_int(run_config, "data_count")
    timestamp = require_string(run_config, "timestamp")
    return Path("data") / f"{dataset_name}_{data_count}_{timestamp}.json"


def resolve_output_root(args: argparse.Namespace, run_config: dict[str, Any]) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    paths = run_config.get("paths")
    if isinstance(paths, dict):
        output_dir = paths.get("output_dir")
        if isinstance(output_dir, str) and output_dir.strip():
            return Path(output_dir)
    dataset_name = require_string(run_config, "dataset_name")
    data_count = require_int(run_config, "data_count")
    timestamp = require_string(run_config, "timestamp")
    return Path("outputs") / dataset_name / f"{data_count}_{timestamp}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_payload = load_yaml(args.run_config)
    run_config = get_run_config(run_payload)
    reference_config = load_yaml(args.config)

    data_path = resolve_input_data(args, run_config)
    output_root = resolve_output_root(args, run_config)
    output_path = Path(args.output_path) if args.output_path else output_root / "results" / "enriched_data.json"
    summary_path = Path(args.summary_path) if args.summary_path else output_root / "results" / "reference_summary.json"

    preparer = ReferencePreparer(reference_config)
    summary = preparer.prepare(
        data_path=data_path,
        output_path=output_path,
        summary_path=summary_path,
        output_dir=output_root,
    )

    print(f"samples={summary['num_samples']} generators={summary['num_generators']} failed={len(summary['failed_samples'])}")
    print(f"enriched_data={output_path}")
    print(f"reference_summary={summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"FileNotFoundError: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except ValueError as exc:
        print(f"ValueError: {exc}", file=sys.stderr)
        raise SystemExit(1)
