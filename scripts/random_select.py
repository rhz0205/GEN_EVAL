from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import DEFAULT_CAMERA_VIEWS, build_dataset

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG_PATH = Path("configs/run.yaml")
DEFAULT_DATASET_CONFIG_PATH = Path("configs/dataset.yaml")
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_SEED = 42


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a worldbench data file and select random valid samples.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the run config YAML file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of valid samples to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic selection.",
    )
    parser.add_argument(
        "--check-paths",
        dest="check_paths",
        action="store_true",
        default=True,
        help="Check whether expected camera video paths exist.",
    )
    parser.add_argument(
        "--no-check-paths",
        dest="check_paths",
        action="store_false",
        help="Skip filesystem checks for camera video paths.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any invalid sample exists.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory from the run config.",
    )
    return parser


def require_yaml() -> None:
    if yaml is None:
        print(
            "PyYAML is required to load YAML config files for scripts/random_select.py.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def load_yaml(path: Path) -> dict[str, Any]:
    require_yaml()
    if not path.is_file():
        raise FileNotFoundError(f"YAML config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must load as an object: {path}")
    return payload


def get_run_config(payload: dict[str, Any]) -> dict[str, Any]:
    run_config = payload.get("run")
    if isinstance(run_config, dict):
        return run_config
    return payload


def get_datasets_config(payload: dict[str, Any]) -> dict[str, Any]:
    datasets = payload.get("datasets")
    if isinstance(datasets, dict):
        return datasets
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


def resolve_data_file(run_config: dict[str, Any]) -> Path:
    paths = run_config.get("paths")
    if isinstance(paths, dict):
        data_file = paths.get("data_file")
        if isinstance(data_file, str) and data_file.strip():
            return Path(data_file)
    dataset_name = require_string(run_config, "dataset_name")
    data_count = require_int(run_config, "data_count")
    timestamp = require_string(run_config, "timestamp")
    return Path("data") / f"{dataset_name}_{data_count}_{timestamp}.json"


def resolve_output_dir(run_config: dict[str, Any], output_dir_override: str | None) -> Path:
    if output_dir_override:
        return Path(output_dir_override)
    paths = run_config.get("paths")
    if isinstance(paths, dict):
        output_dir = paths.get("output_dir")
        if isinstance(output_dir, str) and output_dir.strip():
            return Path(output_dir)
    dataset_name = require_string(run_config, "dataset_name")
    data_count = require_int(run_config, "data_count")
    timestamp = require_string(run_config, "timestamp")
    return Path("outputs") / dataset_name / f"{data_count}_{timestamp}"


def resolve_sample_size(args: argparse.Namespace, run_config: dict[str, Any]) -> int:
    if args.sample_size is not None:
        return max(0, args.sample_size)
    value = run_config.get("sample_size")
    if isinstance(value, int):
        return max(0, value)
    selection = run_config.get("selection")
    if isinstance(selection, dict) and isinstance(selection.get("sample_size"), int):
        return max(0, selection["sample_size"])
    return DEFAULT_SAMPLE_SIZE


def resolve_seed(args: argparse.Namespace, run_config: dict[str, Any]) -> int:
    if args.seed is not None:
        return args.seed
    value = run_config.get("seed")
    if isinstance(value, int):
        return value
    selection = run_config.get("selection")
    if isinstance(selection, dict) and isinstance(selection.get("seed"), int):
        return selection["seed"]
    return DEFAULT_SEED


def load_dataset_entry(dataset_name: str) -> dict[str, Any]:
    payload = load_yaml(DEFAULT_DATASET_CONFIG_PATH)
    datasets = get_datasets_config(payload)
    entry = datasets.get(dataset_name)
    if not isinstance(entry, dict):
        raise ValueError(f"Dataset config does not contain a valid entry for '{dataset_name}'.")
    return dict(entry)


def ensure_output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, logs_dir, logs_dir / "random_select.log"


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("random_select")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def select_samples(valid_samples: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if not valid_samples:
        return []
    selected_count = min(max(0, sample_size), len(valid_samples))
    if selected_count >= len(valid_samples):
        return sorted(valid_samples, key=lambda item: item["sample_id"])
    rng = random.Random(seed)
    selected = rng.sample(valid_samples, selected_count)
    return sorted(selected, key=lambda item: item["sample_id"])


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_payload = load_yaml(Path(args.config))
    run_config = get_run_config(run_payload)

    dataset_name = require_string(run_config, "dataset_name")
    data_count = require_int(run_config, "data_count")
    timestamp = require_string(run_config, "timestamp")
    data_file = resolve_data_file(run_config)
    output_dir = resolve_output_dir(run_config, args.output_dir)
    sample_size = resolve_sample_size(args, run_config)
    seed = resolve_seed(args, run_config)

    dataset_entry = load_dataset_entry(dataset_name)
    dataset_entry["data_file"] = str(data_file)
    dataset = build_dataset(dataset_name, dataset_entry)

    results_dir, _, log_path = ensure_output_paths(output_dir)
    logger = configure_logger(log_path)

    logger.info("Resolved data file path: %s", data_file)
    logger.info("Resolved output directory: %s", output_dir)
    logger.info("Path checking enabled: %s", args.check_paths)

    inspection = dataset.inspect(check_paths=args.check_paths)
    valid_samples = [sample.to_dict() for sample in dataset.load_valid_samples(check_paths=args.check_paths)]
    selected_samples = select_samples(valid_samples, sample_size, seed)

    inspection_payload = {
        "dataset_name": dataset_name,
        "data_count": data_count,
        "timestamp": timestamp,
        "data_file": str(data_file),
        "num_samples": inspection["num_samples"],
        "num_valid_samples": inspection["num_valid_samples"],
        "num_invalid_samples": inspection["num_invalid_samples"],
        "check_paths": bool(args.check_paths),
        "expected_camera_views": list(
            dataset.expected_camera_views if dataset.expected_camera_views else DEFAULT_CAMERA_VIEWS
        ),
        "invalid_samples": inspection["invalid_samples"],
    }
    selected_payload = {
        "dataset_name": dataset_name,
        "data_count": data_count,
        "timestamp": timestamp,
        "seed": seed,
        "sample_size": sample_size,
        "selected_count": len(selected_samples),
        "selected_samples": selected_samples,
    }

    write_json(results_dir / "data_inspection.json", inspection_payload)
    write_json(results_dir / "selected_samples.json", selected_payload)

    logger.info("Number of samples: %s", inspection["num_samples"])
    logger.info("Valid sample count: %s", inspection["num_valid_samples"])
    logger.info("Invalid sample count: %s", inspection["num_invalid_samples"])
    logger.info("Selected sample count: %s", len(selected_samples))

    if inspection["num_valid_samples"] == 0:
        logger.error("No valid samples were found.")
        return 1
    if args.strict and inspection["num_invalid_samples"] > 0:
        logger.error("Strict mode is enabled and invalid samples were found.")
        return 1
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
