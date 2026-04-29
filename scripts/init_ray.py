from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import ray
except ImportError:
    ray = None

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_RUN_CONFIG = Path("configs/run.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize a Ray runtime from worldbench run config.")
    parser.add_argument("--config", default=str(DEFAULT_RUN_CONFIG), help="Path to run config YAML file.")
    parser.add_argument("--profile", default=None, help="Optional run profile override, for example debug or eval.")
    parser.add_argument("--address", default=None, help="Optional Ray address override, for example auto or 10.0.0.1:6379.")
    parser.add_argument("--namespace", default=None, help="Optional Ray namespace override.")
    parser.add_argument("--shutdown", action="store_true", help="Shutdown Ray after initialization check.")
    parser.add_argument("--print-config", action="store_true", help="Print resolved Ray runtime config.")
    return parser


def require_yaml() -> None:
    if yaml is None:
        print("PyYAML is required to load YAML config files for scripts/init_ray.py.", file=sys.stderr)
        raise SystemExit(2)


def require_ray() -> None:
    if ray is None:
        print("ray is required to run scripts/init_ray.py.", file=sys.stderr)
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
        return dict(run_config)
    return dict(payload)


def resolve_runtime_config(run_config: dict[str, Any]) -> dict[str, Any]:
    runtime: dict[str, Any] = {}
    base_runtime = run_config.get("runtime")
    if isinstance(base_runtime, dict):
        runtime.update(base_runtime)

    profile_name = run_config.get("profile")
    profiles = run_config.get("profiles")
    if isinstance(profile_name, str) and profile_name and isinstance(profiles, dict):
        selected = profiles.get(profile_name)
        if isinstance(selected, dict):
            profile_runtime = selected.get("runtime")
            if isinstance(profile_runtime, dict):
                runtime.update(profile_runtime)
    return runtime


def build_init_kwargs(runtime_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    address = args.address or runtime_config.get("ray_address", "auto")
    init_kwargs: dict[str, Any] = {
        "address": address,
        "ignore_reinit_error": True,
        "log_to_driver": True,
    }
    if args.namespace:
        init_kwargs["namespace"] = str(args.namespace)
    return init_kwargs


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    require_ray()
    run_payload = load_yaml(args.config)
    run_config = get_run_config(run_payload)
    if args.profile:
        run_config["profile"] = str(args.profile)
    runtime_config = resolve_runtime_config(run_config)
    init_kwargs = build_init_kwargs(runtime_config, args)

    if str(runtime_config.get("backend", "serial")) != "ray":
        runtime_config["backend"] = "ray"

    if args.print_config:
        print(
            json.dumps(
                {
                    "profile": run_config.get("profile"),
                    "runtime": runtime_config,
                    "ray_init": init_kwargs,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    if not ray.is_initialized():
        ray.init(**init_kwargs)

    resources = ray.cluster_resources()
    available_resources = ray.available_resources()
    node_count = len(ray.nodes())
    print(f"status=connected address={init_kwargs['address']}")
    print(f"nodes={node_count}")
    print(f"cluster_resources={json.dumps(resources, ensure_ascii=False, sort_keys=True)}")
    print(f"available_resources={json.dumps(available_resources, ensure_ascii=False, sort_keys=True)}")

    if args.shutdown and ray.is_initialized():
        ray.shutdown()
        print("status=shutdown")

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
