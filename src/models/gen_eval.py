from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from models.executor import normalize_runtime_devices, run_evaluate_stage, run_prepare_reference_stage
from models.result import (
    ensure_output_layout,
    write_json,
    write_metrics_result,
    write_summary_result,
)

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_RUN_CONFIG_PATH = Path("configs/run.yaml")
DEFAULT_DATASET_CONFIG_PATH = Path("configs/dataset.yaml")
DEFAULT_METRICS_CONFIG_PATH = Path("configs/metrics.yaml")
DEFAULT_REFERENCE_CONFIG_PATH = Path("configs/reference.yaml")

DEFAULT_STAGES: dict[str, bool] = {
    "inspect_data": False,
    "prepare_reference": False,
    "evaluate": True,
    "summarize": True,
    "visualize": False,
}


class GenEval:
    def __init__(
        self,
        *,
        run_config_path: str | Path = DEFAULT_RUN_CONFIG_PATH,
        dataset_config_path: str | Path = DEFAULT_DATASET_CONFIG_PATH,
        metrics_config_path: str | Path = DEFAULT_METRICS_CONFIG_PATH,
        reference_config_path: str | Path = DEFAULT_REFERENCE_CONFIG_PATH,
    ) -> None:
        self.run_config_path = Path(run_config_path)
        self.dataset_config_path = Path(dataset_config_path)
        self.metrics_config_path = Path(metrics_config_path)
        self.reference_config_path = Path(reference_config_path)

        self.run_config = self._normalize_run_config(self._load_config(self.run_config_path))
        self.dataset_config = self._normalize_dataset_config(self._load_config(self.dataset_config_path))
        self.metrics_config = self._normalize_metrics_config(self._load_config(self.metrics_config_path))
        self.reference_config = self._load_config(self.reference_config_path)

        self.dataset_name = self._require_string(self.run_config, "dataset_name")
        self.data_count = self._require_int(self.run_config, "data_count")
        self.timestamp = self._require_string(self.run_config, "timestamp")
        self.data_file = self._resolve_data_file(self.run_config)
        self.output_dir = self._resolve_output_dir(self.run_config)
        self.output_paths = ensure_output_layout(self.output_dir)
        self.result_paths = self._build_result_paths(self.output_paths)
        self.logger = self._configure_logger(self.output_paths["logs_dir"] / "evaluate.log")

    def describe(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "data_count": self.data_count,
            "timestamp": self.timestamp,
            "profile": self.run_config.get("profile"),
            "data_file": str(self.data_file),
            "output_dir": str(self.output_dir),
            "config_paths": {
                "run": str(self.run_config_path),
                "dataset": str(self.dataset_config_path),
                "metrics": str(self.metrics_config_path),
                "reference": str(self.reference_config_path),
            },
            "result_paths": {key: str(value) for key, value in self.result_paths.items()},
            "effective_stages": self.effective_stages(),
        }

    def effective_stages(self, stage_overrides: dict[str, bool] | None = None) -> dict[str, bool]:
        configured = self.run_config.get("stages")
        stages = dict(DEFAULT_STAGES)
        if isinstance(configured, dict):
            for key, value in configured.items():
                stages[str(key)] = bool(value)
        if stage_overrides:
            for key, value in stage_overrides.items():
                stages[str(key)] = bool(value)
        return stages

    def prepare_reference(self) -> dict[str, Any]:
        reference_section = self.reference_config.get("reference")
        if isinstance(reference_section, dict) and not reference_section.get("enabled", True):
            return {"status": "skipped", "reason": "reference config is disabled"}

        started_at = time.perf_counter()
        runtime_config = self._runtime_config()
        backend = str(runtime_config.get("backend", "serial"))
        if backend == "ray":
            ray_address = runtime_config.get("ray_address", "auto")
            self.logger.info("Running prepare_reference with backend=%s address=%s", backend, ray_address)
        else:
            self.logger.info("Running prepare_reference with backend=%s", backend)
        summary = run_prepare_reference_stage(
            reference_config=self.reference_config,
            data_path=str(self.data_file),
            output_path=str(self.result_paths["enriched_data_path"]),
            summary_path=str(self.result_paths["reference_summary_path"]),
            output_dir=str(self.output_dir),
            runtime_config=runtime_config,
        )
        result = {
            "status": "success",
            "summary": summary,
            "duration_seconds": round(time.perf_counter() - started_at, 6),
        }
        if isinstance(summary, dict) and summary.get("wall_time_seconds") is not None:
            result["wall_time_seconds"] = summary.get("wall_time_seconds")
        return result

    def evaluate(self) -> dict[str, Any]:
        evaluate_started_at = time.perf_counter()
        data_path = self.result_paths["enriched_data_path"] if self.result_paths["enriched_data_path"].exists() else self.data_file
        dataset = self._build_dataset(data_path=data_path)
        inspection = dataset.inspect()
        if inspection.get("error"):
            raise ValueError(f"Dataset inspection failed: {inspection['error']}")

        samples = dataset.load_valid_samples()
        if not samples:
            raise ValueError("No valid samples were found for evaluation.")
        num_samples = len(samples)

        runtime_config = self._runtime_config()
        backend = str(runtime_config.get("backend", "serial"))
        if backend in {"local_multi_gpu", "ray"}:
            if backend == "local_multi_gpu":
                devices = normalize_runtime_devices(runtime_config)
                self.logger.info("Running evaluate with backend=%s devices=%s", backend, devices)
            else:
                ray_address = runtime_config.get("ray_address", "auto")
                self.logger.info("Running evaluate with backend=%s address=%s", backend, ray_address)
            metrics_result = run_evaluate_stage(
                samples=samples,
                metrics_config=self.metrics_config,
                runtime_config=runtime_config,
            )
            for module_name, module_result in metrics_result.items():
                wall_time_seconds = float(module_result.get("wall_time_seconds", module_result.get("duration_seconds", 0.0)) or 0.0)
                aggregated_compute_seconds = float(module_result.get("aggregated_compute_seconds", 0.0) or 0.0)
                self.logger.info(
                    "Module %s finished in %.3fs wall time (aggregated compute %.3fs)",
                    module_name,
                    wall_time_seconds,
                    aggregated_compute_seconds,
                )
        else:
            self.logger.info("Running evaluate with backend=%s", backend)
            modules = self._build_modules()
            metrics_result = {}
            for module_name, module in modules:
                module_started_at = time.perf_counter()
                try:
                    module_result = module.evaluate(samples)
                except Exception as exc:
                    module_result = {
                        "metric": module_name,
                        "status": "failed",
                        "num_samples": num_samples,
                        "error": f"{type(exc).__name__}: {exc}",
                        "details": {
                            "evaluated_samples": [],
                            "skipped_samples": [],
                            "failed_samples": [],
                        },
                    }
                module_duration_seconds = round(time.perf_counter() - module_started_at, 6)
                if isinstance(module_result, dict):
                    module_result["wall_time_seconds"] = module_duration_seconds
                    module_result["aggregated_compute_seconds"] = module_duration_seconds
                    module_result["duration_seconds"] = module_duration_seconds
                metrics_result[module_name] = module_result
                self.logger.info("Module %s finished in %.3fs", module_name, module_duration_seconds)

        payload = write_metrics_result(
            metrics_result=metrics_result,
            path=self.result_paths["metrics_path"],
            dataset_name=self.dataset_name,
            data_count=self.data_count,
            timestamp=self.timestamp,
            data_file=data_path,
            num_samples=num_samples,
        )
        payload["duration_seconds"] = round(time.perf_counter() - evaluate_started_at, 6)
        write_summary_result(
            metrics_result=metrics_result,
            summary_path=self.result_paths["summary_path"],
            failed_samples_path=self.result_paths["failed_samples_path"],
            dataset_name=self.dataset_name,
            data_count=self.data_count,
            timestamp=self.timestamp,
            data_file=data_path,
            output_dir=self.output_dir,
            num_samples=num_samples,
        )
        return payload

    def run(self, stage_overrides: dict[str, bool] | None = None) -> dict[str, Any]:
        stages = self.effective_stages(stage_overrides)
        stage_results: dict[str, Any] = {}

        self.logger.info("Resolved data file: %s", self.data_file)
        self.logger.info("Resolved output dir: %s", self.output_dir)
        self.logger.info("Effective stages: %s", stages)

        if stages.get("inspect_data", False):
            dataset = self._build_dataset(data_path=self.data_file)
            inspection = dataset.inspect()
            write_json(inspection, self.result_paths["inspection_path"])
            if inspection.get("error"):
                stage_results["inspect_data"] = {
                    "status": "failed",
                    "path": str(self.result_paths["inspection_path"]),
                    "reason": str(inspection["error"]),
                }
            else:
                stage_results["inspect_data"] = {"status": "success", "path": str(self.result_paths["inspection_path"])}

        if stages.get("prepare_reference", False):
            reference_result = self.prepare_reference()
            stage_results["prepare_reference"] = reference_result

        metrics_payload: dict[str, Any] | None = None
        if stages.get("evaluate", False):
            evaluate_started_at = time.perf_counter()
            metrics_payload = self.evaluate()
            evaluate_status = "success"
            evaluate_reason = None
            metric_results = metrics_payload.get("results") if isinstance(metrics_payload, dict) else None
            if isinstance(metric_results, dict):
                for item in metric_results.values():
                    if isinstance(item, dict) and item.get("status") == "failed":
                        evaluate_status = "failed"
                        evaluate_reason = "One or more metrics failed."
                        break
            stage_results["evaluate"] = {
                "status": evaluate_status,
                "path": str(self.result_paths["metrics_path"]),
                "duration_seconds": round(time.perf_counter() - evaluate_started_at, 6),
            }
            if evaluate_reason is not None:
                stage_results["evaluate"]["reason"] = evaluate_reason

        if stages.get("summarize", False):
            if self.result_paths["metrics_path"].exists():
                metrics_payload = self._load_json(self.result_paths["metrics_path"])
                metric_results = metrics_payload.get("results")
                if not isinstance(metric_results, dict):
                    raise ValueError("metrics.json must contain a 'results' object.")
                write_summary_result(
                    metrics_result=metric_results,
                    summary_path=self.result_paths["summary_path"],
                    failed_samples_path=self.result_paths["failed_samples_path"],
                    dataset_name=self.dataset_name,
                    data_count=self.data_count,
                    timestamp=self.timestamp,
                    data_file=metrics_payload.get("data_file", self.data_file),
                    output_dir=self.output_dir,
                    num_samples=metrics_payload.get("num_samples") if isinstance(metrics_payload.get("num_samples"), int) else None,
                )
                stage_results["summarize"] = {"status": "success", "path": str(self.result_paths["summary_path"])}
            else:
                stage_results["summarize"] = {"status": "skipped", "reason": "metrics.json does not exist"}

        if stages.get("visualize", False):
            stage_results["visualize"] = {
                "status": "skipped",
                "reason": "visualization is handled by scripts/visualize.py in a later step",
            }

        result = {
            "status": self._overall_status(stage_results, metrics_payload),
            "describe": self.describe(),
            "stages": stage_results,
        }
        if metrics_payload is not None:
            result["metrics"] = metrics_payload
        return result

    def _build_dataset(self, *, data_path: str | Path) -> Any:
        from dataset import build_dataset

        dataset_entry = self.dataset_config.get(self.dataset_name)
        if not isinstance(dataset_entry, dict):
            raise ValueError(f"Dataset config does not contain a valid entry for '{self.dataset_name}'.")
        config = dict(dataset_entry)
        config["data_file"] = str(data_path)
        return build_dataset(self.dataset_name, config)

    def _build_modules(self) -> list[tuple[str, Any]]:
        from modules import build_module

        modules: list[tuple[str, Any]] = []
        for module_name, module_config in self.metrics_config.items():
            if not isinstance(module_config, dict):
                continue
            if not module_config.get("enabled", False):
                continue
            modules.append((module_name, build_module(module_name, dict(module_config))))
        return modules

    def _build_result_paths(self, output_paths: dict[str, Path]) -> dict[str, Path]:
        results_dir = output_paths["results_dir"]
        return {
            "inspection_path": results_dir / "data_inspection.json",
            "enriched_data_path": results_dir / "enriched_data.json",
            "reference_summary_path": results_dir / "reference_summary.json",
            "metrics_path": results_dir / "metrics.json",
            "summary_path": results_dir / "summary.json",
            "failed_samples_path": results_dir / "failed_samples.json",
        }

    def _resolve_data_file(self, run_config: dict[str, Any]) -> Path:
        paths = run_config.get("paths")
        if isinstance(paths, dict):
            data_file = paths.get("data_file")
            if isinstance(data_file, str) and data_file.strip():
                return Path(data_file)
        return Path("data") / f"{self.dataset_name}_{self.data_count}_{self.timestamp}.json"

    def _resolve_output_dir(self, run_config: dict[str, Any]) -> Path:
        paths = run_config.get("paths")
        if isinstance(paths, dict):
            output_dir = paths.get("output_dir")
            if isinstance(output_dir, str) and output_dir.strip():
                return Path(output_dir)
        return Path("outputs") / self.dataset_name / f"{self.data_count}_{self.timestamp}"

    def _runtime_config(self) -> dict[str, Any]:
        runtime: dict[str, Any] = {}
        base_runtime = self.run_config.get("runtime")
        if isinstance(base_runtime, dict):
            runtime.update(base_runtime)

        profile_name = self.run_config.get("profile")
        profiles = self.run_config.get("profiles")
        if isinstance(profile_name, str) and profile_name and isinstance(profiles, dict):
            selected = profiles.get(profile_name)
            if isinstance(selected, dict):
                profile_runtime = selected.get("runtime")
                if isinstance(profile_runtime, dict):
                    runtime.update(profile_runtime)
        return runtime

    def _normalize_run_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_config = payload.get("run")
        if isinstance(run_config, dict):
            return dict(run_config)
        return dict(payload)

    def _normalize_dataset_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        datasets = payload.get("datasets")
        if isinstance(datasets, dict):
            return dict(datasets)
        return dict(payload)

    def _normalize_metrics_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        metrics = payload.get("metrics")
        metrics_payload = metrics if isinstance(metrics, dict) else payload
        normalized: dict[str, Any] = {}
        for metric_name, metric_config in metrics_payload.items():
            if not isinstance(metric_config, dict):
                normalized[str(metric_name)] = metric_config
                continue
            flat_config: dict[str, Any] = {}
            if "enabled" in metric_config:
                flat_config["enabled"] = bool(metric_config.get("enabled"))
            metric_section = metric_config.get("metric")
            if isinstance(metric_section, dict):
                flat_config.update(metric_section)
            model_section = metric_config.get("model")
            if isinstance(model_section, dict):
                flat_config.update(model_section)
            for key, value in metric_config.items():
                if key in {"enabled", "metric", "model"}:
                    continue
                flat_config[key] = value
            normalized[str(metric_name)] = flat_config
        return normalized

    def _load_config(self, path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        if config_path.suffix.lower() == ".json":
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load worldbench config files.")
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config file must load as an object: {config_path}")
        return payload

    def _load_json(self, path: str | Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON file must contain an object: {path}")
        return payload

    def _configure_logger(self, log_path: Path) -> logging.Logger:
        logger = logging.getLogger("run_eval")
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

    def _overall_status(self, stage_results: dict[str, Any], metrics_payload: dict[str, Any] | None) -> str:
        for result in stage_results.values():
            if isinstance(result, dict) and result.get("status") == "failed":
                return "failed"
        if metrics_payload is not None:
            metric_results = metrics_payload.get("results")
            if isinstance(metric_results, dict):
                for item in metric_results.values():
                    if isinstance(item, dict) and item.get("status") == "failed":
                        return "failed"
        return "success"

    @staticmethod
    def _require_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Run config field '{key}' must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _require_int(payload: dict[str, Any], key: str) -> int:
        value = payload.get(key)
        if not isinstance(value, int):
            raise ValueError(f"Run config field '{key}' must be an integer.")
        return value
