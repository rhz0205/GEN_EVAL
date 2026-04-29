from __future__ import annotations

import importlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from reference.base import ReferenceGenerator

_REFERENCE_SPECS: dict[str, tuple[str, str]] = {
    "openseed_semantic": ("reference.openseed", "OpenSeeDReference"),
    "depth_reference": ("reference.depth", "DepthReference"),
    "object_tracks": ("reference.tracking", "ObjectTrackReference"),
    "planning_response": ("reference.planning", "PlanningResponseReference"),
}

REFERENCE_ALIASES: dict[str, str] = {
    "openseed_semantics": "openseed_semantic",
}

_REFERENCE_CLASS_CACHE: dict[str, type[ReferenceGenerator]] = {}
PROTECTED_METADATA_KEYS: tuple[str, ...] = ("camera_videos",)


def _load_reference_class(module_path: str, class_name: str) -> type[ReferenceGenerator]:
    cache_key = f"{module_path}:{class_name}"
    cached = _REFERENCE_CLASS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    module = importlib.import_module(module_path)
    loaded = getattr(module, class_name)
    _REFERENCE_CLASS_CACHE[cache_key] = loaded
    return loaded


REFERENCE_REGISTRY: dict[str, tuple[str, str]] = dict(_REFERENCE_SPECS)


def build_reference_generator(name: str, config: dict[str, Any] | None = None) -> ReferenceGenerator:
    canonical_name = REFERENCE_ALIASES.get(name, name)
    spec = REFERENCE_REGISTRY.get(canonical_name)
    if spec is None:
        available = ", ".join(sorted(REFERENCE_REGISTRY))
        raise ValueError(f"Unknown reference generator '{name}'. Available generators: {available}")
    module_path, class_name = spec
    generator_cls = _load_reference_class(module_path, class_name)
    normalized_config = dict(config or {})
    normalized_config.setdefault("name", canonical_name)
    return generator_cls(normalized_config)


def load_json(path: str | Path) -> Any:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def extract_samples(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return payload["samples"]
    raise ValueError("Input data must be a top-level list or an object containing a 'samples' list.")


def set_samples(payload: Any, samples: list[dict[str, Any]]) -> Any:
    if isinstance(payload, list):
        return samples
    if isinstance(payload, dict):
        enriched = deepcopy(payload)
        enriched["samples"] = samples
        return enriched
    raise ValueError("Input data must be a top-level list or an object containing a 'samples' list.")


def merge_metadata(metadata: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(metadata)
    for key, value in patch.items():
        if key in PROTECTED_METADATA_KEYS:
            raise ValueError(f"Reference generators must not overwrite protected metadata key: {key}")
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def infer_run_context(data_path: str | Path) -> dict[str, Any]:
    path = Path(data_path)
    match = re.fullmatch(r"(?P<dataset_name>[a-zA-Z0-9]+)_(?P<data_count>\d+)_(?P<timestamp>\d+)", path.stem)
    if not match:
        return {
            "dataset_name": None,
            "data_count": None,
            "timestamp": None,
        }
    groups = match.groupdict()
    return {
        "dataset_name": groups["dataset_name"],
        "data_count": int(groups["data_count"]),
        "timestamp": groups["timestamp"],
    }


class ReferencePreparer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.reference_config = self._resolve_reference_config(self.config)
        self.enabled = bool(self.reference_config.get("enabled", True))
        self.continue_on_error = bool(self.reference_config.get("continue_on_error", False))
        self.generators = self._build_generators(self.config)

    def prepare(
        self,
        *,
        data_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        output_path: str | Path,
        summary_path: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        source_path = data_path or manifest_path
        if source_path is None:
            raise ValueError("ReferencePreparer requires data_path or manifest_path.")
        if not self.enabled:
            raise ValueError("Reference preparation is disabled in config.")

        raw_payload = load_json(source_path)
        samples = extract_samples(raw_payload)
        reference_root = Path(output_dir)
        reference_root.mkdir(parents=True, exist_ok=True)
        run_context = infer_run_context(source_path)

        enriched_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        generator_summaries: dict[str, dict[str, int]] = {
            generator.name: {"prepared": 0, "failed": 0} for generator in self.generators
        }

        for sample in samples:
            enriched_sample = deepcopy(sample)
            raw_metadata = enriched_sample.get("metadata")
            if raw_metadata is None:
                metadata = {}
            elif isinstance(raw_metadata, dict):
                metadata = dict(raw_metadata)
            else:
                sample_id = str(enriched_sample.get("sample_id") or "unknown")
                raise ValueError(f"sample {sample_id} metadata must be an object.")
            enriched_sample["metadata"] = metadata
            sample_id = str(enriched_sample.get("sample_id") or "unknown")
            sample_failures: list[dict[str, str]] = []

            for generator in self.generators:
                try:
                    patch = generator.prepare_sample(enriched_sample, reference_root)
                    metadata = merge_metadata(metadata, patch)
                    enriched_sample["metadata"] = metadata
                    generator_summaries[generator.name]["prepared"] += 1
                except Exception as exc:
                    generator_summaries[generator.name]["failed"] += 1
                    failure = {
                        "generator": generator.name,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    sample_failures.append(failure)
                    if not self.continue_on_error:
                        raise RuntimeError(
                            f"Reference generation failed for sample={sample_id}, generator={generator.name}: {exc}"
                        ) from exc

            if sample_failures:
                failed_samples.append({"sample_id": sample_id, "errors": sample_failures})
            enriched_samples.append(enriched_sample)

        enriched_payload = set_samples(raw_payload, enriched_samples)
        write_json(output_path, enriched_payload)
        summary = {
            "status": "success",
            "dataset_name": run_context["dataset_name"],
            "data_count": run_context["data_count"],
            "timestamp": run_context["timestamp"],
            "data_file": str(Path(source_path)),
            "output_dir": str(Path(output_dir)),
            "num_samples": len(samples),
            "num_generators": len(self.generators),
            "enriched_data_path": str(Path(output_path)),
            "reference_output_dir": str(reference_root),
            "generator_summary": generator_summaries,
            "failed_samples": failed_samples,
            "continue_on_error": self.continue_on_error,
        }
        write_json(summary_path, summary)
        return summary

    def _resolve_reference_config(self, config: dict[str, Any]) -> dict[str, Any]:
        nested = config.get("reference")
        if isinstance(nested, dict):
            return nested
        return config

    def _build_generators(self, config: dict[str, Any]) -> list[ReferenceGenerator]:
        nested = self._resolve_reference_config(config)
        generators: list[ReferenceGenerator] = []

        generator_map = nested.get("generators")
        if isinstance(generator_map, dict):
            for name, raw_generator_config in generator_map.items():
                generator_config = dict(raw_generator_config or {})
                if not generator_config.get("enabled", True):
                    continue
                generators.append(build_reference_generator(str(name), generator_config))

        legacy_generators = config.get("reference_generators")
        if isinstance(legacy_generators, list):
            for raw_generator_config in legacy_generators:
                if not isinstance(raw_generator_config, dict):
                    continue
                if not raw_generator_config.get("enabled", True):
                    continue
                name = raw_generator_config.get("name")
                if not name:
                    raise ValueError("Each legacy reference generator config must contain 'name'.")
                generators.append(build_reference_generator(str(name), raw_generator_config))

        deduped: dict[str, ReferenceGenerator] = {}
        for generator in generators:
            deduped[generator.name] = generator

        return list(deduped.values())
