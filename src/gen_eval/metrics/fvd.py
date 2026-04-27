from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gen_eval.schemas import GenerationSample


class FVDMetric:
    name = "fvd"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.feature_mode = str(config.get("feature_mode", "precomputed"))
        self.generated_feature_key = str(
            config.get("generated_feature_key", "generated_fvd_features")
        )
        self.reference_feature_key = str(
            config.get("reference_feature_key", "reference_fvd_features")
        )
        self.generated_stats_key = str(config.get("generated_stats_key", "generated_fvd_stats"))
        self.reference_stats_key = str(config.get("reference_stats_key", "reference_fvd_stats"))

    def evaluate(self, samples: list[GenerationSample]) -> dict[str, Any]:
        paired_samples = []
        skipped_samples = []
        failed_samples = []

        for sample in samples:
            if not sample.reference_video:
                skipped_samples.append(
                    {
                        "sample_id": sample.sample_id,
                        "reason": "FVD requires reference_video for paired evaluation.",
                    }
                )
                continue
            paired_samples.append(sample)

        if self.feature_mode == "runtime":
            return self._result(
                score=None,
                num_samples=0,
                details={
                    "evaluated_samples": [],
                    "skipped_samples": skipped_samples,
                    "failed_samples": failed_samples,
                },
                status="skipped",
                reason=(
                    "Runtime FVD feature extraction is not implemented in this lightweight migration. "
                    "Provide precomputed features or feature statistics through config or sample metadata."
                ),
            )

        try:
            config_stats = self._load_stats_from_config()
        except Exception as exc:  # pragma: no cover - defensive config guard
            return self._result(
                score=None,
                num_samples=0,
                details={
                    "evaluated_samples": [],
                    "skipped_samples": skipped_samples,
                    "failed_samples": [{"sample_id": "<config>", "reason": str(exc)}],
                },
                status="failed",
                reason="Invalid FVD config-level feature/stat inputs.",
            )

        if config_stats is not None:
            generated_stats, reference_stats = config_stats
            score = self._frechet_distance_from_stats(generated_stats, reference_stats)
            return self._result(
                score=score,
                num_samples=len(paired_samples),
                details={
                    "evaluated_samples": [],
                    "skipped_samples": skipped_samples,
                    "failed_samples": failed_samples,
                    "source": "config_stats",
                    "generated_count": int(generated_stats["count"]),
                    "reference_count": int(reference_stats["count"]),
                },
                status="ok",
                reason=None,
            )

        generated_feature_batches: list[np.ndarray] = []
        reference_feature_batches: list[np.ndarray] = []
        generated_stats_batches: list[dict[str, Any]] = []
        reference_stats_batches: list[dict[str, Any]] = []
        evaluated_samples = []

        for sample in paired_samples:
            try:
                sample_payload = self._load_sample_payload(sample)
            except _SkipSample as exc:
                skipped_samples.append({"sample_id": sample.sample_id, "reason": str(exc)})
                continue
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                failed_samples.append({"sample_id": sample.sample_id, "reason": str(exc)})
                continue

            if sample_payload["kind"] == "stats":
                generated_stats_batches.append(sample_payload["generated_stats"])
                reference_stats_batches.append(sample_payload["reference_stats"])
            else:
                generated_feature_batches.append(sample_payload["generated_features"])
                reference_feature_batches.append(sample_payload["reference_features"])

            evaluated_samples.append(
                {
                    "sample_id": sample.sample_id,
                    "source": sample_payload["kind"],
                    "generated_count": int(sample_payload["generated_count"]),
                    "reference_count": int(sample_payload["reference_count"]),
                }
            )

        if not evaluated_samples:
            return self._result(
                score=None,
                num_samples=0,
                details={
                    "evaluated_samples": [],
                    "skipped_samples": skipped_samples,
                    "failed_samples": failed_samples,
                },
                status="skipped" if not failed_samples else "failed",
                reason=(
                    "No usable precomputed FVD features or statistics were found. "
                    f"Provide metadata['{self.generated_feature_key}'] and "
                    f"metadata['{self.reference_feature_key}'], or matching stats keys."
                ),
            )

        if generated_feature_batches or reference_feature_batches:
            generated_features = np.concatenate(generated_feature_batches, axis=0)
            reference_features = np.concatenate(reference_feature_batches, axis=0)
            generated_stats_batches.append(self._feature_stats(generated_features))
            reference_stats_batches.append(self._feature_stats(reference_features))

        generated_stats = self._combine_stats(generated_stats_batches)
        reference_stats = self._combine_stats(reference_stats_batches)
        score = self._frechet_distance_from_stats(generated_stats, reference_stats)

        return self._result(
            score=score,
            num_samples=len(evaluated_samples),
            details={
                "evaluated_samples": evaluated_samples,
                "skipped_samples": skipped_samples,
                "failed_samples": failed_samples,
                "generated_feature_count": int(generated_stats["count"]),
                "reference_feature_count": int(reference_stats["count"]),
                "feature_dim": int(generated_stats["mean"].shape[0]),
            },
            status="ok",
            reason=None,
        )

    def _load_stats_from_config(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        generated_stats = self._load_optional_stats(
            self.config.get("generated_stats"),
            self.config.get("generated_stats_path"),
        )
        reference_stats = self._load_optional_stats(
            self.config.get("reference_stats"),
            self.config.get("reference_stats_path"),
        )
        if generated_stats is not None and reference_stats is not None:
            return generated_stats, reference_stats

        generated_features = self._load_optional_features(
            self.config.get("generated_features"),
            self.config.get("generated_features_path"),
        )
        reference_features = self._load_optional_features(
            self.config.get("reference_features"),
            self.config.get("reference_features_path"),
        )
        if generated_features is not None and reference_features is not None:
            return (
                self._feature_stats(generated_features),
                self._feature_stats(reference_features),
            )
        return None

    def _load_sample_payload(self, sample: GenerationSample) -> dict[str, Any]:
        metadata = sample.metadata or {}
        generated_stats = self._load_optional_stats(metadata.get(self.generated_stats_key))
        reference_stats = self._load_optional_stats(metadata.get(self.reference_stats_key))
        if generated_stats is not None and reference_stats is not None:
            self._validate_feature_dim_match(
                generated_stats["mean"],
                reference_stats["mean"],
                sample.sample_id,
            )
            return {
                "kind": "stats",
                "generated_stats": generated_stats,
                "reference_stats": reference_stats,
                "generated_count": generated_stats["count"],
                "reference_count": reference_stats["count"],
            }

        generated_features = self._load_optional_features(metadata.get(self.generated_feature_key))
        reference_features = self._load_optional_features(metadata.get(self.reference_feature_key))
        if generated_features is not None and reference_features is not None:
            self._validate_feature_dim_match(
                generated_features,
                reference_features,
                sample.sample_id,
            )
            return {
                "kind": "features",
                "generated_features": generated_features,
                "reference_features": reference_features,
                "generated_count": generated_features.shape[0],
                "reference_count": reference_features.shape[0],
            }

        raise _SkipSample(
            "Missing precomputed FVD inputs. "
            f"Expected metadata['{self.generated_feature_key}'] and "
            f"metadata['{self.reference_feature_key}'], or matching stats keys."
        )

    def _load_optional_features(
        self, raw_value: Any = None, path_value: Any = None
    ) -> np.ndarray | None:
        value = path_value if path_value is not None else raw_value
        if value is None:
            return None
        array = self._load_array_like(value, kind="features")
        return self._normalize_feature_array(array)

    def _load_optional_stats(
        self, raw_value: Any = None, path_value: Any = None
    ) -> dict[str, Any] | None:
        value = path_value if path_value is not None else raw_value
        if value is None:
            return None

        if isinstance(value, dict):
            payload = value
        elif isinstance(value, str):
            path = Path(value)
            if not path.exists():
                raise _SkipSample(f"FVD stats path does not exist: {value}")
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    payload = {key: data[key].tolist() for key in data.files}
            else:
                raise _SkipSample(
                    f"Unsupported FVD stats file format: {path.suffix or '<none>'}"
                )
        else:
            raise _SkipSample("FVD stats must be a dict or a .json/.npz path.")

        if "mean" not in payload or "cov" not in payload:
            raise _SkipSample("FVD stats must contain 'mean' and 'cov'.")
        mean = np.asarray(payload["mean"], dtype=np.float64).reshape(-1)
        cov = np.asarray(payload["cov"], dtype=np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise _SkipSample("FVD cov must be a square matrix.")
        if cov.shape[0] != mean.shape[0]:
            raise _SkipSample("FVD mean and cov dimensions do not match.")
        count = int(payload.get("count", 1))
        return {"mean": mean, "cov": cov, "count": count}

    def _load_array_like(self, value: Any, *, kind: str) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, list):
            return np.asarray(value)
        if isinstance(value, str):
            path = Path(value)
            if not path.exists():
                raise _SkipSample(f"FVD {kind} path does not exist: {value}")
            if path.suffix.lower() == ".npy":
                return np.load(path, allow_pickle=False)
            if path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    first_key = data.files[0] if data.files else None
                    if first_key is None:
                        raise _SkipSample(f"FVD {kind} archive is empty: {value}")
                    return data[first_key]
            if path.suffix.lower() == ".json":
                return np.asarray(json.loads(path.read_text(encoding="utf-8")))
            raise _SkipSample(
                f"Unsupported FVD {kind} file format: {path.suffix or '<none>'}"
            )
        raise _SkipSample(f"FVD {kind} must be an array, list, or supported file path.")

    def _normalize_feature_array(self, array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float64)
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim == 2:
            return array
        return array.reshape(array.shape[0], -1)

    def _validate_feature_dim_match(
        self, generated: np.ndarray, reference: np.ndarray, sample_id: str
    ) -> None:
        generated_dim = int(generated.shape[-1])
        reference_dim = int(reference.shape[-1])
        if generated_dim != reference_dim:
            raise _SkipSample(
                f"Generated/reference FVD feature dimensions do not match for sample '{sample_id}'."
            )

    def _feature_stats(self, features: np.ndarray) -> dict[str, Any]:
        features = self._normalize_feature_array(features)
        mean = features.mean(axis=0)
        if features.shape[0] > 1:
            cov = np.cov(features, rowvar=False)
        else:
            cov = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
        return {"mean": mean, "cov": cov, "count": int(features.shape[0])}

    def _frechet_distance_from_features(
        self, generated_features: np.ndarray, reference_features: np.ndarray
    ) -> float:
        generated_stats = self._feature_stats(generated_features)
        reference_stats = self._feature_stats(reference_features)
        return self._frechet_distance_from_stats(generated_stats, reference_stats)

    def _frechet_distance_from_stats(
        self, generated_stats: dict[str, Any], reference_stats: dict[str, Any]
    ) -> float:
        mean_a = np.asarray(generated_stats["mean"], dtype=np.float64).reshape(-1)
        mean_b = np.asarray(reference_stats["mean"], dtype=np.float64).reshape(-1)
        cov_a = np.asarray(generated_stats["cov"], dtype=np.float64)
        cov_b = np.asarray(reference_stats["cov"], dtype=np.float64)
        diff = mean_a - mean_b
        mean_term = float(diff @ diff)
        trace_term = float(np.trace(cov_a) + np.trace(cov_b) - 2.0 * self._trace_sqrt_product(cov_a, cov_b))
        return mean_term + trace_term

    def _combine_stats(self, stats_list: list[dict[str, Any]]) -> dict[str, Any]:
        if not stats_list:
            raise _SkipSample("No FVD statistics were available to combine.")
        if len(stats_list) == 1:
            return stats_list[0]

        total_count = sum(max(int(stats["count"]), 1) for stats in stats_list)
        feature_dim = int(np.asarray(stats_list[0]["mean"]).shape[0])
        combined_mean = np.zeros(feature_dim, dtype=np.float64)
        for stats in stats_list:
            count = max(int(stats["count"]), 1)
            combined_mean += count * np.asarray(stats["mean"], dtype=np.float64)
        combined_mean /= total_count

        combined_cov = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        for stats in stats_list:
            count = max(int(stats["count"]), 1)
            mean = np.asarray(stats["mean"], dtype=np.float64)
            cov = np.asarray(stats["cov"], dtype=np.float64)
            diff = (mean - combined_mean).reshape(-1, 1)
            combined_cov += count * (cov + diff @ diff.T)
        combined_cov /= total_count

        return {"mean": combined_mean, "cov": combined_cov, "count": int(total_count)}

    def _trace_sqrt_product(self, cov_a: np.ndarray, cov_b: np.ndarray) -> float:
        sqrt_cov_a = self._symmetric_matrix_square_root(cov_a)
        sqrt_product = sqrt_cov_a @ cov_b @ sqrt_cov_a
        return float(np.trace(self._symmetric_matrix_square_root(sqrt_product)))

    def _symmetric_matrix_square_root(self, matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        clipped = np.where(eigenvalues < eps, 0.0, eigenvalues)
        sqrt_diag = np.diag(np.sqrt(clipped))
        return eigenvectors @ sqrt_diag @ eigenvectors.T

    def _result(
        self,
        *,
        score: float | None,
        num_samples: int,
        details: dict[str, Any],
        status: str,
        reason: str | None,
    ) -> dict[str, Any]:
        result = {
            "metric": self.name,
            "score": score,
            "num_samples": num_samples,
            "details": details,
            "status": status,
        }
        if reason is not None:
            result["reason"] = reason
        return result


class _SkipSample(Exception):
    pass
