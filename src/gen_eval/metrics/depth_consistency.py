from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gen_eval.schemas import GenerationSample


class DepthConsistencyMetric:
    name = "depth_consistency"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.generated_depth_key = str(config.get("generated_depth_key", "generated_depth"))
        self.reference_depth_key = str(config.get("reference_depth_key", "reference_depth"))
        self.generated_depth_frames_key = str(
            config.get("generated_depth_frames_key", "generated_depth_frames")
        )
        self.reference_depth_frames_key = str(
            config.get("reference_depth_frames_key", "reference_depth_frames")
        )
        self.generated_depth_video_key = str(
            config.get("generated_depth_video_key", "generated_depth_video")
        )
        self.reference_depth_video_key = str(
            config.get("reference_depth_video_key", "reference_depth_video")
        )
        self.depth_model_path = config.get("depth_model_path")
        self.epsilon = float(config.get("epsilon", 1e-8))

    def evaluate(self, samples: list[GenerationSample]) -> dict[str, Any]:
        if self.depth_model_path and not self._has_any_precomputed_depth(samples):
            return self._result(
                score=None,
                num_samples=0,
                details={"evaluated_samples": []},
                status="skipped",
                reason=(
                    "Model-backed depth inference is intentionally not implemented in this lightweight "
                    "migration. Provide prepared depth metadata instead."
                ),
            )

        runtime, reason = self._get_runtime()
        if runtime is None:
            return self._result(
                score=None,
                num_samples=0,
                details={"evaluated_samples": []},
                status="skipped",
                reason=reason,
            )

        evaluated_samples = []
        skipped_samples = []
        failed_samples = []
        total_score = 0.0
        total_mae = 0.0
        total_rel = 0.0
        total_temporal = 0.0

        for sample in samples:
            try:
                generated_depth, reference_depth, source_details = self._load_depth_pair(sample, runtime)
                sample_scores = self._score_depth_pair(generated_depth, reference_depth)
            except _SkipSample as exc:
                skipped_samples.append({"sample_id": sample.sample_id, "reason": str(exc)})
                continue
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                failed_samples.append({"sample_id": sample.sample_id, "reason": str(exc)})
                continue

            evaluated_samples.append(
                {
                    "sample_id": sample.sample_id,
                    "score": sample_scores["score"],
                    "depth_mae": sample_scores["depth_mae"],
                    "depth_rel": sample_scores["depth_rel"],
                    "temporal_alignment": sample_scores["temporal_alignment"],
                    "num_frames": int(sample_scores["num_frames"]),
                    "source": source_details,
                }
            )
            total_score += sample_scores["score"]
            total_mae += sample_scores["depth_mae"]
            total_rel += sample_scores["depth_rel"]
            total_temporal += sample_scores["temporal_alignment"]

        if not evaluated_samples:
            return self._result(
                score=None,
                num_samples=0,
                details={
                    "evaluated_samples": [],
                    "skipped_samples": skipped_samples,
                    "failed_samples": failed_samples,
                },
                status="failed" if failed_samples else "skipped",
                reason=(
                    "No prepared generated/reference depth data was found. "
                    f"Provide metadata['{self.generated_depth_key}'] and metadata['{self.reference_depth_key}'], "
                    f"or the matching frame/video keys."
                ),
            )

        num_samples = len(evaluated_samples)
        summary = {
            "depth_consistency": total_score / num_samples,
            "depth_mae": total_mae / num_samples,
            "depth_rel": total_rel / num_samples,
            "temporal_alignment": total_temporal / num_samples,
        }
        return self._result(
            score=summary["depth_consistency"],
            num_samples=num_samples,
            details={
                "average_results": summary,
                "evaluated_samples": evaluated_samples,
                "skipped_samples": skipped_samples,
                "failed_samples": failed_samples,
            },
            status="ok",
            reason=None,
        )

    def _get_runtime(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception as exc:
            return None, f"Required depth-loading dependency is unavailable: {exc}"
        return {"imageio": imageio}, None

    def _has_any_precomputed_depth(self, samples: list[GenerationSample]) -> bool:
        for sample in samples:
            metadata = sample.metadata or {}
            if any(
                key in metadata
                for key in (
                    self.generated_depth_key,
                    self.reference_depth_key,
                    self.generated_depth_frames_key,
                    self.reference_depth_frames_key,
                    self.generated_depth_video_key,
                    self.reference_depth_video_key,
                )
            ):
                return True
        return False

    def _load_depth_pair(
        self, sample: GenerationSample, runtime: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
        metadata = sample.metadata or {}

        generated_depth = self._load_single_depth(
            metadata,
            array_key=self.generated_depth_key,
            frames_key=self.generated_depth_frames_key,
            video_key=self.generated_depth_video_key,
            runtime=runtime,
            label="generated",
        )
        reference_depth = self._load_single_depth(
            metadata,
            array_key=self.reference_depth_key,
            frames_key=self.reference_depth_frames_key,
            video_key=self.reference_depth_video_key,
            runtime=runtime,
            label="reference",
        )

        if generated_depth.shape[0] == 0 or reference_depth.shape[0] == 0:
            raise _SkipSample("Prepared depth data must contain at least one frame.")

        if generated_depth.shape[1:] != reference_depth.shape[1:]:
            raise _SkipSample("Generated/reference depth shapes do not match.")

        frame_count = min(generated_depth.shape[0], reference_depth.shape[0])
        if frame_count == 0:
            raise _SkipSample("No overlapping depth frames were available for comparison.")

        return (
            generated_depth[:frame_count],
            reference_depth[:frame_count],
            {
                "generated": self._detect_source_kind(metadata, self.generated_depth_key, self.generated_depth_frames_key, self.generated_depth_video_key),
                "reference": self._detect_source_kind(metadata, self.reference_depth_key, self.reference_depth_frames_key, self.reference_depth_video_key),
            },
        )

    def _load_single_depth(
        self,
        metadata: dict[str, Any],
        *,
        array_key: str,
        frames_key: str,
        video_key: str,
        runtime: dict[str, Any],
        label: str,
    ) -> np.ndarray:
        if array_key in metadata:
            return self._normalize_depth_array(
                self._load_array_like(metadata[array_key], kind=f"{label}_depth")
            )
        if frames_key in metadata:
            return self._normalize_depth_array(
                self._load_array_like(metadata[frames_key], kind=f"{label}_depth_frames")
            )
        if video_key in metadata:
            raw_value = metadata[video_key]
            if not isinstance(raw_value, str):
                raise _SkipSample(f"{video_key} must be a file path.")
            path = Path(raw_value)
            if not path.exists():
                raise _SkipSample(f"{video_key} path does not exist: {raw_value}")
            frames = np.asarray(runtime["imageio"].mimread(path))
            return self._normalize_depth_array(frames)

        raise _SkipSample(
            f"Missing prepared {label} depth data. Expected metadata key '{array_key}', '{frames_key}', or '{video_key}'."
        )

    def _load_array_like(self, value: Any, *, kind: str) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, list):
            return np.asarray(value)
        if isinstance(value, str):
            path = Path(value)
            if not path.exists():
                raise _SkipSample(f"{kind} path does not exist: {value}")
            if path.suffix.lower() == ".npy":
                return np.load(path, allow_pickle=False)
            if path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    first_key = data.files[0] if data.files else None
                    if first_key is None:
                        raise _SkipSample(f"{kind} archive is empty: {value}")
                    return data[first_key]
            if path.suffix.lower() == ".json":
                return np.asarray(json.loads(path.read_text(encoding="utf-8")))
            raise _SkipSample(f"Unsupported {kind} file format: {path.suffix or '<none>'}")
        raise _SkipSample(f"{kind} must be an array, list, or supported file path.")

    def _normalize_depth_array(self, array: np.ndarray) -> np.ndarray:
        depth = np.asarray(array, dtype=np.float64)
        if depth.ndim == 2:
            depth = depth[None, ...]
        elif depth.ndim == 4:
            if depth.shape[-1] in (3, 4):
                depth = depth[..., 0]
            elif depth.shape[1] in (1, 3, 4):
                depth = depth[:, 0, ...]
            else:
                raise _SkipSample("Depth arrays with 4 dimensions must be frame-channel images.")
        elif depth.ndim != 3:
            raise _SkipSample("Depth data must have shape [T,H,W], [H,W], or image-style depth frames.")
        return depth

    def _score_depth_pair(
        self, generated_depth: np.ndarray, reference_depth: np.ndarray
    ) -> dict[str, float | int]:
        mae = float(np.mean(np.abs(generated_depth - reference_depth)))
        denom = np.maximum(np.abs(reference_depth), self.epsilon)
        rel = float(np.mean(np.abs(generated_depth - reference_depth) / denom))

        if generated_depth.shape[0] >= 2 and reference_depth.shape[0] >= 2:
            generated_delta = np.diff(generated_depth, axis=0)
            reference_delta = np.diff(reference_depth, axis=0)
            temporal_mae = float(np.mean(np.abs(generated_delta - reference_delta)))
            temporal_scale = float(np.mean(np.abs(reference_delta)) + self.epsilon)
            temporal_alignment = float(np.exp(-temporal_mae / temporal_scale))
        else:
            temporal_alignment = 1.0

        spatial_scale = float(np.mean(np.abs(reference_depth)) + self.epsilon)
        score = float(np.exp(-mae / spatial_scale) * np.exp(-rel) * temporal_alignment)
        return {
            "score": score,
            "depth_mae": mae,
            "depth_rel": rel,
            "temporal_alignment": temporal_alignment,
            "num_frames": int(generated_depth.shape[0]),
        }

    def _detect_source_kind(
        self, metadata: dict[str, Any], array_key: str, frames_key: str, video_key: str
    ) -> str:
        if array_key in metadata:
            return array_key
        if frames_key in metadata:
            return frames_key
        if video_key in metadata:
            return video_key
        return "unknown"

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


DepthConsistency = DepthConsistencyMetric


class _SkipSample(Exception):
    pass
