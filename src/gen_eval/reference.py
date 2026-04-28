from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# 默认相机视角用于约束参考文件生成范围，并与后续多视角指标读取逻辑保持一致。
DEFAULT_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)


class ReferenceGenerator(ABC):
    name: str

    # 参考生成器只保存自身配置，具体输入检查、文件生成和 metadata 补丁由子类实现。
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    # 子类需要将单个样本转换为参考文件，并返回可合并到 manifest metadata 中的字段。
    @abstractmethod
    def prepare_sample(
        self,
        sample: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError


class OpenSeeDAdapter:
    # 适配器只封装 OpenSeeD 的模型侧配置和推理入口，不负责 manifest 读写和指标计算。
    def __init__(self, config: dict[str, Any]) -> None:
        self.repo_path = config.get("repo_path")
        self.config_path = config.get("config_path")
        self.weight_path = config.get("weight_path")
        self.device = str(config.get("device", "cuda"))
        self.vocabulary = list(config.get("vocabulary", []))
        self.score_threshold = float(config.get("score_threshold", 0.3))
        self.ignore_label = int(config.get("ignore_label", -1))
        self.max_frames = config.get("max_frames")
        self.frame_stride = int(config.get("frame_stride", 1))

        self._model: Any | None = None
        self._predictor: Any | None = None

    @property
    def num_classes(self) -> int:
        return len(self.vocabulary)

    # 视频级推理先完成帧读取与抽帧控制，再交给帧序列推理函数统一处理。
    def infer_video(self, video_path: str | Path) -> np.ndarray:
        frames = read_video_frames(
            video_path=video_path,
            frame_stride=self.frame_stride,
            max_frames=self.max_frames,
        )
        if not frames:
            raise ValueError(f"No frames can be read from video: {video_path}")
        return self.infer_frames(frames)

    # 帧序列推理保证模型只初始化一次，并将每帧结果堆叠为 [T, H, W] 的整型 mask。
    def infer_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        self._ensure_model()

        masks: list[np.ndarray] = []
        for frame_rgb in frames:
            masks.append(self.infer_frame(frame_rgb))

        return np.stack(masks, axis=0).astype(np.int32, copy=False)

    # 单帧推理是本地 OpenSeeD API 的主要接入点，后续需要在此处返回固定词表下的类别 ID 图。
    def infer_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        self._ensure_model()

        raise NotImplementedError(
            "OpenSeeD single-frame inference is not implemented yet. "
            "Please connect this method to your local OpenSeeD predictor and return "
            "a [H, W] int32 semantic mask whose class ids follow config['vocabulary']."
        )

    # 模型初始化负责路径注册、必要配置检查和本地 OpenSeeD 加载，避免在每帧推理时重复初始化。
    def _ensure_model(self) -> None:
        if self._model is not None or self._predictor is not None:
            return

        if self.repo_path:
            repo_path = str(Path(self.repo_path).expanduser().resolve())
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        if not self.config_path:
            raise ValueError("OpenSeeD config_path is required.")
        if not self.weight_path:
            raise ValueError("OpenSeeD weight_path is required.")
        if not self.vocabulary:
            raise ValueError("OpenSeeD vocabulary must not be empty.")

        raise NotImplementedError(
            "OpenSeeD model loading is not implemented yet. "
            "Please load your local OpenSeeD config, checkpoint, and predictor here. "
            "The adapter should keep the loaded object in self._model or self._predictor."
        )


class OpenSeeDSemanticGenerator(ReferenceGenerator):
    name = "openseed_semantics"

    # 语义参考生成器负责从样本 metadata 中读取相机视频，并将 OpenSeeD 输出保存为语义 mask 文件。
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        self.camera_videos_key = str(config.get("camera_videos_key", "camera_videos"))
        self.output_key = str(config.get("output_key", "semantic_masks"))
        self.num_classes_key = str(
            config.get("num_classes_key", "semantic_num_classes")
        )
        self.ignore_label_key = str(
            config.get("ignore_label_key", "semantic_ignore_label")
        )
        self.expected_camera_views = tuple(
            config.get("expected_camera_views", DEFAULT_CAMERA_VIEWS)
        )
        self.force = bool(config.get("force", False))
        self.strict_views = bool(config.get("strict_views", False))

        self.adapter = OpenSeeDAdapter(config)

    # 单样本处理会逐视角生成或复用缓存 mask，并返回供 enriched manifest 使用的 metadata 补丁。
    def prepare_sample(
        self,
        sample: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        sample_id = str(sample.get("sample_id") or "unknown")
        metadata = sample.get("metadata") or {}
        camera_videos = metadata.get(self.camera_videos_key)

        if not isinstance(camera_videos, dict) or not camera_videos:
            raise ValueError(
                f"sample {sample_id} metadata['{self.camera_videos_key}'] "
                "must be a non-empty camera video dict."
            )

        sample_dir = output_dir / self.name / sanitize_path_part(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        semantic_masks: dict[str, str] = {}
        missing_views: list[str] = []

        for view in self.expected_camera_views:
            video_path = camera_videos.get(view)
            if not video_path:
                missing_views.append(view)
                continue

            mask_path = sample_dir / f"{view}.npy"
            if self.force or not mask_path.exists():
                masks = self.adapter.infer_video(video_path)
                np.save(mask_path, masks.astype(np.int32, copy=False))

            semantic_masks[view] = normalize_path(mask_path)

        if self.strict_views and missing_views:
            raise ValueError(
                f"sample {sample_id} missing camera views: {', '.join(missing_views)}"
            )

        if not semantic_masks:
            raise ValueError(f"sample {sample_id} produced no semantic mask files.")

        return {
            self.output_key: semantic_masks,
            self.num_classes_key: self.adapter.num_classes,
            self.ignore_label_key: self.adapter.ignore_label,
        }


class DepthReferenceGenerator(ReferenceGenerator):
    name = "depth_reference"

    # 深度参考接口预留给后续深度图、深度序列或几何参考文件生成。
    def prepare_sample(
        self,
        sample: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Depth reference generation is reserved for future extension."
        )


class ObjectTrackReferenceGenerator(ReferenceGenerator):
    name = "object_tracks"

    # 目标轨迹接口预留给后续检测、跟踪、实例级一致性等参考文件生成。
    def prepare_sample(
        self,
        sample: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Object track reference generation is reserved for future extension."
        )


class PlanningResponseReferenceGenerator(ReferenceGenerator):
    name = "planning_response"

    # 规划响应接口预留给后续端到端规划代理输出或任务效用参考文件生成。
    def prepare_sample(
        self,
        sample: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Planning response reference generation is reserved for future extension."
        )


# 注册表用于将配置文件中的生成器名称映射到具体实现，后续扩展只需新增类并登记。
REFERENCE_GENERATORS: dict[str, type[ReferenceGenerator]] = {
    OpenSeeDSemanticGenerator.name: OpenSeeDSemanticGenerator,
    DepthReferenceGenerator.name: DepthReferenceGenerator,
    ObjectTrackReferenceGenerator.name: ObjectTrackReferenceGenerator,
    PlanningResponseReferenceGenerator.name: PlanningResponseReferenceGenerator,
}


class ReferencePreparer:
    # 调度器负责批量读取 manifest、调用参考生成器、合并 metadata，并写出 enriched manifest。
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.generators = self._build_generators(config)
        self.continue_on_error = bool(config.get("continue_on_error", False))
        self.error_key = str(config.get("error_key", "reference_errors"))

    # 主流程按样本依次执行所有启用的参考生成器，并保留错误摘要便于数据准备阶段排查。
    def prepare_manifest(
        self,
        manifest_path: str | Path,
        output_manifest_path: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        manifest = load_json(manifest_path)
        samples = get_manifest_samples(manifest)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        enriched_samples: list[dict[str, Any]] = []
        summary: dict[str, Any] = {
            "num_samples": len(samples),
            "num_generators": len(self.generators),
            "failed_samples": [],
        }

        for sample in samples:
            enriched_sample = deepcopy(sample)
            metadata = dict(enriched_sample.get("metadata") or {})
            sample_errors: list[dict[str, str]] = []

            for generator in self.generators:
                try:
                    patch = generator.prepare_sample(enriched_sample, output_dir)
                    metadata = merge_metadata(metadata, patch)
                    enriched_sample["metadata"] = metadata
                except Exception as exc:
                    error_item = {
                        "generator": generator.name,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    sample_errors.append(error_item)

                    if not self.continue_on_error:
                        sample_id = str(enriched_sample.get("sample_id", "unknown"))
                        raise RuntimeError(
                            f"Reference generation failed for sample={sample_id}, "
                            f"generator={generator.name}: {exc}"
                        ) from exc

            if sample_errors:
                metadata.setdefault(self.error_key, []).extend(sample_errors)
                summary["failed_samples"].append(
                    {
                        "sample_id": enriched_sample.get("sample_id", "unknown"),
                        "errors": sample_errors,
                    }
                )

            enriched_sample["metadata"] = metadata
            enriched_samples.append(enriched_sample)

        enriched_manifest = set_manifest_samples(manifest, enriched_samples)
        write_json(enriched_manifest, output_manifest_path)

        summary_path = output_dir / "reference_summary.json"
        write_json(summary, summary_path)

        return enriched_manifest

    # 生成器构建阶段只读取配置中启用的项，并对未知名称进行显式报错。
    def _build_generators(
        self,
        config: dict[str, Any],
    ) -> list[ReferenceGenerator]:
        generators: list[ReferenceGenerator] = []

        for generator_config in config.get("reference_generators", []):
            if not generator_config.get("enabled", True):
                continue

            name = generator_config.get("name")
            if not name:
                raise ValueError("Each reference generator config must contain 'name'.")

            generator_cls = REFERENCE_GENERATORS.get(str(name))
            if generator_cls is None:
                available = ", ".join(sorted(REFERENCE_GENERATORS))
                raise ValueError(
                    f"Unknown reference generator: {name}. "
                    f"Available generators: {available}"
                )

            generators.append(generator_cls(generator_config))

        if not generators:
            raise ValueError("No enabled reference generators are configured.")

        return generators


# 视频读取工具统一完成路径检查、抽帧控制和 BGR 到 RGB 的颜色空间转换。
def read_video_frames(
    video_path: str | Path,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    frame_stride = max(1, int(frame_stride))
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: list[np.ndarray] = []
    frame_index = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            if frame_index % frame_stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                if max_frames is not None and len(frames) >= int(max_frames):
                    break

            frame_index += 1
    finally:
        cap.release()

    return frames


# JSON 工具函数保持 manifest 和摘要文件的读写格式一致。
def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


# manifest 工具函数同时兼容列表式 manifest 和包含 samples 字段的对象式 manifest。
def get_manifest_samples(manifest: Any) -> list[dict[str, Any]]:
    if isinstance(manifest, list):
        return manifest

    if isinstance(manifest, dict):
        samples = manifest.get("samples")
        if isinstance(samples, list):
            return samples

    raise ValueError("Manifest must be a list or a dict containing a 'samples' list.")


def set_manifest_samples(
    manifest: Any,
    samples: list[dict[str, Any]],
) -> Any:
    if isinstance(manifest, list):
        return samples

    if isinstance(manifest, dict):
        enriched_manifest = deepcopy(manifest)
        enriched_manifest["samples"] = samples
        return enriched_manifest

    raise ValueError("Manifest must be a list or a dict containing a 'samples' list.")


# metadata 合并以浅层合并为主，字典字段会保留已有内容并补充生成器输出。
def merge_metadata(
    metadata: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(metadata)

    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    return merged


# 路径工具函数用于统一输出路径格式，并避免 sample_id 中的特殊字符污染目录结构。
def normalize_path(path: str | Path) -> str:
    return os.path.normpath(str(path))


def sanitize_path_part(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) or "unknown"
