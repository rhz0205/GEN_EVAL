from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

EXPECTED_CAMERA_VIEWS: tuple[str, ...] = (
    "camera_front",
    "camera_cross_left",
    "camera_cross_right",
    "camera_rear_left",
    "camera_rear_right",
    "camera_rear",
)


class AppearanceConsistency:
    name = "appearance_consistency"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # 读取指标运行配置，并统一约束输入字段、模型路径和推理参数。
        self.config = config or {}
        self.camera_videos_key = self.config.get("camera_videos_key", "camera_videos")
        self.expected_camera_views = tuple(
            self.config.get("expected_camera_views", EXPECTED_CAMERA_VIEWS)
        )
        self.device = self.config.get("device", "cuda")

        self.repo_path = self.config.get("repo_path")
        self.weight_path = self.config.get("weight_path")
        self.model_name = self.config.get("model_name", "dino_vitb16")
        self.use_fp16 = bool(self.config.get("use_fp16", False))
        self.strict_load = bool(self.config.get("strict_load", True))
        self.image_size = int(self.config.get("image_size", 224))
        self.batch_size = int(self.config.get("batch_size", 16))
        self.eps = float(self.config.get("eps", 1e-8))

        self._dino_model: Any | None = None
        self._transform: Any | None = None

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        # 先完成 DINO 模型初始化；若运行条件缺失，则以指标级 skipped 返回。
        runtime_error = self._ensure_dino()
        if runtime_error is not None:
            return {
                "metric": self.name,
                "status": "skipped",
                "num_samples": len(samples),
                "valid_sample_count": 0,
                "mean_appearance_consistency_score": None,
                "details": {
                    "evaluated_samples": [],
                    "skipped_samples": [],
                    "failed_samples": [],
                },
                "reason": runtime_error,
            }

        evaluated_samples: list[dict[str, Any]] = []
        skipped_samples: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        valid_scores: list[float] = []

        # 逐样本执行评价，并将成功、跳过和失败结果整理为统一输出结构。
        for sample in samples:
            sample_id = getattr(sample, "sample_id", None) or "unknown"

            try:
                sample_result = self._evaluate_sample(sample)
            except Exception as exc:
                sample_result = {
                    "sample_id": sample_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }

            status = sample_result.get("status")
            score = sample_result.get("appearance_consistency_score")

            if status == "success" and is_finite_number(score):
                valid_score = float(score)
                valid_scores.append(valid_score)
                evaluated_samples.append(
                    {
                        "sample_id": sample_id,
                        "appearance_consistency_score": valid_score,
                    }
                )
            elif status == "failed":
                failed_samples.append(simplify_sample_result(sample_result))
            else:
                skipped_samples.append(simplify_sample_result(sample_result))

        # 汇总所有有效样本分数；若没有有效分数，则保留跳过状态和原因。
        mean_score = mean_or_none(valid_scores)
        status = "success" if mean_score is not None else "skipped"
        reason = (
            None if mean_score is not None else "No valid appearance consistency score."
        )

        result: dict[str, Any] = {
            "metric": self.name,
            "status": status,
            "num_samples": len(samples),
            "valid_sample_count": len(valid_scores),
            "mean_appearance_consistency_score": mean_score,
            "details": {
                "evaluated_samples": evaluated_samples,
                "skipped_samples": skipped_samples,
                "failed_samples": failed_samples,
            },
        }
        if reason:
            result["reason"] = reason
        return result

    def _evaluate_sample(self, sample: Any) -> dict[str, Any]:
        # 从样本 metadata 中解析多相机视频路径，并检查固定视角输入是否完整。
        sample_id = getattr(sample, "sample_id", None) or "unknown"
        metadata = getattr(sample, "metadata", None) or {}
        camera_videos = metadata.get(self.camera_videos_key)
        if not isinstance(camera_videos, dict) or not camera_videos:
            return skipped_result(
                sample_id,
                f"metadata['{self.camera_videos_key}'] must be a non-empty dict.",
            )

        generated_videos = {
            str(view): str(path)
            for view, path in camera_videos.items()
            if path is not None
        }
        missing_views = [
            view for view in self.expected_camera_views if view not in generated_videos
        ]
        if missing_views:
            return skipped_result(
                sample_id,
                f"Missing camera views: {', '.join(missing_views)}.",
            )

        # 分别计算各相机视频的外观时序一致性，再形成样本级平均分。
        view_scores: list[float] = []
        for view in self.expected_camera_views:
            view_score = self._evaluate_view_video(generated_videos[view])
            if is_finite_number(view_score):
                view_scores.append(float(view_score))

        if not view_scores:
            return skipped_result(sample_id, "No valid video score.")

        return {
            "sample_id": sample_id,
            "status": "success",
            "appearance_consistency_score": mean_or_none(view_scores),
        }

    def _evaluate_view_video(self, video_path: str) -> float | None:
        # 对单个视角视频执行读取、特征提取和无参考外观一致性计算。
        path = Path(video_path)
        if not path.is_file():
            return None

        frames = self._read_all_frames(path)
        if len(frames) < 2:
            return None

        features = self._extract_dino_features(frames)
        if features is None or len(features) < 2:
            return None

        return self._compute_appearance_consistency_score(features)

    def _compute_appearance_consistency_score(
        self, generated_features: Any
    ) -> float | None:
        # 使用 ACM 表征相邻帧平滑度，使用 TJI 惩罚二阶时间抖动。
        if generated_features is None or len(generated_features) < 2:
            return None

        with torch.no_grad():
            acm = self._compute_acm(generated_features)
            tji = self._compute_tji(generated_features)
            score = acm / (1.0 + tji)
            return clamp01(float(score))

    def _compute_acm(self, features: Any) -> float:
        # 计算相邻帧 DINO 特征余弦相似度，作为外观连续性的主项。
        adjacent_similarities = torch.nn.functional.cosine_similarity(
            features[:-1],
            features[1:],
            dim=-1,
        )
        adjacent_similarities = torch.clamp(adjacent_similarities, min=0.0)
        return clamp01(float(adjacent_similarities.mean().item()))

    def _compute_tji(self, features: Any) -> float:
        # 计算归一化二阶特征变化，作为帧间闪烁和突变的惩罚项。
        if len(features) < 3:
            return 0.0

        velocity = (features[1:] - features[:-1]).norm(dim=1)
        acceleration = (features[2:] - 2 * features[1:-1] + features[:-2]).norm(dim=1)
        denominator = 0.5 * (velocity[1:] + velocity[:-1]) + self.eps
        jitter = (acceleration / denominator).mean()
        return max(0.0, float(jitter.item()))

    def _ensure_dino(self) -> str | None:
        # 检查设备和本地模型资源，按需加载 DINO 主干与图像预处理流水线。
        if self._dino_model is not None and self._transform is not None:
            return None

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        if not self.repo_path:
            return "Missing DINO repo path: set config['repo_path']."
        repo_path = Path(str(self.repo_path)).expanduser().resolve()
        if not repo_path.exists():
            return f"DINO repo path does not exist: {repo_path}"

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        if not self.weight_path:
            return "Missing DINO weight path: set config['weight_path']."
        weight_path = Path(str(self.weight_path)).expanduser().resolve()
        if not weight_path.exists():
            return f"DINO weight path does not exist: {weight_path}"

        try:
            model = torch.hub.load(
                str(repo_path),
                self.model_name,
                source="local",
                pretrained=False,
            )
            model.to(self.device)

            state_dict = torch.load(str(weight_path), map_location=self.device)
            model.load_state_dict(state_dict, strict=self.strict_load)

            if self.use_fp16 and self.device == "cuda":
                model = model.half()

            model.eval()
            self._dino_model = model
            self._transform = self._build_transform()
        except Exception as exc:
            self._dino_model = None
            self._transform = None
            return f"Failed to load DINO model: {exc}"

        return None

    def _build_transform(self) -> Any:
        # 构建与 DINO 输入要求一致的尺寸变换和 ImageNet 归一化流程。
        def robust_to_tensor(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                if x.dtype == torch.uint8:
                    return x.float() / 255.0
                return x
            return TF.to_tensor(x)

        return transforms.Compose(
            [
                transforms.Lambda(robust_to_tensor),
                transforms.Resize(
                    self.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(self.image_size),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

    def _extract_dino_features(self, frames: list[Any]) -> Any:
        # 将视频帧批量送入 DINO，并对输出特征进行 L2 归一化。
        if self._dino_model is None or self._transform is None:
            raise RuntimeError("DINO model is not initialized")

        images = [self._transform(Image.fromarray(frame_rgb)) for frame_rgb in frames]
        features_list: list[Any] = []

        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                batch_tensor = torch.stack(batch, dim=0).to(self.device)
                if self.use_fp16 and self.device == "cuda":
                    batch_tensor = batch_tensor.half()

                features = self._dino_model(batch_tensor)
                features = torch.nn.functional.normalize(features, dim=-1, p=2)
                features_list.append(features)

        if not features_list:
            return None
        return torch.cat(features_list, dim=0)

    def _read_all_frames(self, video_path: Path) -> list[Any]:
        # 使用 OpenCV 顺序读取视频帧，并统一转换为 RGB 格式。
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return []

        frames: list[Any] = []
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        finally:
            cap.release()
        return frames


def skipped_result(sample_id: str, reason: str) -> dict[str, Any]:
    # 生成统一的样本级跳过结果，便于上层汇总。
    return {"sample_id": sample_id, "status": "skipped", "reason": reason}


def simplify_sample_result(result: dict[str, Any]) -> dict[str, Any]:
    # 仅保留样本编号和原因，避免 details 中重复写入冗余字段。
    return {
        "sample_id": result.get("sample_id", "unknown"),
        "reason": result.get("reason", "unknown"),
    }


def mean_or_none(values: list[float]) -> float | None:
    # 对有效分数求均值；空列表返回 None 以表示没有可用结果。
    if not values:
        return None
    return float(sum(values) / len(values))


def is_finite_number(value: Any) -> bool:
    # 检查输入是否为有限数值，过滤 None、NaN 和无穷大。
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def clamp01(value: float) -> float:
    # 将分数裁剪到 0 到 1 区间，保证指标输出范围稳定。
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
