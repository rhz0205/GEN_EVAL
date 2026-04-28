from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# 将不同来源的数据、manifest、脚本生成的样本，转换成统一的 GenerationSample。
@dataclass
class ObjectTrack:
    """目标实例或轨迹，服务于后续的 instance_consistency 指标等需要进行实例级评估的指标"""

    object_id: str
    category: str
    boxes: list[dict[str, Any]] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectTrack":
        return cls(
            object_id=str(data.get("object_id", "")),
            category=str(data.get("category", "")),
            boxes=list(data.get("boxes", [])),
            attributes=dict(data.get("attributes", {})),
        )

    # 返回：
    # - 目标实例 ID: str,
    # - 目标实例类别: str,
    # - 目标实例在视频中的位置和时间信息: list[dict]，
    # - 目标实例的其他属性信息: dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "boxes": self.boxes,
            "attributes": self.attributes,
        }


@dataclass
class GenerationSample:
    """评估样本单元"""

    sample_id: str
    generated_video: str
    reference_video: str | None = None
    prompt: str = ""
    objects: list[ObjectTrack] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationSample":
        object_items = data.get("objects", [])
        return cls(
            sample_id=str(data["sample_id"]),
            generated_video=str(data["generated_video"]),
            reference_video=data.get("reference_video"),
            prompt=str(data.get("prompt", "")),
            objects=[ObjectTrack.from_dict(item) for item in object_items],
            metadata=dict(data.get("metadata", {})),
        )

    # - sample_id: str, 评估样本的唯一标识符
    # - generated_video: str, 生成视频的路径或 URL
    # - reference_video: str | None, 可选的参考视频路径或 URL
    # - prompt: str, 生成视频所使用的文本提示信息
    # - objects: list[ObjectTrack], 视频中涉及的目标实例或轨迹列表
    # - metadata: dict[str, Any], 其他与评估样本相关的元信息（多视角）
    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "generated_video": self.generated_video,
            "reference_video": self.reference_video,
            "prompt": self.prompt,
            "objects": [item.to_dict() for item in self.objects],
            "metadata": self.metadata,
        }
