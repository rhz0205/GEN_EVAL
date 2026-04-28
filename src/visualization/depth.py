from __future__ import annotations

from pathlib import Path
from typing import Any

from visualization.base import BaseVisualizer
from visualization.composer import compose_6v_image, compose_6v_video


class DepthVisualizer(BaseVisualizer):
    name = "depth"

    def render(self, input_dir: Path, output_dir: Path) -> dict[str, Any]:
        image_dir = output_dir / "depth_6v_image"
        video_dir = output_dir / "depth_6v_video"
        image_result = compose_6v_image(input_dir, image_dir, name=self.name)
        video_result = compose_6v_video(input_dir, video_dir, name=self.name)
        return _merge_visualization_result(self.name, input_dir, image_dir, video_dir, image_result, video_result)


def _merge_visualization_result(
    name: str,
    input_dir: Path,
    image_dir: Path,
    video_dir: Path,
    image_result: dict[str, Any],
    video_result: dict[str, Any],
) -> dict[str, Any]:
    statuses = {image_result.get("status"), video_result.get("status")}
    if "success" in statuses:
        status = "success"
    elif "partial" in statuses:
        status = "partial"
    else:
        status = "skipped"
    reason = image_result.get("reason") or video_result.get("reason")
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "input_dir": str(input_dir),
        "image_output_dir": str(image_dir),
        "video_output_dir": str(video_dir),
        "image_result": image_result,
        "video_result": video_result,
    }
