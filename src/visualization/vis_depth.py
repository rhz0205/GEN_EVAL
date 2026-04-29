from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visualization.base import BaseVisualizer
from visualization.composer import compose_6v_image, compose_6v_video


class DepthVisualizer(BaseVisualizer):
    name = "depth"

    def render(self, input_dir: Path, output_dir: Path) -> dict[str, Any]:
        raw_result = self._render_raw_depth(input_dir, output_dir)
        image_dir = output_dir / "depth_6v_image"
        video_dir = output_dir / "depth_6v_video"
        image_result = compose_6v_image(input_dir, image_dir, name=self.name)
        video_result = compose_6v_video(input_dir, video_dir, name=self.name)
        return _merge_visualization_result(
            self.name,
            input_dir,
            image_dir,
            video_dir,
            raw_result,
            image_result,
            video_result,
        )

    def _render_raw_depth(self, input_dir: Path, output_dir: Path) -> dict[str, Any]:
        enriched_path = output_dir / "results" / "enriched_data.json"
        if not enriched_path.is_file():
            return {
                "name": self.name,
                "status": "skipped",
                "reason": f"enriched_data.json is missing: {enriched_path}",
                "input_dir": str(input_dir),
                "output_dir": str(input_dir),
            }

        payload = json.loads(enriched_path.read_text(encoding="utf-8"))
        samples = payload if isinstance(payload, list) else payload.get("samples", [])
        if not isinstance(samples, list):
            return {
                "name": self.name,
                "status": "skipped",
                "reason": f"samples payload is invalid in {enriched_path}",
                "input_dir": str(input_dir),
                "output_dir": str(input_dir),
            }

        input_dir.mkdir(parents=True, exist_ok=True)
        rendered_count = 0
        skipped_samples: list[dict[str, str]] = []

        for sample in samples:
            if not isinstance(sample, dict):
                continue
            sample_id = str(sample.get("sample_id") or "unknown")
            metadata = sample.get("metadata")
            if not isinstance(metadata, dict):
                skipped_samples.append({"sample_id": sample_id, "reason": "metadata is missing or invalid"})
                continue
            depth_maps = metadata.get("depth_maps")
            if not isinstance(depth_maps, dict) or not depth_maps:
                skipped_samples.append({"sample_id": sample_id, "reason": "metadata.depth_maps is missing"})
                continue

            sample_dir = input_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_rendered = 0

            for view, depth_path in depth_maps.items():
                npy_path = Path(str(depth_path))
                if not npy_path.is_file():
                    continue
                try:
                    depth_array = np.load(npy_path)
                    frame = select_depth_frame(depth_array)
                    if frame is None:
                        continue
                    image = render_depth_frame(frame)
                    image.save(sample_dir / f"{view}_0000.png")
                    image.close()
                    sample_rendered += 1
                except Exception:
                    continue

            if sample_rendered > 0:
                rendered_count += 1
            else:
                skipped_samples.append({"sample_id": sample_id, "reason": "no depth frames could be rendered"})

        if rendered_count == 0:
            return {
                "name": self.name,
                "status": "skipped",
                "reason": "No samples produced raw depth images.",
                "input_dir": str(input_dir),
                "output_dir": str(input_dir),
                "num_rendered_samples": 0,
                "skipped_samples": skipped_samples,
            }

        payload = {
            "name": self.name,
            "status": "partial" if skipped_samples else "success",
            "input_dir": str(input_dir),
            "output_dir": str(input_dir),
            "num_rendered_samples": rendered_count,
            "skipped_samples": skipped_samples,
        }
        if skipped_samples:
            payload["reason"] = skipped_samples[0]["reason"]
        return payload


def _merge_visualization_result(
    name: str,
    input_dir: Path,
    image_dir: Path,
    video_dir: Path,
    raw_result: dict[str, Any],
    image_result: dict[str, Any],
    video_result: dict[str, Any],
) -> dict[str, Any]:
    statuses = {
        raw_result.get("status"),
        image_result.get("status"),
        video_result.get("status"),
    }
    if "success" in statuses:
        status = "success"
    elif "partial" in statuses:
        status = "partial"
    else:
        status = "skipped"
    reason = raw_result.get("reason") or image_result.get("reason") or video_result.get("reason")
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "input_dir": str(input_dir),
        "raw_result": raw_result,
        "image_output_dir": str(image_dir),
        "video_output_dir": str(video_dir),
        "image_result": image_result,
        "video_result": video_result,
    }


def select_depth_frame(depth_array: Any) -> np.ndarray | None:
    array = np.asarray(depth_array)
    if array.size == 0:
        return None
    if array.ndim >= 3:
        frame = array[0]
    elif array.ndim == 2:
        frame = array
    else:
        return None
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 2:
        return None
    return frame


def render_depth_frame(frame: np.ndarray) -> Image.Image:
    finite_mask = np.isfinite(frame)
    if not finite_mask.any():
        image = np.zeros_like(frame, dtype=np.uint8)
    else:
        valid = frame[finite_mask]
        min_value = float(valid.min())
        max_value = float(valid.max())
        if max_value > min_value:
            normalized = (frame - min_value) / (max_value - min_value)
        else:
            normalized = np.zeros_like(frame, dtype=np.float32)
        normalized = np.where(np.isfinite(normalized), normalized, 0.0)
        image = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([image, image, image], axis=-1)
    return Image.fromarray(rgb, mode="RGB")
