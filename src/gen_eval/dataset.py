from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gen_eval.schemas import GenerationSample


def load_manifest(path: str | Path) -> list[GenerationSample]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples_data = _extract_samples(payload)
    return [GenerationSample.from_dict(item) for item in samples_data]


def _extract_samples(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "samples" in payload and isinstance(payload["samples"], list):
            return payload["samples"]
    raise ValueError("Manifest JSON must be a list or an object with a 'samples' list.")
