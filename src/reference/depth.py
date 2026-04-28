from __future__ import annotations

from pathlib import Path
from typing import Any

from reference.base import ReferenceGenerator


class DepthReference(ReferenceGenerator):
    name = "depth_reference"

    def prepare_sample(self, sample: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        raise NotImplementedError("Depth reference generation is reserved for a later migration step.")
