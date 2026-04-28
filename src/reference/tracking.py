from __future__ import annotations

from pathlib import Path
from typing import Any

from reference.base import ReferenceGenerator


class ObjectTrackReference(ReferenceGenerator):
    name = "object_tracks"

    def prepare_sample(self, sample: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        raise NotImplementedError("Object track reference generation is reserved for a later migration step.")
