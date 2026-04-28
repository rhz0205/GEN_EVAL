from __future__ import annotations

from pathlib import Path
from typing import Any


class ReferenceGenerator:
    name: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def prepare_sample(self, sample: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        raise NotImplementedError
