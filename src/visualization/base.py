from __future__ import annotations

from pathlib import Path
from typing import Any


class BaseVisualizer:
    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def render(self, input_dir: Path, output_dir: Path) -> dict[str, Any]:
        raise NotImplementedError
