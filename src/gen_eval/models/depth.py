from __future__ import annotations

from typing import Any

from gen_eval.models.backbones import LazyModelHandle


class DepthModelAdapter:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.handle = LazyModelHandle(name="depth_model", config=config or {})

    def load(self) -> None:
        self.handle.load()
