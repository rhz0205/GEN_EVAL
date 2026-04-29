from __future__ import annotations
from typing import Any



class BaseModule:
    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def evaluate(self, samples: list[Any]) -> dict[str, Any]:
        raise NotImplementedError
