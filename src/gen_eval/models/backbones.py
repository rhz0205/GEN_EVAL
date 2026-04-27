from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LazyModelHandle:
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    _loaded: bool = field(default=False, init=False, repr=False)

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        raise NotImplementedError(
            f"Model '{self.name}' is a placeholder. Implement lazy loading in a runtime-ready environment."
        )
