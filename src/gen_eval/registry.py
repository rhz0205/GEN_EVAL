from __future__ import annotations

from collections.abc import Iterable


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, type] = {}

    def register(self, metric_class: type) -> type:
        name = getattr(metric_class, "name", None)
        if not name or not isinstance(name, str):
            raise ValueError("Metric class must define a string 'name' attribute.")
        self._metrics[name] = metric_class
        return metric_class

    def get(self, name: str) -> type:
        if name not in self._metrics:
            raise KeyError(f"Unknown metric: {name}")
        return self._metrics[name]

    def names(self) -> list[str]:
        return sorted(self._metrics)

    def items(self) -> Iterable[tuple[str, type]]:
        return self._metrics.items()


registry = MetricRegistry()
