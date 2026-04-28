from __future__ import annotations

from collections.abc import Iterable


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, type] = {}
        self._aliases: dict[str, str] = {}

    def register(self, metric_class: type) -> type:
        name = getattr(metric_class, "name", None)
        if not name or not isinstance(name, str):
            raise ValueError("Metric class must define a string 'name' attribute.")
        self._metrics[name] = metric_class
        return metric_class

    def register_alias(self, alias: str, target: str) -> None:
        if target not in self._metrics:
            raise KeyError(f"Cannot register alias '{alias}' for unknown metric '{target}'.")
        self._aliases[alias] = target

    def get(self, name: str) -> type:
        canonical_name = self._aliases.get(name, name)
        if canonical_name not in self._metrics:
            raise KeyError(f"Unknown metric: {name}")
        return self._metrics[canonical_name]

    def names(self) -> list[str]:
        return sorted(set(self._metrics) | set(self._aliases))

    def items(self) -> Iterable[tuple[str, type]]:
        return ((name, self.get(name)) for name in self.names())


registry = MetricRegistry()


def get_metric(name: str) -> type:
    return registry.get(name)
