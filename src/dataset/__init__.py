from typing import Any

from .base import DEFAULT_CAMERA_VIEWS, BaseDataset
from .cosmos import CosmosDataset
from .geely import GeelyDataset
from .real import RealDataset
from .sample import SampleDataset

DATASET_REGISTRY: dict[str, type[BaseDataset]] = {
    "cosmos": CosmosDataset,
    "geely": GeelyDataset,
    "real": RealDataset,
    "sample": SampleDataset,
}


def build_dataset(dataset_name: str, dataset_config: dict[str, Any]) -> BaseDataset:
    dataset_class = DATASET_REGISTRY.get(dataset_name)
    if dataset_class is None:
        expected = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of: {expected}.")
    config = dict(dataset_config)
    config.setdefault("name", dataset_name)
    return dataset_class(config=config)


__all__ = [
    "BaseDataset",
    "CosmosDataset",
    "GeelyDataset",
    "RealDataset",
    "SampleDataset",
    "DEFAULT_CAMERA_VIEWS",
    "DATASET_REGISTRY",
    "build_dataset",
]
