from __future__ import annotations

from pathlib import Path


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute() or base_dir is None:
        return raw_path
    return Path(base_dir) / raw_path


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
