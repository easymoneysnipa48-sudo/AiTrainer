"""Project directory layout helpers."""
from __future__ import annotations

from pathlib import Path

LAYOUT = [
    "data/raw",
    "data/clean",
    "data/segments",
    "data/train",
    "data/val",
    "data/test",
    "metadata",
    "configs",
    "checkpoints",
    "outputs",
    "logs",
    "scripts",
    "notebooks",
]


def ensure_layout(root: Path) -> None:
    root = Path(root)
    for rel in LAYOUT:
        (root / rel).mkdir(parents=True, exist_ok=True)


def data_dir(root: Path, which: str) -> Path:
    return Path(root) / "data" / which


def meta_dir(root: Path) -> Path:
    return Path(root) / "metadata"
