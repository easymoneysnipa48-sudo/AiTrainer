"""Shared utility helpers."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def sanitize_slug(name: str) -> str:
    stem = Path(name).stem
    stem = stem.replace(" ", "_")
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem)
    return stem.strip("._-") or "track"


def format_seconds(sec: float) -> str:
    sec = int(float(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def human_size(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def unique_path(dst: Path) -> Path:
    """Return dst, or dst with an incrementing suffix if it already exists."""
    if not dst.exists():
        return dst
    stem, suffix = dst.stem, dst.suffix
    parent = dst.parent
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1
