"""Safe file upload/download helpers with local-directory permission checks.

The Settings page lets users point uploads/downloads at any local directory,
but only when ``allow_external_paths`` is enabled. Everything else stays
rooted under the project directory for safety.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple


def resolve_dir(raw: str, root: Path, default_rel: str, allow_external: bool = False) -> Path:
    """Resolve a user-specified directory into an absolute path.

    - empty -> ``root / default_rel`` (always inside the project)
    - ``~`` and relative paths expand; a relative path is resolved against ``root``
    - absolute paths outside ``root`` raise ``PermissionError`` unless
      ``allow_external`` is True.
    """
    root = Path(root).expanduser().resolve()
    if not raw or not raw.strip():
        return root / default_rel
    p = Path(raw.strip()).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if not allow_external:
        try:
            p.relative_to(root)
        except ValueError:
            raise PermissionError(
                f"Path {p} is outside the project root ({root}). "
                "Enable 'allow external paths' in Settings to permit it."
            )
    return p


def is_within(path: Path, root: Path) -> bool:
    """True when ``path`` resolves inside ``root``."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(name: str) -> str:
    """Sanitize a filename so uploads can't path-traverse out of the target dir."""
    # normalize both separators, then take only the final path component
    base = Path(str(name).replace("\\", "/")).name
    return "".join(ch for ch in base if ch not in "\x00/") or "unnamed"


def save_upload(data: bytes, dest_dir: Path, name: str) -> Path:
    """Write uploaded bytes into ``dest_dir`` under a sanitized filename."""
    dest = ensure_dir(dest_dir) / safe_name(name)
    dest.write_bytes(data)
    return dest


def list_artifacts(root: Path, dirs: Iterable[str]) -> List[Path]:
    """Collect downloadable artifacts (files) from the given relative dirs.

    Returns a sorted list of existing file paths. Symlinks are resolved so a
    path that escapes the project is flagged by callers via ``is_within``.
    """
    root = Path(root)
    seen = set()
    out: List[Path] = []
    for rel in dirs:
        base = root / rel
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(p)
    return out


def copy_download(src: Path, dest_dir: Path) -> Tuple[Path, bool]:
    """Copy ``src`` into ``dest_dir``. Returns (destination, overwrote)."""
    import shutil

    dest = ensure_dir(dest_dir) / src.name
    overwrote = dest.exists()
    shutil.copy2(src, dest)
    return dest, overwrote
