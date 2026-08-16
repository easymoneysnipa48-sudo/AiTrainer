"""Lyric project persistence (feature #10).

Saves a full writing session — the beat reference, style recipe, structure,
weights/negatives, and the generated result — as one JSON file under
``metadata/lyric_projects/<name>.json``, so you can pick up where you left off.

Kept deliberately plain (JSON, no schema migration) so projects are portable
and diffable in git.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECTS_DIR = "lyric_projects"


def _dir(root: Path) -> Path:
    return Path(root) / "metadata" / _PROJECTS_DIR


def _path(root: Path, name: str) -> Path:
    return _dir(root) / f"{name}.json"


def save_project(root: Path, name: str, payload: Dict[str, Any]) -> Path:
    p = _path(root, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p


def load_project(root: Path, name: str) -> Optional[Dict[str, Any]]:
    p = _path(root, name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def list_projects(root: Path) -> List[str]:
    d = _dir(root)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def delete_project(root: Path, name: str) -> bool:
    p = _path(root, name)
    if not p.exists():
        return False
    p.unlink()
    return True
