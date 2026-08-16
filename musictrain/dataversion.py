"""DVC-style content-addressed dataset versioning (#14).

A self-contained, dependency-free version store: every file in ``data/<which>``
is hashed, unique content is copied into a blob store keyed by SHA-256, and a
version manifest records the file→hash map. From that you can diff two versions
or roll a directory back to any prior version without external tooling.

Layout:
* ``data_versions/objects/<aa>/<sha256>`` — content-addressed blob store
* ``metadata/dataset_versions.json`` — ordered list of version manifests
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import console
from .logging import get_logger

log = get_logger("dataversion")

_VERSIONS = "metadata/dataset_versions.json"
_OBJECTS = "data_versions/objects"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan(src: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(src.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(src))] = _hash_file(p)
    return out


def load_versions(root: Path) -> List[dict]:
    p = Path(root) / _VERSIONS
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_versions(root: Path, versions: List[dict]) -> None:
    p = Path(root) / _VERSIONS
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(versions, indent=2))


def commit(root: Path, which: str = "clean", label: str = "") -> dict:
    """Hash ``data/<which>``, dedupe into the blob store, record a version."""
    root = Path(root)
    src = root / "data" / which
    if not src.exists():
        console.error(f"No data/{which} dir to version.")
        return {"error": f"missing data/{which}"}

    files = _scan(src)
    store = root / _OBJECTS
    n_stored = 0
    seen = set()
    for rel, h in files.items():
        if h in seen:
            continue
        seen.add(h)
        obj = store / h[:2] / h
        if not obj.exists():
            obj.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, obj)
            n_stored += 1

    versions = load_versions(root)
    version = {
        "name": f"v{len(versions) + 1}",
        "label": label or f"data/{which}",
        "which": which,
        "at": datetime.now(timezone.utc).isoformat(),
        "n_files": len(files),
        "files": files,
    }
    versions.append(version)
    _save_versions(root, versions)
    console.ok(
        f"Version {version['name']} -> data/{which} "
        f"({len(files)} files, {n_stored} new blob(s) stored)"
    )
    return {"version": version, "n_stored": n_stored}


def _resolve(root: Path, ref: str) -> Optional[dict]:
    versions = load_versions(root)
    for v in versions:
        if v["name"] == ref or v.get("label") == ref:
            return v
    return None


def diff(root: Path, v1: str, v2: str) -> dict:
    a = _resolve(root, v1)
    b = _resolve(root, v2)
    if a is None or b is None:
        console.error(f"Unknown version ref(s): {v1}={a is not None}, {v2}={b is not None}")
        return {}
    fa, fb = a["files"], b["files"]
    added = sorted(set(fb) - set(fa))
    removed = sorted(set(fa) - set(fb))
    changed = sorted(k for k in set(fa) & set(fb) if fa[k] != fb[k])
    out = {
        "v1": a["name"], "v2": b["name"],
        "added": added, "removed": removed, "changed": changed,
        "n_added": len(added), "n_removed": len(removed), "n_changed": len(changed),
    }
    console.ok(
        f"diff {a['name']} -> {b['name']}: +{len(added)} -{len(removed)} ~{len(changed)}"
    )
    return out


def rollback(root: Path, version: str) -> dict:
    """Restore ``data/<which>`` to a prior version from the blob store."""
    v = _resolve(root, version)
    if v is None:
        console.error(f"Unknown version {version!r}.")
        return {"error": "unknown version"}
    root = Path(root)
    store = root / _OBJECTS
    dst = root / "data" / v["which"]
    dst.mkdir(parents=True, exist_ok=True)
    restored = 0
    missing = []
    for rel, h in v["files"].items():
        obj = store / h[:2] / h
        if not obj.exists():
            missing.append(rel)
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(obj, target)
        restored += 1
    console.ok(f"Rollback {v['name']} -> data/{v['which']} ({restored} restored)")
    if missing:
        console.warn(f"{len(missing)} blob(s) missing from store: {missing[:5]}")
    return {"restored": restored, "missing": missing, "version": v["name"]}
