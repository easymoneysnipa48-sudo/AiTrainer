"""Versioned, hierarchical vocabulary tooling (Phase 4 #27, #32).

* ``render_tree``  — print the controlled vocabulary as a tree (parents with
  their children), so labelers see the structure at a glance (#27).
* ``migrate``      — rename terms across a labels CSV atomically, with a
  backup, and stamp ``metadata/vocab_version.json`` (#32). Handles both
  dimension-specific maps ({"genre": {"trap": "melodic trap"}}) and flat maps
  ({"trap": "melodic trap"}) applied to every vocabulary field.

The vocabulary itself lives in ``labels.py`` (VOCAB / HIERARCHY /
VOCAB_VERSION) — this module only renders and migrates it.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from . import console
from .labels import ENFORCED_FIELDS, HIERARCHY, VOCAB, VOCAB_VERSION, _split


def render_tree() -> str:
    """Render the vocabulary as a tree, e.g. ``instruments -> 808 bass -> sub bass``."""
    lines = [f"vocabulary v{VOCAB_VERSION}"]
    for dim in sorted(VOCAB):
        lines.append(f"{dim}:")
        parents = HIERARCHY.get(dim, {})
        child_terms = {c for children in parents.values() for c in children}
        roots = sorted(t for t in VOCAB[dim] if t not in child_terms)
        for i, term in enumerate(roots):
            last_root = i == len(roots) - 1
            if term in parents:
                lines.append(f"  {'└─' if last_root else '├─'} {term} (parent)")
                children = sorted(parents[term])
                for j, child in enumerate(children):
                    last_child = j == len(children) - 1
                    branch = "   " if last_root else "│  "
                    lines.append(f"  {branch}{'└─' if last_child else '├─'} {child}")
            else:
                lines.append(f"  {'└─' if last_root else '├─'} {term}")
    return "\n".join(lines)


def _map_for(rename_map: Dict[str, dict], dim: str) -> Dict[str, str]:
    """Dimension-specific map if present, else the flat map applied to all dims."""
    if dim in rename_map:
        return rename_map[dim]
    if not any(k in ENFORCED_FIELDS for k in rename_map):
        return rename_map
    return {}


def migrate(
    root: Path,
    labels_path: Path,
    rename_map_path: Path,
    backup: bool = True,
) -> Dict[str, object]:
    """Rename vocabulary terms in a labels CSV (with backup) and stamp the version."""
    labels_path = Path(labels_path)
    rename_map: Dict[str, dict] = json.loads(Path(rename_map_path).read_text())

    rows = list(csv.DictReader(labels_path.open(newline="")))
    if not rows:
        console.error(f"No rows in {labels_path}")
        return {"renames": 0}

    changed = 0
    for row in rows:
        for dim in ENFORCED_FIELDS:
            if dim not in row or not (row.get(dim) or "").strip():
                continue
            mapping = _map_for(rename_map, dim)
            if not mapping:
                continue
            items = _split(row[dim])
            new_items: List[str] = []
            for item in items:
                if item in mapping:
                    new_items.append(mapping[item])
                    changed += 1
                else:
                    new_items.append(item)
            if new_items != items:
                sep = "|" if "|" in row[dim] else "; " if "; " in row[dim] else ", "
                row[dim] = sep.join(new_items)

    if changed == 0:
        console.warn("No terms matched the rename map — nothing to migrate.")
        return {"renames": 0}

    if backup:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        bak = Path(f"{labels_path}.bak.{ts}")
        bak.write_bytes(labels_path.read_bytes())
        console.info(f"Backup -> {bak.relative_to(root)}")

    fieldnames = list(rows[0].keys())
    with labels_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stamp = {
        "from_version": VOCAB_VERSION,
        "to_version": VOCAB_VERSION + 1,
        "renames": changed,
        "at": datetime.now(timezone.utc).isoformat(),
        "map": rename_map,
    }
    vp = root / "metadata" / "vocab_version.json"
    vp.parent.mkdir(parents=True, exist_ok=True)
    vp.write_text(json.dumps(stamp, indent=2))
    console.ok(f"Migrated {changed} term(s) -> {labels_path.relative_to(root)}")
    console.info(f"Version stamp -> metadata/vocab_version.json (v{VOCAB_VERSION} -> v{VOCAB_VERSION + 1})")
    return stamp
