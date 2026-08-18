"""Remove transient/generated artifacts (safe list only).

Never touches user data: no ``data/raw|clean|segments``, no ``lyrics/``, no
``adapters/``, no ``checkpoints/``. It only clears:

* ``metadata/session.json`` — the dashboard's resume-state artifact (recreated
  on the next dashboard run)
* ``*.tmp`` / ``*.part`` / ``*.partial`` leftovers under ``metadata/``,
  ``outputs/`` and ``data/``
* rotated eval-result backups ``metadata/eval_results.jsonl.*`` beyond
  ``keep_eval`` newest
* rotated log files ``logs/*.log.*`` older than ``max_age_days``
* ``__pycache__`` directories

``plan()`` lists candidate entries with sizes; ``run()`` deletes them (unless
``dry_run``).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List, Tuple

from . import console

_TMP_SUFFIXES = (".tmp", ".part", ".partial", ".download", ".bak")
_IGNORED_DIRS = {'.venv', '.git', '.hg', '.svn', '.agents', '.claude', 'node_modules', '__pycache__'}


def _inside_ignored(p: Path, root: Path) -> bool:
    """True when p sits under an ignored dir (venv/git/agent tooling)."""
    try:
        rel = p.relative_to(root)
    except ValueError:
        return True
    return any(part in _IGNORED_DIRS for part in rel.parts)


def _dir_size(p: Path) -> int:
    total = 0
    try:
        for f in p.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total


def plan(root: Path, keep_eval: int = 5, max_age_days: float = 2.0) -> List[Tuple[Path, int]]:
    """Return [(path, bytes)] of removable artifacts, newest-first per kind."""
    items: List[Tuple[Path, int]] = []

    session = root / "metadata" / "session.json"
    if session.exists():
        items.append((session, session.stat().st_size))

    for base in ("metadata", "outputs", "data"):
        d = root / base
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in _TMP_SUFFIXES:
                try:
                    items.append((p, p.stat().st_size))
                except OSError:
                    pass

    # eval-result backups: keep the N newest siblings of eval_results.jsonl
    ev = root / "metadata" / "eval_results.jsonl"
    if ev.exists():
        backups = sorted(ev.parent.glob(ev.name + ".*"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        for p in backups[keep_eval:]:
            items.append((p, p.stat().st_size))

    # rotated logs older than max_age_days
    logs = root / "logs"
    if logs.exists():
        cutoff = time.time() - max_age_days * 86400
        for p in logs.glob("*.log.*"):
            try:
                if p.stat().st_mtime < cutoff:
                    items.append((p, p.stat().st_size))
            except OSError:
                pass

    for p in root.rglob("__pycache__"):
        if p.is_dir() and not _inside_ignored(p, root):
            items.append((p, _dir_size(p)))

    # dedupe by path, keep the largest of any dupes
    seen: dict = {}
    for p, sz in items:
        key = str(p)
        if key in seen:
            seen[key] = (p, max(sz, seen[key][1]))
        else:
            seen[key] = (p, sz)
    return list(seen.values())


def run(root: Path, dry_run: bool = False, keep_eval: int = 5,
        max_age_days: float = 2.0) -> dict:
    items = plan(root, keep_eval=keep_eval, max_age_days=max_age_days)
    if not items:
        console.ok("Nothing to clean — project is tidy.")
        return {"deleted": 0, "bytes_freed": 0}

    label = "Would remove" if dry_run else "Removing"
    console.step(f"{label} {len(items)} item(s)")
    total = 0
    for p, sz in items:
        total += sz
        tag = "[dir] " if p.is_dir() else ""
        console.info(f"  {tag}{p} ({sz / 1e6:.1f} MB)")
        if not dry_run:
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink(missing_ok=True)
            except OSError as exc:
                console.warn(f"  ! could not remove {p}: {exc}")
    if not dry_run:
        console.ok(f"Freed {total / 1e6:.1f} MB") if total else console.ok("Done.")
    return {"deleted": len(items), "bytes_freed": total}
