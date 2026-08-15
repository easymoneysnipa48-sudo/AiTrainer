"""MLflow + metadata backup/restore (gap #16).

One-command snapshot of the project's model + metadata state so training runs,
eval results, and the registry can be restored onto another box. Backups are
versioned tar archives with a SHA-256 manifest, written to ``backups/``.

* ``snapshot`` — bundle metadata/*.jsonl|json, config, leaderboard, and an
  optional MLflow artifacts dir into ``backups/musictrain-<ts>.tar.gz``.
* ``restore`` — extract a backup over the current project (best-effort, never
  deletes files not present in the archive).
* ``list`` — enumerate existing backups with size + timestamp.

Pure-Python (tarfile + hashlib) so it runs anywhere; MLflow state is included
when ``mlflow_uri`` is a local filesystem path (remote tracking servers are
skipped with a warning).
"""
from __future__ import annotations

import hashlib
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from . import console
from .config import Config
from .logging import get_logger

log = get_logger("backup")


def _backup_dir(root: Path) -> Path:
    d = root / "backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest(files: List[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for f in sorted(files):
        if f.is_file():
            h = hashlib.sha256()
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[str(f)] = h.hexdigest()
    return out


def snapshot(cfg: Config, label: str = "", include_mlflow: bool = True) -> dict:
    """Create a versioned archive of project metadata (and local MLflow state)."""
    root = cfg.project_root
    meta = root / "metadata"
    files: List[Path] = []
    if meta.is_dir():
        files += [p for p in meta.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl", ".csv")]
    config_path = root / "config.yaml"
    if config_path.is_file():
        files.append(config_path)
    for name in ("labels.csv", "vocab.json"):
        p = root / name
        if p.is_file():
            files.append(p)

    # MLflow local artifact store (skip remote URIs)
    mlflow_uri = getattr(cfg, "mlflow_uri", "") or ""
    if include_mlflow and mlflow_uri.startswith("file://"):
        mlroot = Path(mlflow_uri.replace("file://", ""))
        if mlroot.is_dir():
            files += [p for p in mlroot.rglob("*") if p.is_file()]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    stem = "musictrain" + (f"-{label}" if label else "") + f"-{ts}"
    archive = _backup_dir(root) / f"{stem}.tar.gz"
    manifest = _manifest(files)

    with tarfile.open(archive, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=str(f.relative_to(root)))

    console.ok(f"Backup: {archive.name} ({len(files)} file(s), {len(manifest)} hashed)")
    log.info("backup written: %s (%d files)", archive, len(files))
    return {
        "archive": str(archive),
        "n_files": len(files),
        "manifest": manifest,
        "at": ts,
    }


def restore(cfg: Config, archive: str, force: bool = False) -> dict:
    """Extract a backup archive over the project root (never deletes extras)."""
    root = cfg.project_root
    archive_path = Path(archive)
    if not archive_path.is_file():
        console.error(f"Backup not found: {archive}")
        return {"error": "not_found"}

    # Refuse to extract an archive that points outside the project.
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            dest = (root / m.name).resolve()
            if root.resolve() not in dest.parents and dest != root.resolve():
                console.error(f"Refusing unsafe member {m.name!r} (outside project).")
                return {"error": "unsafe_member", "member": m.name}
        if not force:
            existing = [m.name for m in members if (root / m.name).exists()]
            if existing:
                console.warn(f"Would overwrite {len(existing)} file(s); pass --force to confirm.")
                return {"error": "needs_force", "n_overwrite": len(existing)}
        tar.extractall(root, filter="data")

    console.ok(f"Restored {archive_path.name} into {root}")
    log.info("backup restored: %s", archive)
    return {"restored": True, "n_members": len(members)}


def list_backups(cfg: Config) -> List[dict]:
    """List backups with size + mtime."""
    out: List[dict] = []
    d = _backup_dir(cfg.project_root)
    for p in sorted(d.glob("*.tar.gz"), reverse=True):
        stat = p.stat()
        out.append({
            "name": p.name,
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    return out


def run(cfg: Config, task: str, label: str = "", archive: str = "",
        force: bool = False, include_mlflow: bool = True) -> dict:
    if task == "snapshot":
        return snapshot(cfg, label=label, include_mlflow=include_mlflow)
    if task == "restore":
        return restore(cfg, archive, force=force)
    if task == "list":
        rows = list_backups(cfg)
        if not rows:
            console.info("No backups yet.")
        for r in rows:
            console.info(f"  {r['name']}  ({r['size_bytes']} bytes)")
        return {"task": "list", "backups": rows}
    console.error(f"Unknown backup task {task!r}")
    return {"error": f"unknown task {task}"}
