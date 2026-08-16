"""Reproducibility manifest (Phase 5 #38).

Every generation/eval run appends a compact JSONL record to
``metadata/repro_manifest.jsonl`` pinning the exact conditions it ran under:
config snapshot, vocabulary version, git commit (and dirty flag), model,
prompt, and the generation parameters. ``musictrain manifest`` lists recent
entries and can diff two of them, so an old result can be traced back to the
exact eval set / vocab / config / checkpoint that produced it.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config


def git_state(root: Path) -> Dict[str, Any]:
    """Current commit hash + dirty flag (best effort; empty when not a repo)."""
    try:
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        commit = rev.stdout.strip() if rev.returncode == 0 else ""
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return {"git_commit": commit, "git_dirty": bool(dirty.stdout.strip())}
    except Exception:  # noqa: BLE001 - git may be unavailable
        return {"git_commit": "", "git_dirty": True}


def _snapshot_cfg(cfg: Config) -> Dict[str, Any]:
    data = cfg.to_dict()
    data.pop("project_root", None)
    return data


def capture_run(cfg: Config, kind: str, extra: Optional[dict] = None, manifest: bool = True) -> Dict[str, Any]:
    """Append a manifest entry and return it. No-op (still returns entry) when manifest=False."""
    from .labels import VOCAB_VERSION

    entry: Dict[str, Any] = {
        "kind": kind,
        "at": datetime.now(timezone.utc).isoformat(),
        "vocab_version": VOCAB_VERSION,
        "model": cfg.inference.model_name,
        **git_state(cfg.project_root),
        "config": _snapshot_cfg(cfg),
    }
    if extra:
        entry.update(extra)

    if manifest:
        p = cfg.project_root / "metadata" / "repro_manifest.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
    return entry


def load_entries(root: Path) -> List[dict]:
    p = root / "metadata" / "repro_manifest.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def latest(root: Path, n: int = 1) -> List[dict]:
    return load_entries(root)[-n:]


def diff(a: dict, b: dict) -> List[str]:
    """Human-readable list of differences between two manifest entries."""
    lines: List[str] = []
    for key in ("kind", "model", "vocab_version", "git_commit", "git_dirty"):
        if a.get(key) != b.get(key):
            lines.append(f"{key}: {a.get(key)!r} -> {b.get(key)!r}")
    ca, cb = a.get("config", {}), b.get("config", {})
    inf_a, inf_b = ca.get("inference", {}), cb.get("inference", {})
    for key in ("preset", "temperature", "top_k", "top_p", "guidance_scale",
                "max_new_tokens", "target_seconds", "negative_prompt", "seed"):
        if inf_a.get(key) != inf_b.get(key):
            lines.append(f"inference.{key}: {inf_a.get(key)!r} -> {inf_b.get(key)!r}")
    # per-run extras are merged at the top level by capture_run
    for key in ("prompt", "seed", "preset", "max_new_tokens", "target_seconds",
                "conditioned_on", "conditioning_kind", "negative_prompt",
                "negative_violation", "attempts", "duration"):
        if a.get(key) != b.get(key):
            lines.append(f"{key}: {a.get(key)!r} -> {b.get(key)!r}")
    return lines or ["(no differences)"]
