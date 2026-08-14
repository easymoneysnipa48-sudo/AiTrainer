"""Duplicate & near-duplicate detection for dataset hygiene (Phase 1 #1).

Two layers:
  1. exact      — SHA-256 content hash (byte-identical files)
  2. perceptual — pitch/tempo-robust chroma fingerprint (catches the same loop
                  re-exported at a different pitch or tempo)

Emits metadata/duplicates.json and can optionally move non-canonical copies to
data/dupes/ so the training set has no hidden leakage.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import console
from .audio.inventory import AUDIO_GLOB
from .config import Config
from .util import sha256_file


def _scan(dir_path: Path) -> List[Path]:
    found: List[Path] = []
    for pattern in AUDIO_GLOB:
        found.extend(sorted(dir_path.glob(pattern)))
    return sorted(set(found))


def chroma_fingerprint(path: Path) -> np.ndarray:
    """12-dim mean CENS chroma (tempo-robust); pitch handled by circular shift."""
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    return chroma.mean(axis=1).astype(np.float32)


def _pitch_invariant_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(max(np.dot(a, np.roll(b, i)) for i in range(12)))


def find_duplicates(root: Path, cfg: Config, which: str = "clean") -> Dict:
    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return {}

    files = _scan(target)
    if not files:
        console.warn(f"No audio files under {target}")
        return {}

    # -- exact (content hash) -------------------------------------------------
    by_hash: Dict[str, List[Path]] = defaultdict(list)
    for p in files:
        try:
            by_hash[sha256_file(p)].append(p)
        except Exception:  # noqa: BLE001
            continue
    exact = [g for g in by_hash.values() if len(g) > 1]

    clusters: List[dict] = []
    seen: set = set()

    # -- perceptual ------------------------------------------------------------
    if cfg.dedup.exact_only:
        fp: Dict[Path, np.ndarray] = {}
    else:
        console.info("Computing chroma fingerprints…")
        fp = {}
        for p in files:
            try:
                fp[p] = chroma_fingerprint(p)
            except Exception as exc:  # noqa: BLE001
                console.warn(f"Fingerprint failed {p.name}: {exc}")

        # greedy clustering: representative = first file of each cluster
        reps: List[Path] = []
        for p in files:
            if p in fp:
                matched = None
                for rep in reps:
                    if _pitch_invariant_sim(fp[rep], fp[p]) >= cfg.dedup.threshold:
                        matched = rep
                        break
                if matched is None:
                    reps.append(p)
                else:
                    seen.add(p)

        # build clusters from representatives
        members: Dict[Path, List[Path]] = {rep: [] for rep in reps}
        for p in files:
            if p in fp:
                for rep in reps:
                    if rep == p or _pitch_invariant_sim(fp[rep], fp[p]) >= cfg.dedup.threshold:
                        members[rep].append(p)
                        break
        for rep, mem in members.items():
            if len(mem) > 1:
                clusters.append({"canonical": str(rep.relative_to(root)), "members": [str(m.relative_to(root)) for m in mem], "kind": "perceptual"})

    # exact clusters (dedupe against perceptual, just report)
    for group in exact:
        rel = [str(p.relative_to(root)) for p in group]
        if not any(c["members"] == rel for c in clusters):
            clusters.append({"canonical": rel[0], "members": rel, "kind": "exact"})

    report = {
        "dir": f"data/{which}",
        "scanned": len(files),
        "duplicate_groups": len(clusters),
        "duplicate_files": sum(len(c["members"]) - 1 for c in clusters),
        "groups": clusters,
    }

    out = root / "metadata" / "duplicates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    console.ok(
        f"{report['duplicate_files']} duplicate(s) in {report['duplicate_groups']} group(s) "
        f"-> metadata/duplicates.json"
    )
    for c in clusters:
        console.warn(f"[{c['kind']}] keep {c['canonical']} — drop {len(c['members']) - 1}")

    if cfg.dedup.action == "move" and clusters:
        _move_dupes(root, clusters)
    return report


def dedup_segments(root: Path, cfg: Config, which: str = "segments") -> Dict:
    """Post-segment dedup (Advanced #25).

    Runs the same exact + perceptual pipeline over a segment directory (default
    ``data/segments``) — segments cut from the same bar are near-identical, so
    this catches redundant training examples before they enter the fine-tune set.
    """
    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return {}

    files = _scan(target)
    if not files:
        console.warn(f"No audio files under {target}")
        return {}

    console.step(f"Deduplicating {len(files)} segment(s) under data/{which}…")
    by_hash: Dict[str, List[Path]] = defaultdict(list)
    for p in files:
        try:
            by_hash[sha256_file(p)].append(p)
        except Exception:  # noqa: BLE001
            continue
    exact = [g for g in by_hash.values() if len(g) > 1]

    clusters: List[dict] = []
    if cfg.dedup.exact_only:
        fp: Dict[Path, np.ndarray] = {}
    else:
        fp = {}
        for p in files:
            try:
                fp[p] = chroma_fingerprint(p)
            except Exception as exc:  # noqa: BLE001
                console.warn(f"Fingerprint failed {p.name}: {exc}")
        reps: List[Path] = []
        for p in files:
            if p not in fp:
                continue
            if not any(_pitch_invariant_sim(fp[rep], fp[p]) >= cfg.dedup.threshold for rep in reps):
                reps.append(p)
        members: Dict[Path, List[Path]] = {rep: [] for rep in reps}
        for p in files:
            if p not in fp:
                continue
            for rep in reps:
                if rep == p or _pitch_invariant_sim(fp[rep], fp[p]) >= cfg.dedup.threshold:
                    members[rep].append(p)
                    break
        for rep, mem in members.items():
            if len(mem) > 1:
                clusters.append(
                    {
                        "canonical": str(rep.relative_to(root)),
                        "members": [str(m.relative_to(root)) for m in mem],
                        "kind": "perceptual",
                    }
                )

    for group in exact:
        rel = [str(p.relative_to(root)) for p in group]
        if not any(c["members"] == rel for c in clusters):
            clusters.append({"canonical": rel[0], "members": rel, "kind": "exact"})

    report = {
        "dir": f"data/{which}",
        "scanned": len(files),
        "duplicate_groups": len(clusters),
        "duplicate_files": sum(len(c["members"]) - 1 for c in clusters),
        "groups": clusters,
    }
    out = root / "metadata" / "segment_duplicates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    console.ok(
        f"{report['duplicate_files']} duplicate segment(s) -> metadata/segment_duplicates.json"
    )
    if cfg.dedup.action == "move" and clusters:
        _move_dupes(root, clusters, dest="segment_dupes")
    return report


def _move_dupes(root: Path, clusters: List[dict], dest: str = "dupes") -> None:
    dupes_dir = root / "data" / dest
    dupes_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for c in clusters:
        for rel in c["members"][1:]:
            src = root / rel
            if not src.exists():
                continue
            dst = dupes_dir / src.name
            if dst.exists():
                dst = dupes_dir / f"{src.stem}_{src.stat().st_size}{src.suffix}"
            shutil.move(str(src), str(dst))
            moved += 1
    console.ok(f"Moved {moved} duplicate(s) -> data/{dest}/")
