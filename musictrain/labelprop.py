"""Semi-supervised labels + leakage check (Advanced #30).

Two utilities for label hygiene:

* **``propagate``** — pseudo-labels unlabeled tracks from their nearest *labeled*
  neighbors in the CLAP embedding space (label-weighted vote with confidence),
  writing ``metadata/pseudo_labels.json`` for a human to review/adopt.
* **``leakage_check``** — scans train/val/test (and any other ``data/<split>``
  dirs) for near-duplicate audio across splits using the chroma fingerprint, so
  no track silently appears in both training and eval.

Writes ``metadata/leakage.json``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import console
from .config import Config

_LABEL_DIMS = ("genre", "mood", "instruments")


def _labeled_rows(root: Path) -> Dict[str, dict]:
    p = root / "metadata" / "labels.csv"
    if not p.exists():
        return {}
    rows: Dict[str, dict] = {}
    for row in csv.DictReader(p.open(newline="")):
        sid = (row.get("source_id") or "").strip()
        if sid:
            rows[sid] = row
    return rows


def propagate(
    root: Path,
    cfg: Config,
    which: str = "clean",
    min_confidence: float = 0.55,
    top_k: int = 5,
) -> List[dict]:
    if not cfg.clap.enabled:
        console.warn("CLAP disabled (clap.enabled=false) — cannot propagate.")
        return []

    from .embeddings import embed_dir, _scan

    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return []

    labeled = _labeled_rows(root)
    index = embed_dir(root, cfg, which=which)
    if not labeled:
        console.warn("No labeled rows in metadata/labels.csv — nothing to propagate from.")
        return []

    labeled_vecs: List[Tuple[str, np.ndarray]] = []
    for sid, row in labeled.items():
        for rel, e in index.items():
            if Path(rel).stem == sid:
                labeled_vecs.append((sid, e / (np.linalg.norm(e) + 1e-12)))
                break

    if not labeled_vecs:
        console.warn("No labeled tracks have embeddings — run `musictrain embed` first.")
        return []

    L = np.stack([v for _, v in labeled_vecs])
    names = [n for n, _ in labeled_vecs]

    pseudo: List[dict] = []
    for p in _scan(target):
        rel = str(p.relative_to(root))
        sid = Path(rel).stem
        if sid in labeled or rel not in index:
            continue
        e = index[rel]
        e = e / (np.linalg.norm(e) + 1e-12)
        sims = L @ e
        order = np.argsort(sims)[::-1][:top_k]

        votes: Dict[str, Dict[str, float]] = {}
        total = 0.0
        for idx in order:
            w = float(np.clip(sims[idx], 0, 1))
            if w <= 0:
                continue
            row = labeled[names[idx]]
            for dim in _LABEL_DIMS:
                val = (row.get(dim) or "").strip()
                if not val:
                    continue
                votes.setdefault(dim, {})
                votes[dim][val] = votes[dim].get(val, 0.0) + w
                total += w

        if not votes:
            continue
        best = {dim: max(v.items(), key=lambda kv: kv[1]) for dim, v in votes.items()}
        confidence = min(
            (b[1] / max(total, 1e-9) if total > 0 else 0.0) for b in best.values()
        )
        if confidence < min_confidence:
            continue

        pseudo.append(
            {
                "path": rel,
                "confidence": round(float(confidence), 4),
                "labels": {dim: b[0] for dim, b in best.items()},
            }
        )

    pseudo.sort(key=lambda r: r["confidence"], reverse=True)
    out = root / "metadata" / "pseudo_labels.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pseudo, indent=2))
    console.ok(
        f"Propagated {len(pseudo)} pseudo-label(s) (conf>={min_confidence}) "
        f"-> metadata/pseudo_labels.json"
    )
    for r in pseudo[:10]:
        console.info(f"{r['confidence']:.3f}  {r['path']}  {r['labels']}")
    return pseudo


# --------------------------------------------------------------------------- #
# Leakage check
# --------------------------------------------------------------------------- #


def _scan_dir(root: Path, d: str) -> List[Path]:
    from .audio.inventory import AUDIO_GLOB

    target = root / "data" / d
    if not target.exists():
        return []
    files: List[Path] = []
    for pattern in AUDIO_GLOB:
        files.extend(sorted(target.glob(pattern)))
    return sorted(set(files))


def leakage_check(root: Path, cfg: Config, splits: List[str] | None = None) -> Dict[str, object]:
    from .dedup import chroma_fingerprint, _pitch_invariant_sim

    chosen = splits or ["train", "val", "test"]
    available = [d for d in chosen if (root / "data" / d).exists()]
    if not available:
        console.error("No split dirs found — expected one of: " + ", ".join(chosen))
        return {}

    fps: Dict[str, Tuple[str, np.ndarray]] = {}
    for d in available:
        for p in _scan_dir(root, d):
            try:
                fps[str(p.relative_to(root))] = (d, chroma_fingerprint(p))
            except Exception as exc:  # noqa: BLE001
                console.warn(f"Fingerprint failed {p.name}: {exc}")

    if len(fps) < 2:
        console.warn("Need at least 2 files across splits to check leakage.")
        return {}

    rels = list(fps.keys())
    leaks: List[dict] = []
    seen: set = set()
    for i, a in enumerate(rels):
        for b in rels[i + 1 :]:
            if fps[a][0] == fps[b][0]:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            sim = _pitch_invariant_sim(fps[a][1], fps[b][1])
            if sim >= cfg.dedup.threshold:
                seen.add(key)
                leaks.append(
                    {
                        "a": a,
                        "b": b,
                        "similarity": round(sim, 4),
                        "split_a": fps[a][0],
                        "split_b": fps[b][0],
                    }
                )

    report = {
        "splits": available,
        "files_checked": len(fps),
        "cross_split_duplicates": len(leaks),
        "leaks": leaks,
        "at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    out = root / "metadata" / "leakage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    if leaks:
        console.warn(
            f"⚠ {len(leaks)} cross-split near-duplicate(s) found -> metadata/leakage.json"
        )
        for l in leaks[:10]:
            console.warn(f"  {l['split_a']}/{l['a']} ~ {l['split_b']}/{l['b']} ({l['similarity']:.3f})")
    else:
        console.ok(f"No cross-split leakage across {available} (checked {len(fps)} files).")
    return report
