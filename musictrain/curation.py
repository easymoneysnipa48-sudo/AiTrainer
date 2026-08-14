"""Curation score (Advanced #28).

Ranks every track in the dataset with a single 0–100 score that blends the
signals already computed by other tools:

* **quality** — from ``metadata/quality.json`` (CLAP text adherence, loudness)
* **novelty** — embedding distance to the corpus centroid (from the CLAP cache)
* **duplication** — penalty for being a non-canonical duplicate
* **coverage** — bonus for under-represented genres (balanced corpus)

Higher is better. Writes ``metadata/curation_scores.json`` with per-track
breakdowns plus the composite.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import console
from .config import Config


def _load_manifest_rows(root: Path) -> List[dict]:
    """manifest.jsonl may be a single JSON object, a JSON array, or JSONL."""
    p = root / "metadata" / "manifest.jsonl"
    if not p.exists():
        return []
    text = p.read_text()
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        data = None
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        return [data]
    out: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _load(root: Path, name: str):
    p = root / "metadata" / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _genres_from_labels(root: Path) -> Dict[str, str]:
    import csv

    p = root / "metadata" / "labels.csv"
    if not p.exists():
        return {}
    out: Dict[str, str] = {}
    for row in csv.DictReader(p.open(newline="")):
        sid = (row.get("source_id") or "").strip()
        g = (row.get("genre") or "").strip()
        if sid and g:
            out[sid] = g
    return out


def _dup_penalty(root: Path) -> Dict[str, float]:
    """rel path -> 0.0 if canonical, 0.25 if duplicate member."""
    dup = _load(root, "duplicates.json") or {}
    penalty: Dict[str, float] = {}
    for group in dup.get("groups", []):
        members = group.get("members", [])
        if not members:
            continue
        for rel in members[1:]:
            penalty[rel] = 0.25
        if members:
            penalty[members[0]] = 0.0
    return penalty


def curation_score(
    root: Path,
    cfg: Config,
    which: str = "clean",
    top_k: int = 0,
) -> List[dict]:
    quality = _load(root, "quality.json") or {}
    manifest = _load_manifest_rows(root)
    embeddings = _load(root, "audio_embeddings.json") or {}
    genres = _genres_from_labels(root)
    dup_penalty = _dup_penalty(root)

    # corpus centroid from the embedding cache
    centroid = None
    vecs = [np.asarray(v["vec"], dtype=np.float32) if isinstance(v, dict) else np.asarray(v, dtype=np.float32) for v in embeddings.values() if v is not None]
    if vecs:
        stacked = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in vecs])
        centroid = stacked.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    # genre frequency for coverage bonus
    genre_counts: Dict[str, int] = {}
    for sid, g in genres.items():
        genre_counts[g] = genre_counts.get(g, 0) + 1

    quality_map: Dict[str, dict] = {}
    if isinstance(quality, dict) and "tracks" in quality:
        quality_map = quality["tracks"]
    elif isinstance(quality, list):
        quality_map = {q.get("path") or q.get("track"): q for q in quality if isinstance(q, dict)}

    rows: List[dict] = []
    for rec in manifest:
        rel = rec.get("path", "")
        if which != "clean" and f"data/{which}" not in rel:
            continue
        sid = Path(rel).stem

        # 0-1 quality signal
        q = quality_map.get(rel) or quality_map.get(sid) or {}
        clap = q.get("clap_score") or q.get("score")
        loudness = q.get("loudness")
        quality_sig = 0.6
        if clap is not None:
            quality_sig = 0.5 * float(np.clip(clap, 0, 1)) + 0.5 * quality_sig
        if loudness is not None:
            lufs = float(loudness)
            if -20 <= lufs <= -8:
                quality_sig = min(1.0, quality_sig + 0.1)
            elif lufs < -24:
                quality_sig = max(0.0, quality_sig - 0.2)

        # 0-1 novelty: distance from corpus centroid (1 = farthest/most novel)
        novelty = 0.5
        e = embeddings.get(rel)
        if e is not None and centroid is not None:
            vec = np.asarray(e["vec"] if isinstance(e, dict) else e, dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            novelty = float(np.clip(1.0 - vec @ centroid, 0, 1))

        # 0-1 duplication signal
        dup = dup_penalty.get(rel, 0.0)

        # 0-1 coverage bonus (rare genres get a lift)
        g = genres.get(sid, "")
        coverage = 0.5
        if g and genre_counts:
            frac = genre_counts[g] / max(1, sum(genre_counts.values()))
            coverage = float(np.clip(1.0 - frac, 0.3, 1.0))

        score = 100.0 * (0.45 * quality_sig + 0.25 * novelty + 0.20 * coverage - 0.10 * dup)
        score = round(float(np.clip(score, 0, 100)), 1)

        rows.append(
            {
                "path": rel,
                "score": score,
                "quality": round(quality_sig, 3),
                "novelty": round(novelty, 3),
                "coverage": round(coverage, 3),
                "dup_penalty": dup,
                "genre": g or None,
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    report = {
        "which": f"data/{which}",
        "scored": len(rows),
        "tracks": rows[:top_k] if top_k else rows,
        "at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    out = root / "metadata" / "curation_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    console.ok(f"Scored {len(rows)} track(s) -> metadata/curation_scores.json")
    for r in report["tracks"][:10]:
        console.info(f"{r['score']:5.1f}  {r['path']}")
    return report["tracks"]
