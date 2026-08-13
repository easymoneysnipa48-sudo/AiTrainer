"""Corpus statistics: BPM/key/tag coverage histograms (Phase 1 #7).

Reads the feature manifest (metadata/manifest.jsonl) and reports coverage
distributions so you can see exactly where your dataset is dense or sparse
(e.g. "no 72 BPM bridges"). Writes metadata/corpus_stats.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import jsonlines

from . import console
from .config import Config


def _counts(values: List[Any]) -> Dict[str, int]:
    c: Counter = Counter()
    for v in values:
        if v is None:
            continue
        if isinstance(v, list):
            for x in v:
                c[str(x)] += 1
        else:
            c[str(v)] += 1
    return dict(c)


def _bpm_histogram(bpms: List[float], bins=None) -> Dict[str, int]:
    bins = bins or [0, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 200, 400]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    hist = {lab: 0 for lab in labels}
    for b in bpms:
        if b is None:
            continue
        for i in range(len(bins) - 1):
            if bins[i] <= b < bins[i + 1]:
                hist[labels[i]] += 1
                break
    return hist


def _load_manifest(root: Path) -> List[dict]:
    p = root / "metadata" / "manifest.jsonl"
    if not p.exists():
        return []
    return list(jsonlines.open(p))


def corpus(root: Path, cfg: Config, which: str = "clean") -> Dict:
    rows = _load_manifest(root)
    if not rows:
        console.warn("No manifest.jsonl — run `musictrain features` first for BPM/key/tag coverage.")
        return {}

    n = len(rows)
    durations = [r.get("duration") for r in rows if r.get("duration") is not None]
    bpms = [r.get("bpm") for r in rows if r.get("bpm") is not None]
    keys = [r.get("key") for r in rows if r.get("key")]

    stats = {
        "n_tracks": n,
        "total_duration_s": round(sum(durations), 2),
        "bpm": {"n": len(bpms), "mean": round(sum(bpms) / len(bpms), 2) if bpms else None,
                "min": min(bpms) if bpms else None, "max": max(bpms) if bpms else None,
                "histogram": _bpm_histogram(bpms)},
        "key": _counts(keys),
        "genre": _counts([r.get("genre") for r in rows]),
        "mood": _counts([r.get("mood") for r in rows]),
        "instruments": _counts([r.get("instruments") for r in rows]),
        "sections": _counts([r.get("section") for r in rows]),
    }

    out = root / "metadata" / "corpus_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))

    console.ok(f"Corpus: {n} tracks, {round(sum(durations), 1)}s total -> metadata/corpus_stats.json")
    if bpms:
        console.info(f"BPM: mean {stats['bpm']['mean']} · range {stats['bpm']['min']}–{stats['bpm']['max']}")
    for dim in ("key", "genre", "mood", "instruments", "sections"):
        if stats[dim]:
            top = ", ".join(f"{k}×{v}" for k, v in sorted(stats[dim].items(), key=lambda kv: -kv[1])[:8])
            console.info(f"{dim.capitalize()}: {top}")
    return stats
