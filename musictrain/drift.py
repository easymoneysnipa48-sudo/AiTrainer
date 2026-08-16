"""Data drift monitoring (Advanced #27).

Compares the feature distribution of the *current* dataset against a
*reference* snapshot and flags features that drifted, so you notice when new
tracks shift the corpus (e.g. all-dark-key month) before training on them.

Uses the Kolmogorov–Smirnov test on continuous features (bpm, loudness,
duration, key confidence) and per-category frequency shift (PSI-style) on
discrete ones (key, genre). Writes ``metadata/drift.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import console
from .config import Config

_CONTINUOUS = ["bpm", "loudness", "duration", "key_confidence"]
_DISCRETE = ["key", "genre", "mood"]


def _load_records(root: Path, name: str) -> List[dict]:
    p = root / "metadata" / name
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return []


def _records_for(root: Path, which: str) -> List[dict]:
    """Gather feature rows from manifest.jsonl, filtered by data/<which>."""
    recs: List[dict] = []
    manifest = root / "metadata" / "manifest.jsonl"
    if not manifest.exists():
        return recs
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        path = str(r.get("path", ""))
        if which == "clean" or f"data/{which}" in path or f"data\\{which}" in path:
            recs.append(r)
    return recs


def _ks_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats

    if a.size < 2 or b.size < 2:
        return 1.0
    try:
        return float(stats.ks_2samp(a, b).pvalue)
    except Exception:  # noqa: BLE001
        return 1.0


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index for a continuous feature."""
    if reference.size < 2 or current.size < 2:
        return 0.0
    edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, bins=edges)[0].astype(float) / reference.size
    c = np.histogram(current, bins=edges)[0].astype(float) / current.size
    r = np.clip(r, 1e-6, None)
    c = np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _cat_shift(reference: List[str], current: List[str]) -> float:
    if not reference or not current:
        return 0.0
    cats = sorted(set(reference) | set(current))
    rf = np.array([reference.count(c) for c in cats], dtype=float) / len(reference)
    cf = np.array([current.count(c) for c in cats], dtype=float) / len(current)
    rf = np.clip(rf, 1e-6, None)
    cf = np.clip(cf, 1e-6, None)
    return float(np.sum((cf - rf) * np.log(cf / rf)))


def drift_report(
    root: Path,
    cfg: Config,
    reference: str = "clean",
    current: str = "train",
    threshold: float = 0.05,
) -> Dict[str, object]:
    ref_rows = _records_for(root, reference)
    cur_rows = _records_for(root, current)

    if not ref_rows or not cur_rows:
        console.error(
            f"Need feature rows for both data/{reference} and data/{current} "
            f"(metadata/manifest.jsonl). Run `musictrain features` first."
        )
        return {}

    continuous: Dict[str, dict] = {}
    for feat in _CONTINUOUS:
        rv = np.array([r.get(feat) for r in ref_rows if r.get(feat) is not None], dtype=float)
        cv = np.array([r.get(feat) for r in cur_rows if r.get(feat) is not None], dtype=float)
        if rv.size < 2 or cv.size < 2:
            continue
        p = _ks_pvalue(rv, cv)
        psi = _psi(rv, cv)
        continuous[feat] = {
            "reference_mean": round(float(rv.mean()), 4),
            "current_mean": round(float(cv.mean()), 4),
            "delta": round(float(cv.mean() - rv.mean()), 4),
            "ks_pvalue": round(p, 6),
            "psi": round(psi, 6),
            "drifted": p < threshold,
        }

    discrete: Dict[str, dict] = {}
    for feat in _DISCRETE:
        rv = [str(r.get(feat) or "unknown") for r in ref_rows]
        cv = [str(r.get(feat) or "unknown") for r in cur_rows]
        shift = _cat_shift(rv, cv)
        discrete[feat] = {"psi": round(shift, 6), "drifted": shift > 0.25}

    drifted_feats = [f for f, d in continuous.items() if d["drifted"]] + [
        f for f, d in discrete.items() if d["drifted"]
    ]

    report = {
        "reference": f"data/{reference}",
        "current": f"data/{current}",
        "reference_n": len(ref_rows),
        "current_n": len(cur_rows),
        "threshold": threshold,
        "continuous": continuous,
        "discrete": discrete,
        "drifted_features": drifted_feats,
        "at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    out = root / "metadata" / "drift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    console.ok(f"Drift report -> metadata/drift.json ({reference} vs {current})")
    for feat, d in continuous.items():
        flag = "DRIFT" if d["drifted"] else "ok"
        console.info(
            f"{feat:14s} {d['reference_mean']:>8.2f} -> {d['current_mean']:>8.2f}  "
            f"ks_p={d['ks_pvalue']:.4f} psi={d['psi']:.4f}  [{flag}]"
        )
    for feat, d in discrete.items():
        flag = "DRIFT" if d["drifted"] else "ok"
        console.info(f"{feat:14s} psi={d['psi']:.4f}  [{flag}]")
    if drifted_feats:
        console.warn(f"Drifted features: {', '.join(drifted_feats)}")
    else:
        console.ok("No drifted features detected.")
    return report
