"""Statistical significance between two eval result sets (#44).

Pairs runs from two result files (or two checkpoints within
metadata/eval_results.jsonl) on their shared prompt ids and runs a paired
non-parametric test (Wilcoxon signed-rank) on CLAP scores and |deviation|.
Writes metadata/significance.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import console
from .config import Config


def load_results(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _key(r: dict) -> str:
    """Stable pairing key: prompt + target BPM (prompts repeat across checkpoints)."""
    return f"{r.get('prompt')}|{r.get('bpm_target')}"


def _pairs(a_rows: List[dict], b_rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    bmap = {_key(r): r for r in b_rows}
    paired = [(r, bmap[_key(r)]) for r in a_rows if _key(r) in bmap]
    if not paired:
        return [], []
    return [p[0] for p in paired], [p[1] for p in paired]


def _metric_pairs(a_rows: List[dict], b_rows: List[dict], key: str):
    vals_a, vals_b = [], []
    for ra, rb in zip(a_rows, b_rows):
        va, vb = ra.get(key), rb.get(key)
        if va is None or vb is None:
            continue
        vals_a.append(va)
        vals_b.append(vb)
    return np.asarray(vals_a, dtype=float), np.asarray(vals_b, dtype=float)


def wilcoxon(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Paired Wilcoxon signed-rank -> (statistic, p-value). None if degenerate."""
    d = x - y
    nz = d[np.abs(d) > 1e-12]
    if len(nz) == 0:
        return None, 1.0  # every pair identical -> no difference, p = 1
    if len(nz) < 5:  # too few informative pairs for a meaningful test
        return None, None
    from scipy.stats import wilcoxon as _w

    try:
        stat, p = _w(d)
        return float(stat), float(p)
    except ValueError:
        return None, None


def _verdict(p: Optional[float], alpha: float, delta_mean: float) -> str:
    if p is None:
        return "insufficient data"
    if p >= alpha:
        return "no significant difference"
    return "improved" if delta_mean < 0 else "worsened"


def compare(cfg: Config, a_rows: List[dict], b_rows: List[dict],
            label_a: str = "A", label_b: str = "B") -> dict:
    alpha = cfg.eval.significance_alpha
    a_rows, b_rows = _pairs(a_rows, b_rows)
    if not a_rows:
        console.error("No shared prompts between the two result sets.")
        return {}

    out: dict = {
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": len(a_rows),
        "alpha": alpha,
        "metrics": {},
    }

    # CLAP: higher is better -> negative mean delta = improvement
    ca, cb = _metric_pairs(a_rows, b_rows, "clap_score")
    if len(ca):
        stat, p = wilcoxon(ca, cb)
        out["metrics"]["clap_score"] = {
            "mean_a": round(float(ca.mean()), 4),
            "mean_b": round(float(cb.mean()), 4),
            "delta": round(float(cb.mean() - ca.mean()), 4),
            "direction": "higher is better",
            "statistic": stat,
            "p_value": round(p, 4) if p is not None else None,
            "verdict": _verdict(p, alpha, -(cb.mean() - ca.mean())),
        }

    # |deviation|: lower is better -> positive mean delta = improvement
    da, db = _metric_pairs(a_rows, b_rows, "deviation")
    if len(da):
        aa, ab = np.abs(da), np.abs(db)
        stat, p = wilcoxon(aa, ab)
        out["metrics"]["abs_deviation"] = {
            "mean_a": round(float(aa.mean()), 4),
            "mean_b": round(float(ab.mean()), 4),
            "delta": round(float(ab.mean() - aa.mean()), 4),
            "direction": "lower is better",
            "statistic": stat,
            "p_value": round(p, 4) if p is not None else None,
            "verdict": _verdict(p, alpha, ab.mean() - aa.mean()),
        }

    out["summary"] = (
        f"{label_b} vs {label_a}: "
        + "; ".join(
            f"{k} {v['verdict']} (p={v['p_value']})" if v["p_value"] is not None
            else f"{k} {v['verdict']}"
            for k, v in out["metrics"].items()
        )
    )

    path = cfg.project_root / "metadata" / "significance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    console.ok(f"Significance -> {path.relative_to(cfg.project_root)}")
    return out


def from_checkpoints(cfg: Config, checkpoint_a: str, checkpoint_b: str) -> dict:
    """Compare two checkpoints inside metadata/eval_results.jsonl."""
    rows = load_results(cfg.project_root / "metadata" / "eval_results.jsonl")
    a = [r for r in rows if r.get("checkpoint") == checkpoint_a]
    b = [r for r in rows if r.get("checkpoint") == checkpoint_b]
    if not a or not b:
        console.error(f"Need rows for both checkpoints (found {len(a)} / {len(b)}).")
        return {}
    console.step(f"Pairing {len(a)} x {len(b)} rows by prompt…")
    return compare(cfg, a, b, label_a=checkpoint_a, label_b=checkpoint_b)
