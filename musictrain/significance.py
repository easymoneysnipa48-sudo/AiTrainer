"""Statistical significance between two eval result sets (#44).

Pairs runs from two result files (or two checkpoints within
metadata/eval_results.jsonl) on their shared prompt ids and runs a paired
non-parametric test (Wilcoxon signed-rank) on CLAP scores and |deviation|.
Writes metadata/significance.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

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


# --------------------------------------------------------------------------- #
# Advanced #3 — bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap 95% CI for mean(y - x). Returns (lo, hi, mean)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3:
        return float(np.nan), float(np.nan), float(np.nan)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        xi = x[rng.integers(0, len(x), len(x))]
        yi = y[rng.integers(0, len(y), len(y))]
        diffs[i] = yi.mean() - xi.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4), round(float(diffs.mean()), 4)


# --------------------------------------------------------------------------- #
# Advanced #4 — Bayesian A/B (conjugate normal-normal on the pair differences)
# --------------------------------------------------------------------------- #
def bayesian_ab(x: np.ndarray, y: np.ndarray, prior_prec: float = 1e-3) -> dict:
    """Posterior over the mean pair-difference with a flat normal prior.

    Returns the posterior mean/sd and P(diff > 0) — the probability that B is
    better than A on this metric (in the raw diff direction, so callers
    should flip for "higher is better" metrics).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = y - x
    n = len(d)
    if n < 2:
        return {"n": n, "p_b_over_a": None, "post_mean": None, "post_sd": None}
    s2 = d.var(ddof=1)
    post_prec = prior_prec + n / max(s2, 1e-12)
    post_mean = (n / max(s2, 1e-12)) * d.mean() / post_prec
    post_sd = np.sqrt(1.0 / post_prec)
    from scipy.stats import norm

    p = norm.cdf(0.0, loc=post_mean, scale=post_sd)  # P(diff <= 0)
    return {
        "n": n,
        "p_b_over_a": round(float(1.0 - p), 4),
        "post_mean": round(float(post_mean), 4),
        "post_sd": round(float(post_sd), 4),
    }


# --------------------------------------------------------------------------- #
# Advanced #5 — fixed-effects meta-analysis across experiments
# --------------------------------------------------------------------------- #
def meta_analyze(studies: List[dict]) -> dict:
    """Inverse-variance pooled delta across independent experiments.

    Each study: {"delta": float, "se": float or None, "label": str}. When a
    study lacks a standard error it is excluded from pooling (reported as
    such). Returns the pooled delta, its SE, z and a two-sided p-value.
    """
    from scipy.stats import norm

    deltas, ses, labels = [], [], []
    for s in studies:
        se = s.get("se")
        if se is None or not np.isfinite(se) or se <= 0 or s.get("delta") is None:
            continue
        deltas.append(float(s["delta"]))
        ses.append(float(se))
        labels.append(s.get("label", "?"))
    if not deltas:
        return {"pooled_delta": None, "se": None, "z": None, "p_value": None,
                "n_studies": len(studies), "n_pooled": 0}
    w = [1.0 / se2 for se2 in ses]
    pooled = sum(d * wi for d, wi in zip(deltas, w)) / sum(w)
    se_pooled = np.sqrt(1.0 / sum(w))
    z = pooled / se_pooled
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return {
        "pooled_delta": round(float(pooled), 4),
        "se": round(float(se_pooled), 4),
        "z": round(float(z), 4),
        "p_value": round(float(p), 4),
        "n_studies": len(studies),
        "n_pooled": len(deltas),
        "pooled_labels": labels,
    }


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
        lo, hi, boot_mean = bootstrap_ci(ca, cb)
        bay = bayesian_ab(ca, cb)  # P(B > A) directly (raw diff direction)
        out["metrics"]["clap_score"] = {
            "mean_a": round(float(ca.mean()), 4),
            "mean_b": round(float(cb.mean()), 4),
            "delta": round(float(cb.mean() - ca.mean()), 4),
            "direction": "higher is better",
            "statistic": stat,
            "p_value": round(p, 4) if p is not None else None,
            "verdict": _verdict(p, alpha, -(cb.mean() - ca.mean())),
            "bootstrap_95ci": [lo, hi],
            "bootstrap_mean": boot_mean,
            "bayesian": bay,
        }

    # |deviation|: lower is better -> positive mean delta = improvement
    da, db = _metric_pairs(a_rows, b_rows, "deviation")
    if len(da):
        aa, ab = np.abs(da), np.abs(db)
        stat, p = wilcoxon(aa, ab)
        lo, hi, boot_mean = bootstrap_ci(aa, ab)
        bay = bayesian_ab(aa, ab)  # raw diff = B - A; lower is better -> flip
        bay = {**bay, "p_b_over_a": round(1.0 - bay["p_b_over_a"], 4) if bay["p_b_over_a"] is not None else None}
        out["metrics"]["abs_deviation"] = {
            "mean_a": round(float(aa.mean()), 4),
            "mean_b": round(float(ab.mean()), 4),
            "delta": round(float(ab.mean() - aa.mean()), 4),
            "direction": "lower is better",
            "statistic": stat,
            "p_value": round(p, 4) if p is not None else None,
            "verdict": _verdict(p, alpha, ab.mean() - aa.mean()),
            "bootstrap_95ci": [lo, hi],
            "bootstrap_mean": boot_mean,
            "bayesian": bay,
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
