"""Prompt-set analytics (advanced #6-#9).

* **Prompt difficulty** — ranks eval prompts by how hard they are for the
  current checkpoint (low CLAP adherence + large BPM drift + rejected
  verdict = hard). Writes metadata/prompt_difficulty.json.
* **Section x BPM interaction** — does BPM fidelity depend on the section?
  Produces a per-section table of mean |dev|, mean CLAP, ok-rate and the
  BPM->|dev| correlation.
* **CLAP z-scores** — standardizes CLAP scores within each section so
  cross-section comparisons aren't biased by section difficulty.
* **Auto-reject calibration** — inspects the current ok/rejected split and
  suggests threshold values for max_abs_deviation / min_clap_score based on
  where the good prompts actually sit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import console
from .config import Config
from .report import load_results


# --------------------------------------------------------------------------- #
# Prompt difficulty (#6)
# --------------------------------------------------------------------------- #
def prompt_difficulty(rows: List[dict]) -> List[dict]:
    """Score each prompt: higher = harder for the current checkpoint.

    difficulty = 0.45 * (1 - clap) + 0.35 * min(|dev|/0.5, 1) + 0.20 * rejected
    """
    scored: List[dict] = []
    for r in rows:
        clap = r.get("clap_score")
        dev = r.get("deviation")
        clap_t = 1.0 - float(clap) if clap is not None else 1.0
        dev_t = min(abs(float(dev)) / 0.5, 1.0) if dev is not None else 1.0
        rej_t = 1.0 if r.get("status") == "rejected" else 0.0
        score = round(0.45 * clap_t + 0.35 * dev_t + 0.20 * rej_t, 4)
        scored.append(
            {
                "prompt": r.get("prompt"),
                "section": r.get("section"),
                "bpm_target": r.get("bpm_target"),
                "clap_score": clap,
                "abs_deviation": round(abs(float(dev)), 4) if dev is not None else None,
                "status": r.get("status"),
                "difficulty": score,
            }
        )
    return sorted(scored, key=lambda s: -s["difficulty"])


# --------------------------------------------------------------------------- #
# Section x BPM interaction (#7)
# --------------------------------------------------------------------------- #
def section_bpm_interaction(rows: List[dict]) -> List[dict]:
    """Per-section adherence stats plus the BPM->|dev| correlation."""
    by_section: Dict[str, List[dict]] = {}
    for r in rows:
        by_section.setdefault(r.get("section") or "?", []).append(r)

    out: List[dict] = []
    for section, rs in sorted(by_section.items()):
        devs = [abs(float(r["deviation"])) for r in rs if r.get("deviation") is not None]
        claps = [float(r["clap_score"]) for r in rs if r.get("clap_score") is not None]
        bpm_dev = [
            (float(r["bpm_target"]), abs(float(r["deviation"])))
            for r in rs if r.get("bpm_target") and r.get("deviation") is not None
        ]
        corr = None
        if len(bpm_dev) >= 4:
            xs = np.asarray([p[0] for p in bpm_dev])
            ys = np.asarray([p[1] for p in bpm_dev])
            if xs.std() > 0 and ys.std() > 0:
                corr = round(float(np.corrcoef(xs, ys)[0, 1]), 4)
        out.append(
            {
                "section": section,
                "n": len(rs),
                "ok_rate": round(sum(1 for r in rs if r.get("status") == "ok") / len(rs), 4),
                "mean_abs_dev": round(float(np.mean(devs)), 4) if devs else None,
                "mean_clap": round(float(np.mean(claps)), 4) if claps else None,
                "bpm_dev_corr": corr,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# CLAP z-scores (#9)
# --------------------------------------------------------------------------- #
def clap_zscores(rows: List[dict]) -> List[dict]:
    """Per-section standardized CLAP: z = (clap - mean_section) / sd_section."""
    by_section: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get("clap_score") is None:
            continue
        by_section.setdefault(r.get("section") or "?", []).append(r)

    stats: Dict[str, Tuple[float, float]] = {}
    for sec, rs in by_section.items():
        vals = np.asarray([float(r["clap_score"]) for r in rs])
        stats[sec] = (float(vals.mean()), float(vals.std()))

    out: List[dict] = []
    for r in rows:
        sec = r.get("section") or "?"
        mu, sd = stats.get(sec, (0.0, 1.0))
        clap = r.get("clap_score")
        z = round((float(clap) - mu) / sd, 4) if clap is not None and sd > 0 else None
        out.append(
            {
                "prompt": r.get("prompt"),
                "section": sec,
                "clap_score": clap,
                "clap_z": z,
                "above_median": bool(z is not None and z > 0),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Auto-reject calibration (#8)
# --------------------------------------------------------------------------- #
def calibrate_thresholds(rows: List[dict]) -> dict:
    """Suggest max_abs_deviation / min_clap_score from the current ok split.

    Suggestion: keep the ok prompts, reject the worst 20% by each axis.
    Reports how many prompts each suggested rule would reject.
    """
    devs = sorted(
        abs(float(r["deviation"])) for r in rows
        if r.get("deviation") is not None and r.get("status") == "ok"
    )
    claps = sorted(
        float(r["clap_score"]) for r in rows
        if r.get("clap_score") is not None and r.get("status") == "ok"
    )
    n = len(rows)

    def _count_rejected(threshold: float, axis: str) -> int:
        if axis == "dev":
            return sum(1 for r in rows
                       if r.get("deviation") is not None and abs(float(r["deviation"])) > threshold)
        return sum(1 for r in rows
                   if r.get("clap_score") is not None and float(r["clap_score"]) < threshold)

    dev_sugg = round(float(np.percentile(devs, 90)), 4) if devs else None
    clap_sugg = round(float(np.percentile(claps, 10)), 4) if claps else None

    return {
        "n_prompts": n,
        "suggested_max_abs_deviation": dev_sugg,
        "suggested_min_clap_score": clap_sugg,
        "would_reject_dev": _count_rejected(dev_sugg, "dev") if dev_sugg is not None else None,
        "would_reject_clap": _count_rejected(clap_sugg, "clap") if clap_sugg is not None else None,
        "basis": "90th percentile of |dev| and 10th percentile of CLAP across current ok prompts",
    }


def mine_negatives(rows: List[dict], k: int = 10) -> List[dict]:
    """Advanced #17 — mine the weakest generations per section.

    Lowest-CLAP clips (with their audio paths) are the natural candidates for
    a hard-negative set (e.g. for contrastive audio-text training).
    """
    by_section: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get("clap_score") is None:
            continue
        by_section.setdefault(r.get("section") or "?", []).append(r)
    out: List[dict] = []
    for sec, rs in by_section.items():
        worst = sorted(rs, key=lambda r: r["clap_score"])[: max(k // max(len(by_section), 1), 1)]
        for r in worst:
            out.append(
                {
                    "section": sec,
                    "clap_score": r.get("clap_score"),
                    "audio_path": r.get("audio_path"),
                    "prompt": r.get("prompt"),
                    "candidate_for": "negative set",
                }
            )
    return sorted(out, key=lambda r: r["clap_score"])


def run(root: Path, cfg: Config) -> dict:
    """Compute all four analytics and write metadata/prompt_difficulty.json."""
    rows = load_results(root)
    if not rows:
        console.error("No eval results — run `musictrain eval` first.")
        return {}

    record = {
        "n_prompts": len(rows),
        "difficulty": prompt_difficulty(rows),
        "section_bpm_interaction": section_bpm_interaction(rows),
        "clap_zscores": clap_zscores(rows),
        "calibration": calibrate_thresholds(rows),
        "negative_candidates": mine_negatives(rows, k=10),
    }
    out = root / "metadata" / "prompt_difficulty.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    console.ok(f"Difficulty analytics -> {out.relative_to(root)}")

    top = record["difficulty"][:5]
    console.step("Hardest prompts:")
    for t in top:
        console.info(
            f"  {t['difficulty']:.2f}  [{t['section']}] {str(t['prompt'])[:70]}"
        )
    cal = record["calibration"]
    if cal.get("suggested_max_abs_deviation") is not None:
        console.ok(
            f"Suggested thresholds: |dev| <= {cal['suggested_max_abs_deviation']} "
            f"(rejects {cal['would_reject_dev']}) · CLAP >= {cal['suggested_min_clap_score']} "
            f"(rejects {cal['would_reject_clap']})"
        )
    return record
