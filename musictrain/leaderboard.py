"""Checkpoint leaderboard from eval results (#45).

Ranks every checkpoint present in metadata/eval_results.jsonl by a composite
of prompt adherence (CLAP), BPM fidelity (|deviation|), and verdict share,
with a per-tag CLAP breakdown (#46) and mean human rating merged from
metadata/human_ratings.jsonl (#42). Writes metadata/leaderboard.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from . import console
from .config import Config
from .report import load_results

TAGS = ("section", "genre", "key", "mood", "instruments", "bpm")


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _pct_ok(rows: List[dict]) -> float:
    ok = sum(1 for r in rows if r.get("status") == "ok")
    return round(ok / len(rows), 4) if rows else 0.0


def _human_ratings(root: Path) -> Dict[str, List[float]]:
    """Merge metadata/human_ratings.jsonl (prompt -> ratings)."""
    path = root / "metadata" / "human_ratings.jsonl"
    if not path.exists():
        return {}
    out: Dict[str, List[float]] = {}
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("rating") is not None:
            out.setdefault(r.get("prompt", ""), []).append(float(r["rating"]))
    return out


def build(cfg: Config) -> dict:
    rows = load_results(cfg.project_root)
    if not rows:
        console.error("No eval results — run `musictrain eval` first.")
        return {}

    human = _human_ratings(cfg.project_root)
    by_checkpoint: Dict[str, List[dict]] = {}
    for r in rows:
        by_checkpoint.setdefault(r.get("checkpoint") or "(unknown)", []).append(r)

    entries: List[dict] = []
    for checkpoint, cres in sorted(by_checkpoint.items()):
        claps = [r["clap_score"] for r in cres if r.get("clap_score") is not None]
        devs = [abs(r["deviation"]) for r in cres if r.get("deviation") is not None]
        mean_clap = _mean(claps)
        mean_abs_dev = _mean(devs)
        ok_pct = _pct_ok(cres)

        # per-tag CLAP (#46): mean over rows of each tag's score
        per_tag: Dict[str, Optional[float]] = {}
        for tag in TAGS:
            per_tag[tag] = _mean(
                (r.get("clap_per_tag") or {}).get(tag) for r in cres
            )

        # human ratings (#42)
        ratings = []
        for r in cres:
            ratings.extend(human.get(r.get("prompt", ""), []))
        mean_human = _mean(ratings)

        # composite 0..1: ok-share 40%, CLAP 30%, deviation-fidelity 30%
        fidelity = max(0.0, 1.0 - (mean_abs_dev / 0.20)) if mean_abs_dev is not None else 0.0
        score = round(0.4 * ok_pct + 0.3 * (mean_clap or 0.0) + 0.3 * fidelity, 4)

        # bootstrap CI around the composite score (advanced eval #9)
        score_ci = None
        if len(cres) >= 3:
            from .adherence import bootstrap_score_ci

            score_ci = bootstrap_score_ci(cres)

        entries.append(
            {
                "checkpoint": checkpoint,
                "runs": len(cres),
                "ok_pct": ok_pct,
                "mean_clap": mean_clap,
                "mean_abs_deviation": mean_abs_dev,
                "mean_human_rating": mean_human,
                "clap_per_tag": per_tag,
                "score": score,
                "score_ci": score_ci,
            }
        )

    entries.sort(key=lambda e: e["score"], reverse=True)
    for i, e in enumerate(entries, 1):
        e["rank"] = i

    out = {"n_checkpoints": len(entries), "leaderboard": entries}
    path = cfg.project_root / "metadata" / "leaderboard.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))

    console.ok(f"Leaderboard -> {path.relative_to(cfg.project_root)}")
    for e in entries:
        console.ok(
            f"  #{e['rank']} {e['checkpoint']:<42} score {e['score']:.3f} "
            f"clap {e['mean_clap'] or 0:.3f} dev {e['mean_abs_deviation'] or 0:.3f} "
            f"ok {e['ok_pct']:.0%}"
        )
    return out
