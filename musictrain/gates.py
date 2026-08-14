"""Eval gates + drift detector + promotion reports (Advanced #32/#36/#37).

* **``eval_gate``** — blocks a candidate checkpoint from promotion when its eval
  aggregate regresses beyond tolerance vs a baseline checkpoint (CLAP drop or
  |deviation| increase). Exit code 1 = block, 0 = pass.
* **``drift_detector``** — wraps the drift report into an automated check that
  fails (non-zero) when features drift past the threshold — hook it into CI.
* **``promotion_report``** — renders a markdown promotion bundle: leaderboard
  rank, significance vs baseline, registry info, and eval coverage.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import console
from .config import Config
from .leaderboard import build as build_leaderboard
from .report import load_results


def _aggregate(rows: List[dict]) -> Dict[str, float]:
    claps = [r["clap_score"] for r in rows if r.get("clap_score") is not None]
    devs = [abs(r["deviation"]) for r in rows if r.get("deviation") is not None]
    ok = sum(1 for r in rows if r.get("status") == "ok")
    return {
        "n": len(rows),
        "mean_clap": (sum(claps) / len(claps)) if claps else 0.0,
        "mean_abs_deviation": (sum(devs) / len(devs)) if devs else float("inf"),
        "ok_pct": (ok / len(rows)) if rows else 0.0,
    }


def eval_gate(
    root: Path,
    cfg: Config,
    baseline: str,
    candidate: str,
    max_clap_drop: float = 0.02,
    max_deviation_increase: float = 0.05,
) -> Dict[str, object]:
    """Return gate verdict; caller maps `passed: False` to exit code 1."""
    rows = load_results(root)
    if not rows:
        console.error("No eval results — run `musictrain eval` first.")
        return {"passed": False, "reason": "no eval results"}

    a = [r for r in rows if (r.get("checkpoint") or "").strip() == baseline]
    b = [r for r in rows if (r.get("checkpoint") or "").strip() == candidate]
    if not a or not b:
        console.error(
            f"Need eval rows for both checkpoints (found baseline={len(a)}, "
            f"candidate={len(b)}) in metadata/eval_results.jsonl."
        )
        return {"passed": False, "reason": "missing checkpoint rows"}

    agg_a, agg_b = _aggregate(a), _aggregate(b)
    clap_drop = agg_a["mean_clap"] - agg_b["mean_clap"]
    dev_increase = agg_b["mean_abs_deviation"] - agg_a["mean_abs_deviation"]

    checks = [
        {
            "check": "clap_drop",
            "baseline": round(agg_a["mean_clap"], 4),
            "candidate": round(agg_b["mean_clap"], 4),
            "delta": round(clap_drop, 4),
            "limit": max_clap_drop,
            "blocking": clap_drop > max_clap_drop,
        },
        {
            "check": "deviation_increase",
            "baseline": round(agg_a["mean_abs_deviation"], 4),
            "candidate": round(agg_b["mean_abs_deviation"], 4),
            "delta": round(dev_increase, 4),
            "limit": max_deviation_increase,
            "blocking": dev_increase > max_deviation_increase,
        },
    ]
    passed = not any(c["blocking"] for c in checks)

    report = {
        "baseline": baseline,
        "candidate": candidate,
        "baseline_rows": len(a),
        "candidate_rows": len(b),
        "checks": checks,
        "passed": passed,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    out = root / "metadata" / "eval_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    for c in checks:
        flag = "BLOCK" if c["blocking"] else "ok"
        console.info(
            f"{c['check']:22s} {c['baseline']:>7.3f} -> {c['candidate']:>7.3f} "
            f"(limit {c['limit']})  [{flag}]"
        )
    if passed:
        console.ok(f"Gate PASSED — {candidate} may proceed.")
    else:
        console.warn(f"Gate BLOCKED — {candidate} regressed vs {baseline}.")
    return report


def drift_detector(
    root: Path,
    cfg: Config,
    reference: str = "clean",
    current: str = "train",
    ks_threshold: float = 0.05,
    psi_threshold: float = 0.25,
) -> Dict[str, object]:
    """CI-friendly drift check — non-zero-exit when drift is detected."""
    from .drift import drift_report

    report = drift_report(root, cfg, reference=reference, current=current)
    if not report:
        return {"passed": False, "reason": "drift report unavailable"}

    drifted = list(report.get("drifted_features", []))
    # re-evaluate with the explicit thresholds (drift_report uses its own default)
    continuous = report.get("continuous", {})
    discrete = report.get("discrete", {})
    drifted = [f for f, d in continuous.items() if d["ks_pvalue"] < ks_threshold]
    drifted += [f for f, d in discrete.items() if d["psi"] > psi_threshold]

    result = {
        "reference": report["reference"],
        "current": report["current"],
        "drifted_features": drifted,
        "passed": not drifted,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    if drifted:
        console.warn(f"DRIFT DETECTED: {', '.join(drifted)} (fails CI gate)")
    else:
        console.ok("No drift detected — gate passes.")
    return result


def promotion_report(
    root: Path,
    cfg: Config,
    checkpoint: str,
    baseline: Optional[str] = None,
) -> Optional[Path]:
    """Render a markdown promotion report for a checkpoint."""
    rows = load_results(root)
    if not rows:
        console.error("No eval results to build a promotion report from.")
        return None

    lb = build_leaderboard(cfg)
    entry = next((e for e in lb.get("leaderboard", []) if e["checkpoint"] == checkpoint), None)
    cand_rows = [r for r in rows if (r.get("checkpoint") or "").strip() == checkpoint]

    lines = [
        f"# Promotion report — `{checkpoint}`",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Eval rows: {len(cand_rows)}",
    ]
    if entry:
        lines += [
            f"- Leaderboard rank: **#{entry['rank']}** of {lb.get('n_checkpoints', 1)}",
            f"- Composite score: **{entry['score']:.4f}**",
            f"- Mean CLAP: {entry['mean_clap'] or '—'}",
            f"- Mean |deviation|: {entry['mean_abs_deviation'] or '—'}",
            f"- OK-rate: {entry['ok_pct']:.0%}",
            f"- Mean human rating: {entry['mean_human_rating'] or '—'}",
        ]

    if baseline:
        from .significance import from_checkpoints

        sig = from_checkpoints(cfg, baseline, checkpoint)
        if sig:
            clap = (sig.get("metrics") or {}).get("clap_score") or {}
            dev = (sig.get("metrics") or {}).get("abs_deviation") or {}
            lines += [
                "",
                f"## vs baseline `{baseline}`",
                "",
                f"- Paired prompts: {sig.get('n_paired')}",
                f"- CLAP delta: {clap.get('delta')} (p={clap.get('p_value')}, "
                f"{clap.get('verdict')})",
                f"- |deviation| delta: {dev.get('delta')} (p={dev.get('p_value')}, "
                f"{dev.get('verdict')})",
            ]

    lines += ["", "## Section coverage", ""]
    from collections import Counter

    sections = Counter(r.get("section") or "?" for r in cand_rows)
    for sec, n in sections.most_common():
        lines.append(f"- {sec}: {n}")

    safe = checkpoint.replace("/", "_").replace("\\", "_")
    out = root / "metadata" / f"promotion_{safe}.md"
    out.write_text("\n".join(lines) + "\n")
    console.ok(f"Promotion report -> {out.relative_to(root)}")
    return out
