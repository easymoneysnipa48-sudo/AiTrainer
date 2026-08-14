"""Training monitor + early stopping + experiment matrix + model card
(Advanced #33/#35/#39/#40).

* **``early_stop``** — decides when to halt fine-tuning based on the CLAP
  metric series (patience + minimum improvement), so long runs don't waste
  compute after the curve plateaus.
* **``training_monitor``** — summarizes MLflow eval/inference runs into a
  trend report (CLAP over time per checkpoint).
* **``experiment_matrix``** — rows = runs, columns = key metrics, so you can
  eyeball every experiment at once. Written to metadata/experiment_matrix.json.
* **``model_card``** — renders a markdown model card from the best run's
  metrics + eval aggregate, ready to drop into the repo/docs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import console
from .config import Config


# --------------------------------------------------------------------------- #
# #33 — early stopping on the CLAP series
# --------------------------------------------------------------------------- #
def early_stop(
    clap_series: List[float],
    patience: int = 3,
    min_delta: float = 0.005,
) -> Dict[str, object]:
    """Return whether to stop given the CLAP history.

    Stops when the best-so-far CLAP hasn't improved by >= min_delta for
    `patience` consecutive logged steps.
    """
    if not clap_series:
        return {"should_stop": False, "reason": "empty series"}
    best = max(clap_series)
    best_idx = clap_series.index(best)
    steps_since_best = len(clap_series) - 1 - best_idx
    should_stop = steps_since_best >= patience
    return {
        "should_stop": should_stop,
        "reason": (
            f"no >= {min_delta} improvement for {steps_since_best} step(s) "
            f"(best {best:.4f} at step {best_idx + 1})"
            if should_stop
            else "still improving or within patience"
        ),
        "best_clap": round(best, 4),
        "best_step": best_idx + 1,
        "steps_since_best": steps_since_best,
        "patience": patience,
        "min_delta": min_delta,
        "series": [round(x, 4) for x in clap_series],
    }


# --------------------------------------------------------------------------- #
# #39 — training monitor (MLflow trend)
# --------------------------------------------------------------------------- #
def training_monitor(cfg: Config, limit: int = 50) -> Dict[str, object]:
    """Summarize eval/inference runs from MLflow into a CLAP trend report."""
    from .experiments import search_runs

    df = search_runs(cfg)
    if df.empty:
        console.warn("No MLflow runs found — is tracking enabled and populated?")
        return {"runs": 0, "trend": []}

    out_rows: List[dict] = []
    for _, r in df.iterrows():
        clap = r.get("clap_score")
        if clap is None:
            continue
        out_rows.append(
            {
                "run_id": r.get("run_id"),
                "name": r.get("name"),
                "task": r.get("task"),
                "model": r.get("model"),
                "checkpoint": r.get("model"),
                "clap_score": round(float(clap), 4) if clap == clap else None,
                "deviation": r.get("deviation"),
                "seed": r.get("seed"),
            }
        )
    out_rows = out_rows[-limit:]

    # per-checkpoint trend: mean CLAP over time
    trend: Dict[str, List[dict]] = {}
    for r in out_rows:
        trend.setdefault(r["model"] or "?", []).append(
            {"clap_score": r["clap_score"], "deviation": r["deviation"],
             "seed": r["seed"], "run_id": r["run_id"]}
        )

    report = {
        "runs": len(out_rows),
        "trend": trend,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    out = cfg.project_root / "metadata" / "training_monitor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    console.ok(f"Training monitor -> metadata/training_monitor.json ({len(out_rows)} runs)")
    for model, series in trend.items():
        claps = [s["clap_score"] for s in series if s["clap_score"] is not None]
        if claps:
            console.info(
                f"  {model:<40} n={len(series)} clap {min(claps):.3f}..{max(claps):.3f} "
                f"(mean {sum(claps) / len(claps):.3f})"
            )
    return report


# --------------------------------------------------------------------------- #
# #35 — experiment matrix
# --------------------------------------------------------------------------- #
def experiment_matrix(cfg: Config) -> Dict[str, object]:
    """Flatten MLflow runs into a rows x metrics matrix."""
    from .experiments import search_runs

    df = search_runs(cfg)
    cols = ["run_id", "name", "task", "verdict", "model", "device", "seed",
            "target_bpm", "detected_bpm", "deviation", "clap_score",
            "duration_s", "n_tracks", "bpm_mean"]
    if df.empty:
        console.warn("No MLflow runs to build a matrix from.")
        return {"runs": 0, "columns": cols, "matrix": []}

    rows: List[dict] = []
    for _, r in df.iterrows():
        row = {}
        for c in cols:
            v = r.get(c)
            if hasattr(v, "item"):
                v = v.item()
            row[c] = v
        rows.append(row)

    report = {
        "runs": len(rows),
        "columns": cols,
        "matrix": rows,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    out = cfg.project_root / "metadata" / "experiment_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    console.ok(f"Experiment matrix -> metadata/experiment_matrix.json ({len(rows)} runs)")
    return report


# --------------------------------------------------------------------------- #
# #40 — model card
# --------------------------------------------------------------------------- #
def model_card(cfg: Config, checkpoint: str = "") -> Optional[Path]:
    """Render a markdown model card from eval aggregates + MLflow metrics."""
    from .report import load_results

    root = cfg.project_root
    rows = load_results(root)
    if not rows:
        console.error("No eval results — run `musictrain eval` first.")
        return None

    by_ckpt = {}
    for r in rows:
        by_ckpt.setdefault(r.get("checkpoint") or "(unknown)", []).append(r)
    if checkpoint and checkpoint not in by_ckpt:
        console.error(f"No eval rows for checkpoint '{checkpoint}'.")
        return None
    if not checkpoint:
        # pick the checkpoint with the most rows as the headline model
        checkpoint = max(by_ckpt, key=lambda k: len(by_ckpt[k]))
    crows = by_ckpt[checkpoint]

    claps = [r["clap_score"] for r in crows if r.get("clap_score") is not None]
    devs = [abs(r["deviation"]) for r in crows if r.get("deviation") is not None]
    ok = sum(1 for r in crows if r.get("status") == "ok")

    from collections import Counter

    sections = Counter(r.get("section") or "?" for r in crows)

    lines = [
        f"# Model card — `{checkpoint}`",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Eval rows: {len(crows)}",
        f"- OK-rate: {ok / len(crows):.0%}" if crows else "- OK-rate: —",
        "",
        "## Adherence",
        "",
        f"- Mean CLAP: {round(sum(claps) / len(claps), 4) if claps else '—'}",
        f"- Mean |deviation|: {round(sum(devs) / len(devs), 4) if devs else '—'}",
        "",
        "## Section coverage",
        "",
    ]
    lines += [f"- {s}: {n}" for s, n in sections.most_common()]
    lines += [
        "",
        "## Eval prompt sample",
        "",
    ]
    for r in crows[:3]:
        lines.append(f"- `{r.get('prompt', '')[:120]}`")
    lines.append("")

    safe = checkpoint.replace("/", "_").replace("\\", "_")
    out = root / "metadata" / f"model_card_{safe}.md"
    out.write_text("\n".join(lines))
    console.ok(f"Model card -> {out.relative_to(root)}")
    return out
