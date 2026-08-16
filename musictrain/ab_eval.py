"""Base-vs-fine-tuned A/B eval (gap #2 in the feature list).

Runs the fixed evaluation prompt set twice — once on the stock base model and
once with a LoRA adapter loaded — then diffs the two result sets with a paired
significance test (CLAP + |BPM deviation|). This closes the "train but can't
measure the fine-tune" loop: it answers *did the adapter actually help?* with a
p-value rather than a hand-wavy number.

Outputs (all under metadata/):
* ``ab_eval.json``         — verdict, per-metric deltas, p-values, win rates
* ``ab_base.jsonl``        — the base-model rows
* ``ab_adapter.jsonl``     — the adapter rows
* ``significance.json``    — the raw paired test (via significance.compare)
"""
from __future__ import annotations

import json
from typing import List, Optional

from . import console
from .config import Config


def _win_rate(base: List[dict], adapt: List[dict], key: str,
              higher_is_better: bool) -> dict:
    from .significance import _pairs  # reuse the stable pairing logic

    a, b = _pairs(base, adapt)
    wins = ties = 0
    n = 0
    for ra, rb in zip(a, b):
        va, vb = ra.get(key), rb.get(key)
        if va is None or vb is None:
            continue
        n += 1
        d = vb - va
        if not higher_is_better:
            d = -d
        if d > 1e-9:
            wins += 1
        elif abs(d) <= 1e-9:
            ties += 1
    if not n:
        return {"n": 0, "wins": 0, "ties": 0, "losses": 0, "win_rate": None}
    losses = n - wins - ties
    return {
        "n": n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": round(wins / n, 4),
    }


def run_ab_eval(
    cfg: Config,
    adapter: str,
    limit: int = 0,
    seeds: int = 1,
    section: Optional[str] = None,
    progress=None,
    cancel=None,
) -> dict:
    from .evalset import run_eval
    from .significance import compare

    adapter = str(adapter)
    meta = cfg.project_root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    base_file = meta / "ab_base.jsonl"
    adapter_file = meta / "ab_adapter.jsonl"

    # 1) base model (adapter forced off)
    console.step(f"A/B eval: base model ({cfg.inference.model_name})")
    saved_adapter = cfg.inference.adapter
    cfg.inference.adapter = ""
    base_rows = run_eval(
        cfg,
        limit=limit,
        section=section,
        seeds=seeds,
        results_file=base_file,
        progress=progress,
        cancel=cancel,
    )

    # 2) fine-tuned adapter
    console.step(f"A/B eval: adapter ({adapter})")
    cfg.inference.adapter = adapter
    adapter_rows = run_eval(
        cfg,
        limit=limit,
        section=section,
        seeds=seeds,
        results_file=adapter_file,
        progress=progress,
        cancel=cancel,
    )
    cfg.inference.adapter = saved_adapter

    if not base_rows or not adapter_rows:
        console.error("A/B eval produced no results on one side.")
        return {}

    # 3) paired significance + win rates
    diff = compare(cfg, base_rows, adapter_rows, label_a="base", label_b=adapter)
    win_clap = _win_rate(base_rows, adapter_rows, "clap_score", higher_is_better=True)
    win_dev = _win_rate(base_rows, adapter_rows, "deviation", higher_is_better=False)

    ok_base = sum(1 for r in base_rows if r.get("status") == "ok")
    ok_adapter = sum(1 for r in adapter_rows if r.get("status") == "ok")

    out = {
        "base": cfg.inference.model_name,
        "adapter": adapter,
        "n_prompts": len(adapter_rows),
        "seeds": seeds,
        "ok_rate_base": round(ok_base / max(len(base_rows), 1), 4),
        "ok_rate_adapter": round(ok_adapter / max(len(adapter_rows), 1), 4),
        "metrics": diff.get("metrics", {}),
        "win_rates": {"clap_score": win_clap, "abs_deviation": win_dev},
        "summary": diff.get("summary", ""),
    }
    path = meta / "ab_eval.json"
    path.write_text(json.dumps(out, indent=2))
    console.ok(f"A/B eval -> {path.relative_to(cfg.project_root)}")
    return out
