"""Cost tracking (Advanced #48).

Estimates the compute cost of generation/fine-tune runs from clip count x
model size, so you can compare efficiency across checkpoints and budget long
batches. Logs every estimate to ``metadata/cost_log.jsonl``.

Rough model sizes (params): musicgen-small ~300M, medium ~1.5B, large ~3.3B,
melody ~1.5B, plus LoRA adapters (rank*r^2 per layer).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict

from . import console
from .config import Config

_MODEL_PARAMS: Dict[str, float] = {
    "small": 300e6,
    "medium": 1.5e9,
    "large": 3.3e9,
    "melody": 1.5e9,
}

# rough FLOPs per token per param (2x for fwd+bwd), tokens ~ max_new_tokens
_FLOP_PER_PARAM_PER_TOKEN = 2.0
# Apple-Silicon-ish energy: ~1e-12 J/FLOP on MPS (very rough, order of magnitude)
_JOULES_PER_FLOP = 1e-12


def estimate(
    model_name: str,
    n_clips: int,
    tokens_per_clip: int = 256,
    n_epochs: int = 0,
    lora_rank: int = 0,
) -> Dict[str, float]:
    """Return cost estimate dict for a run."""
    base = 1.5e9
    for key, params in _MODEL_PARAMS.items():
        if key in model_name.lower():
            base = params
            break
    if lora_rank:
        # LoRA trains only the adapter params: ~ rank * hidden^2 * layers * 4
        base = float(lora_rank) * 1024.0 * 1024.0 * 24.0 * 4.0

    forward_ops = base * tokens_per_clip * n_clips
    epochs_mult = n_epochs if n_epochs > 0 else 1
    total_flops = forward_ops * _FLOP_PER_PARAM_PER_TOKEN * epochs_mult
    joules = total_flops * _JOULES_PER_FLOP
    return {
        "model_name": model_name,
        "n_clips": n_clips,
        "params": base,
        "tokens": tokens_per_clip * n_clips,
        "total_flops": total_flops,
        "estimated_joules": round(joules, 3),
        "estimated_kwh": round(joules / 3.6e6, 6),
    }


def log_cost(
    cfg: Config,
    task: str,
    model_name: str,
    n_clips: int,
    tokens_per_clip: int = 256,
    n_epochs: int = 0,
    lora_rank: int = 0,
) -> Dict[str, float]:
    """Estimate + append a cost record to metadata/cost_log.jsonl."""
    est = estimate(model_name, n_clips, tokens_per_clip, n_epochs, lora_rank)
    record = {
        "at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        **est,
    }
    path = cfg.project_root / "metadata" / "cost_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    console.info(
        f"cost: {task} {model_name} x {n_clips} clip(s) "
        f"~ {est['estimated_kwh']} kWh ({est['estimated_joules']:.0f} J)"
    )
    return est


def cost_summary(cfg: Config) -> Dict[str, object]:
    """Aggregate metadata/cost_log.jsonl into a summary."""
    path = cfg.project_root / "metadata" / "cost_log.jsonl"
    if not path.exists():
        return {"runs": 0, "total_kwh": 0.0, "by_task": {}}
    rows = []
    for ln in path.read_text().splitlines():
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                continue
    total_kwh = sum(r.get("estimated_kwh", 0.0) for r in rows)
    by_task: Dict[str, float] = {}
    for r in rows:
        by_task[r.get("task", "?")] = by_task.get(r.get("task", "?"), 0.0) + r.get("estimated_kwh", 0.0)
    return {"runs": len(rows), "total_kwh": round(total_kwh, 6), "by_task": {k: round(v, 6) for k, v in by_task.items()}}
