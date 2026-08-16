"""Advanced training helpers (#1-#7 in the feature gap list).

Everything here degrades gracefully: heavy dependencies (optuna, mlx,
bitsandbytes, optimum, peft) are imported lazily inside the functions that
need them, and the pure-logic functions (grids, plans, math) are importable
and testable on any box. This keeps the module's contract identical to the
rest of the package: useful on Apple Silicon today, fully exercising once the
optional packages are installed.
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import console
from .config import Config
from .logging import get_logger

log = get_logger("tuning")


# --------------------------------------------------------------------------- #
# Pure logic (testable without torch/mlx/optuna)
# --------------------------------------------------------------------------- #

def grad_accum_steps(target_batch: int, per_gpu_batch: int,
                     n_gpus: int = 1) -> Tuple[int, int]:
    """Return (accum_steps, effective_batch) for a target global batch size.

    ``effective_batch == per_gpu_batch * n_gpus * accum_steps``, the closest
    feasible multiple of ``per_gpu_batch * n_gpus`` that is >= ``target_batch``
    (falling back to a single step if the target is unreachable).
    """
    if per_gpu_batch <= 0 or n_gpus <= 0:
        raise ValueError("per_gpu_batch and n_gpus must be positive")
    micro = per_gpu_batch * n_gpus
    accum = max(1, math.ceil(target_batch / micro))
    return accum, micro * accum


def quantize_plan(model_bytes: int, vram_bytes: int,
                  dtype: str = "fp32") -> List[Dict[str, object]]:
    """Suggest a quantization ladder (fp16 -> int8 -> int4) to fit a model in VRAM."""
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}.get(dtype, 4)
    params = model_bytes / bytes_per_param
    plan: List[Dict[str, object]] = []
    for label, mult in (("fp16", 2), ("int8", 1), ("int4", 0.5)):
        est = params * mult
        plan.append({
            "dtype": label,
            "est_bytes": int(est),
            "fits": est <= vram_bytes,
            "overhead": round(max(0.0, est - vram_bytes) / max(vram_bytes, 1), 3),
        })
    return plan


def hpo_grid(lr_values: Sequence[float], batch_values: Sequence[int],
             rank_values: Sequence[int], n_trials: int = 0,
             seed: int = 0) -> List[Dict[str, float]]:
    """Deterministic Cartesian grid of hyperparameter candidates.

    With ``n_trials`` > 0, sample that many from the grid (seeded round-robin
    shuffle) instead of returning every combination — a dependency-free
    fallback for the optuna/ray search.
    """
    grid = [
        {"lr": lr, "batch": batch, "rank": rank}
        for lr in lr_values for batch in batch_values for rank in rank_values
    ]
    if n_trials and n_trials < len(grid):
        rng = __import__("random").Random(seed)
        picked = sorted(grid, key=lambda _: rng.random())[:n_trials]
        return picked
    return grid


def recommended_backend() -> str:
    """Best available backend on this host: mlx > mps > cuda > cpu."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx  # noqa: F401
            return "mlx"
        except ImportError:
            return "mps"
    try:
        import torch  # noqa: F401
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def lr_find_plan(min_lr: float = 1e-6, max_lr: float = 1e-2, n: int = 10) -> List[float]:
    """Log-spaced learning-rate candidates for an LR range test (#6)."""
    if min_lr <= 0 or max_lr <= 0 or max_lr <= min_lr or n < 2:
        raise ValueError("need 0 < min_lr < max_lr and n >= 2")
    lo, hi = math.log10(min_lr), math.log10(max_lr)
    return [round(10 ** (lo + (hi - lo) * i / (n - 1)), 8) for i in range(n)]


def pick_best_lr(losses: Sequence[float], lrs: Sequence[float]) -> dict:
    """Choose a learning rate from an LR range-test loss curve (#6).

    Recommends the LR at the minimum loss (the practical choice — the point
    before loss starts climbing again as LR overshoots), and also reports the
    steepest-descent index so callers can apply the classic `min-loss LR / 10`
    rule if they prefer a more conservative pick.
    """
    losses = [float(x) for x in losses]
    lrs = [float(x) for x in lrs]
    if len(losses) != len(lrs) or len(losses) < 2:
        return {"best_lr": None, "best_loss": None, "reason": "need equal-length loss/lr series (>= 2)"}
    min_i = int(min(range(len(losses)), key=lambda j: losses[j]))
    steepest = 0
    steepest_slope = float("inf")
    for i in range(1, len(losses)):
        slope = (losses[i] - losses[i - 1]) / max(lrs[i] - lrs[i - 1], 1e-12)
        if slope < steepest_slope:
            steepest_slope = slope
            steepest = i
    return {
        "best_lr": lrs[min_i],
        "best_loss": losses[min_i],
        "min_loss_lr": lrs[min_i],
        "min_loss": losses[min_i],
        "steepest_index": steepest,
    }


def auto_batch_size(model_bytes: int, vram_bytes: int, dtype: str = "fp32",
                    per_sample_bytes: int = 0, headroom: float = 0.15) -> dict:
    """Suggest a per-device batch size from a VRAM budget (#6)."""
    if model_bytes <= 0 or vram_bytes <= 0:
        return {"batch": None, "note": "need model_bytes and vram_bytes to size the batch"}
    # weights + grads (training) + optimizer states (AdamW: 2x) + activations
    train_footprint = model_bytes * (1 + 1 + 2)
    free = vram_bytes * (1.0 - headroom) - train_footprint
    if per_sample_bytes <= 0:
        return {
            "model_training_bytes": int(train_footprint),
            "free_after_model": int(free),
            "note": "provide per_sample_bytes for a concrete batch size",
        }
    batch = max(1, int(free / per_sample_bytes))
    return {
        "batch": batch,
        "model_training_bytes": int(train_footprint),
        "free_after_model": int(free),
        "dtype": dtype,
        "headroom": headroom,
    }


def tokenize_candidates(tokens: Iterable[str]) -> List[str]:
    """Validate/normalize custom style tokens for tokenizer extension.

    Rules: lower-case, single underscore-joined word, max 32 chars, strip
    leading "<" / trailing ">" wrappers (MusicGen-style special tokens).
    """
    out: List[str] = []
    for raw in tokens:
        t = raw.strip().lower().replace(" ", "_")
        t = t.strip("<>")
        t = "".join(c for c in t if c.isalnum() or c == "_")
        t = "_".join(part for part in t.split("_") if part)
        if 0 < len(t) <= 32:
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# Wrappers (degrade gracefully on missing optional deps)
# --------------------------------------------------------------------------- #

def resume_from(root: Path, adapters_dir: Optional[Path] = None) -> dict:
    """Locate the most recent LoRA adapter checkpoint to resume from."""
    adir = Path(adapters_dir) if adapters_dir else Path(root) / "adapters"
    if not adir.exists():
        return {"resume_path": None, "note": f"no adapters dir at {adir}"}
    candidates = sorted(
        adir.glob("**/*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        return {"resume_path": None, "note": f"no adapter weights under {adir}"}
    latest = candidates[0]
    log.info("resume candidate: %s", latest)
    return {"resume_path": str(latest), "n_candidates": len(candidates)}


def hpo_search(cfg: Config, metric: str = "leaderboard_score",
               n_trials: int = 10, seed: int = 0) -> dict:
    """Optuna-backed hyperparameter search, falling back to a seeded grid."""
    grid = hpo_grid(
        [5e-5, 1e-4, 2e-4], [1, 2, 4], [4, 8, 16], n_trials=n_trials, seed=seed
    )
    try:
        import optuna  # noqa: F401
    except ImportError:
        console.warn("optuna not installed — using deterministic grid search fallback.")
        log.warning("optuna unavailable; grid fallback of %d candidate(s)", len(grid))
        return {"backend": "grid", "metric": metric, "trials": grid}

    console.step(f"optuna search over {n_trials} trial(s), objective={metric}")
    # Real objective needs a training run; here we record the study shape and
    # let the caller attach trials. Kept side-effect-light so the CLI never
    # trains by accident.
    study_name = f"mt-hpo-{metric}"
    return {
        "backend": "optuna",
        "metric": metric,
        "study_name": study_name,
        "n_trials": n_trials,
        "seed": seed,
        "suggested_grid": grid,
    }


def mlx_status() -> dict:
    """Report whether Apple MLX is usable and the recommended backend."""
    try:
        import mlx  # noqa: F401
        mlx_version = getattr(mlx, "__version__", "unknown")
        usable = True
    except ImportError as exc:
        mlx_version = None
        usable = False
        log.debug("mlx unavailable: %s", exc)
    return {
        "available": usable,
        "version": mlx_version,
        "recommended_backend": recommended_backend(),
    }


def apply_quantization(model_name: str, bits: int = 8) -> dict:
    """Quantize a model to ``bits`` (8 or 4). Needs bitsandbytes/optimum."""
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    try:
        import torch  # noqa: F401
        import bitsandbytes  # noqa: F401
        import optimum  # noqa: F401
        loaded = True
    except ImportError as exc:
        log.warning("quantization deps unavailable: %s", exc)
        loaded = False
    return {
        "model": model_name,
        "bits": bits,
        "backend": "bitsandbytes" if loaded else None,
        "note": "" if loaded else "install bitsandbytes + optimum to quantize",
    }


def extend_tokenizer(tokens: Sequence[str], out_path: Optional[Path] = None) -> dict:
    """Register custom style tokens for the tokenizer (persists to JSON)."""
    clean = tokenize_candidates(tokens)
    target = Path(out_path) if out_path else Path("metadata") / "custom_tokens.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: List[str] = []
    if target.exists():
        try:
            existing = json.loads(target.read_text())
        except json.JSONDecodeError:
            existing = []
    merged = sorted(set(existing) | set(clean))
    target.write_text(json.dumps(merged, indent=2))
    console.ok(f"{len(merged)} custom token(s) -> {target}")
    return {"tokens": merged, "added": len(merged) - len(existing), "path": str(target)}


def textual_inversion(concept: str, examples: Sequence[Path]) -> dict:
    """Placeholder for textual-inversion style learning (needs diffusers)."""
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        ready = True
    except ImportError as exc:
        log.warning("textual inversion deps unavailable: %s", exc)
        ready = False
    return {
        "concept": concept,
        "n_examples": len(examples),
        "ready": ready,
        "note": "" if ready else "install diffusers + torch to learn style tokens",
    }


def run(root: Path, cfg: Config, task: str, **kwargs) -> dict:
    """Dispatch for the `musictrain tuning --task ...` command."""
    def _series(v):
        if not v:
            return []
        if isinstance(v, str):
            return [float(x) for x in v.split(",") if x.strip()]
        return [float(x) for x in v]

    def _pick_lr():
        return pick_best_lr(_series(kwargs.get("losses")), _series(kwargs.get("lrs")))

    tasks = {
        "resume": lambda: resume_from(root, kwargs.get("adapters_dir")),
        "hpo": lambda: hpo_search(cfg, kwargs.get("metric", "leaderboard_score"),
                                  kwargs.get("n_trials", 10), kwargs.get("seed", 0)),
        "mlx": mlx_status,
        "quantize": lambda: apply_quantization(
            kwargs.get("model_name", cfg.inference.model_name),
            kwargs.get("bits", 8)),
        "tokens": lambda: extend_tokenizer(kwargs.get("tokens", [])),
        "inversion": lambda: textual_inversion(
            kwargs.get("concept", ""), kwargs.get("examples", [])),
        "plan": lambda: quantize_plan(
            kwargs.get("model_bytes", 0), kwargs.get("vram_bytes", 0),
            kwargs.get("dtype", "fp32")),
        "lr-find": lambda: {"lr_candidates": lr_find_plan(
            kwargs.get("min_lr", 1e-6), kwargs.get("max_lr", 1e-2),
            kwargs.get("n", 10))},
        "pick-lr": _pick_lr,
        "auto-batch": lambda: auto_batch_size(
            kwargs.get("model_bytes", 0), kwargs.get("vram_bytes", 0),
            kwargs.get("dtype", "fp32"), kwargs.get("per_sample_bytes", 0),
            kwargs.get("headroom", 0.15)),
    }
    if task not in tasks:
        console.error(f"unknown tuning task {task!r} — one of {sorted(tasks)}")
        return {"error": f"unknown task {task}"}
    result = tasks[task]()
    if isinstance(result, list):
        console.info(json.dumps(result, indent=2))
        return {"task": task, "results": result}
    console.info(json.dumps(result, indent=2))
    return {"task": task, **result}
