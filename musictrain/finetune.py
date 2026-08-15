"""LoRA fine-tuning of the MusicGen decoder (advanced #12, #21-#30).

A minimal, honest training loop: encode each segment's audio to EnCodec
tokens, embed the description with the text encoder, and run next-token
cross-entropy on the decoder LM head with LoRA adapters applied via `peft`.

Training-quality knobs added in the advanced eval batch:

* **LoRA (#21)** — lightweight adapters via `peft` (the default path).
* **Gradient checkpointing (#22)** — trades compute for VRAM on the decoder.
* **LR warmup + cosine decay (#23)** — `--lr-mode cosine --warmup N`.
* **bf16 mixed precision (#24)** — `--bf16` autocast on CUDA.
* **Streaming loader (#26)** — batches are read from disk on demand.
* **Curriculum ordering (#27)** — easy-first ordering (duration/size proxy).
* **CFG sweep (#28)** — records candidate guidance scales to MLflow.
* **Weight EMA (#29)** — exponential moving average of LoRA weights.
* **Multi-GPU DDP (#30)** — wraps the decoder in DDP when >1 GPU.

Requires: ``pip install peft`` (lazy-imported) and at least one (audio,
description) pair from data/segments + metadata/labels.csv (falling back to
data/clean + manifest.jsonl). The default is intentionally tiny (a few steps,
one codebook) — it is a starting point, not a production trainer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from . import console
from .config import Config
from .logging import get_logger

log = get_logger("finetune")


def _pairs(root: Path, limit: int = 0) -> List[Tuple[Path, str]]:
    """(audio, description) pairs from segments + labels, or clean + manifest."""
    labels = root / "metadata" / "labels.csv"
    manifest = root / "metadata" / "manifest.jsonl"
    seg_dir = root / "data" / "segments"
    clean_dir = root / "data" / "clean"

    desc_by_key: Dict[str, str] = {}
    if manifest.exists():
        for ln in manifest.open():
            if not ln.strip():
                continue
            row = json.loads(ln)
            key = Path(row.get("path", "")).stem
            desc_by_key[key] = row.get("description") or row.get("prompt") or ""
    if labels.exists():
        import csv

        for row in csv.DictReader(labels.open(newline="")):
            desc_by_key.setdefault(row.get("source_id") or "", row.get("description") or "")

    candidates: List[Path] = []
    if seg_dir.exists():
        candidates = sorted(seg_dir.glob("*.wav"))
    if not candidates and clean_dir.exists():
        candidates = sorted(clean_dir.glob("*.wav"))

    pairs: List[Tuple[Path, str]] = []
    for p in candidates:
        desc = desc_by_key.get(p.stem) or desc_by_key.get(p.stem.split("_seg")[0]) or ""
        if desc.strip():
            pairs.append((p, desc))
        if limit and len(pairs) >= limit:
            break
    return pairs


# --------------------------------------------------------------------------- #
# #23 LR warmup + cosine decay
# --------------------------------------------------------------------------- #
def lr_schedule(step: int, lr: float, warmup_steps: int, total_steps: int,
                mode: str = "cosine") -> float:
    """Scheduled learning rate for step ``step`` (0-indexed).

    * ``constant`` — flat LR after warmup.
    * ``cosine`` — cosine decay from ``lr`` to ~0 after warmup.
    """
    if step < max(warmup_steps, 1):
        return lr * (step + 1) / max(warmup_steps, 1)
    if mode == "constant":
        return lr
    if total_steps <= warmup_steps:
        return lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + np.cos(np.pi * progress))


# --------------------------------------------------------------------------- #
# #27 curriculum ordering (easy -> hard)
# --------------------------------------------------------------------------- #
def sort_curriculum(pairs: List[Tuple[Path, str]],
                    difficulty: Optional[Dict[str, float]] = None,
                    size_proxy: bool = True) -> List[Tuple[Path, str]]:
    """Order pairs easy-first.

    Uses an explicit ``difficulty`` map (keyed by stem) when given; otherwise
    falls back to file size (smaller/shorter = easier) as a cheap proxy.
    """
    if difficulty:
        def _key(item: Tuple[Path, str]) -> float:
            return difficulty.get(item[0].stem, 1.0)
    elif size_proxy:
        def _key(item: Tuple[Path, str]) -> float:
            return float(item[0].stat().st_size)
    else:
        return pairs
    return sorted(pairs, key=_key)


# --------------------------------------------------------------------------- #
# #26 streaming batch iterator
# --------------------------------------------------------------------------- #
def iter_batches(pairs: List[Tuple[Path, str]], batch_size: int = 1) -> Iterator[List[Tuple[Path, str]]]:
    """Yield batches lazily; audio is read on demand inside the loop (#26)."""
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i + batch_size]


# --------------------------------------------------------------------------- #
# #28 CFG sweep candidates
# --------------------------------------------------------------------------- #
def guidance_candidates(base: float = 3.0, n: int = 5) -> List[float]:
    """Spread of classifier-free-guidance scales around ``base`` to sweep."""
    offsets = np.linspace(-1.5, 1.5, n)
    return [round(max(0.5, base + o), 2) for o in offsets]


# --------------------------------------------------------------------------- #
# #29 EMA update
# --------------------------------------------------------------------------- #
def ema_update(ema: List, params: List, decay: float) -> None:
    """Exponential moving average of parameters in place (torch tensors)."""
    import torch

    with torch.no_grad():
        for e, p in zip(ema, params):
            e.mul_(decay).add_(p.detach(), alpha=1.0 - decay)


# --------------------------------------------------------------------------- #
# #30 DDP wrapper
# --------------------------------------------------------------------------- #
def maybe_ddp(model, device: str):
    """Wrap the decoder in DistributedDataParallel when >1 CUDA GPU exists."""
    import torch

    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            import os

            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            try:
                dist.init_process_group(backend="nccl", init_method="env://")
            except Exception:  # noqa: BLE001 - single-node fallback
                return model
        from torch.nn.parallel import DistributedDataParallel as DDP

        return DDP(model, device_ids=[torch.cuda.current_device()])
    return model


def train(
    cfg: Config,
    steps: int = 5,
    lr: float = 1e-4,
    batch_size: int = 1,
    limit: int = 0,
    out_dir: Optional[Path] = None,
    r: int = 8,
    warmup_steps: int = 0,
    lr_mode: str = "cosine",
    gradient_checkpointing: bool = False,
    bf16: bool = False,
    stream: bool = False,
    curriculum: bool = False,
    ema: bool = False,
    ema_decay: float = 0.999,
    ddp: bool = False,
    cfg_base: float = 3.0,
    cfg_sweep: int = 0,
) -> dict:
    """LoRA-train the decoder on (audio, description) pairs, save adapters."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        console.error(f"torch unavailable: {exc}")
        return {}

    pairs = _pairs(cfg.project_root, limit=limit)
    if not pairs:
        console.error(
            "No (audio, description) pairs found — add data/segments/*.wav with "
            "metadata/labels.csv or manifest.jsonl descriptions."
        )
        return {}
    if curriculum:
        pairs = sort_curriculum(pairs)
        console.step(f"Curriculum order: {len(pairs)} pair(s), easiest first")
    console.step(f"Training data: {len(pairs)} pair(s) — first: {pairs[0][0].name}")

    if steps <= 0:
        console.ok(f"Dry run: data prepared, LoRA would train on {len(pairs)} pair(s)")
        return {"dry_run": True, "n_pairs": len(pairs), "steps": 0}

    try:
        import peft  # noqa: F401
    except Exception:  # noqa: BLE001
        console.error("LoRA fine-tuning needs `pip install peft`.")
        return {}

    from .inference import load_model, resolve_device

    device = resolve_device(cfg.inference.device)
    processor, model, device = load_model(cfg.inference)

    # #24 bf16 mixed precision (CUDA only; MPS/CPU stay fp32)
    use_bf16 = bf16 and device.startswith("cuda") and torch.cuda.is_bf16_supported()
    if bf16 and not use_bf16:
        console.warn("bf16 requested but unsupported on this device — using fp32.")

    # #22 gradient checkpointing
    if gradient_checkpointing and hasattr(model.decoder, "gradient_checkpointing_enable"):
        model.decoder.gradient_checkpointing_enable()
        console.info("Gradient checkpointing enabled (VRAM-for-compute trade).")

    model.train()

    # LoRA on the decoder's attention projections
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=r, lora_alpha=2 * r, target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model.decoder = get_peft_model(model.decoder, lora_cfg)

    # #30 multi-GPU
    if ddp:
        model.decoder = maybe_ddp(model.decoder, device)

    if use_bf16:
        model.decoder = model.decoder.to(dtype=torch.bfloat16)
    model.decoder.print_trainable_parameters()

    optim = torch.optim.AdamW(
        (p for p in model.decoder.parameters() if p.requires_grad), lr=lr
    )

    # #29 EMA buffer
    ema_params: List = []
    if ema:
        ema_params = [p.detach().clone().float() for p in model.decoder.parameters() if p.requires_grad]

    total_steps = steps * max(len(pairs) // batch_size, 1)
    losses: List[float] = []
    step_idx = 0

    for step in range(1, steps + 1):
        total = 0.0
        n_batches = 0
        batches = iter_batches(pairs, batch_size) if stream else \
            [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
        for batch in batches:
            lr_now = lr_schedule(step_idx, lr, warmup_steps, total_steps, lr_mode)
            for g in optim.param_groups:
                g["lr"] = lr_now
            step_idx += 1

            input_values = []
            text_inputs = None
            for path, desc in batch:
                import soundfile as sf

                audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
                audio = audio.mean(axis=1)
                input_values.append(torch.from_numpy(audio)[None, None].to(device))
                ti = processor(text=desc, return_tensors="pt", padding=True)
                ti = {k: v.to(device) for k, v in ti.items()}
                if text_inputs is None:
                    text_inputs = ti
            iv = torch.cat(input_values, dim=0)
            if use_bf16:
                iv = iv.to(dtype=torch.bfloat16)

            with torch.no_grad():
                enc = model.encodec(iv)  # EnCodec tokens
                codes = enc.audio_codes[:, 0, :].to(device)  # [B, T] single codebook
                hidden = model.text_encoder(**text_inputs).last_hidden_state
                hidden = model.encoder_proj(hidden) if hasattr(model, "encoder_proj") else hidden

            optim.zero_grad()
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model.decoder(
                        input_ids=codes[:, :-1], encoder_hidden_states=hidden, labels=codes[:, 1:]
                    )
                    loss = out.loss
            else:
                out = model.decoder(
                    input_ids=codes[:, :-1], encoder_hidden_states=hidden, labels=codes[:, 1:]
                )
                loss = out.loss
            loss.backward()
            optim.step()
            if ema:
                trainable = [p for p in model.decoder.parameters() if p.requires_grad]
                ema_update(ema_params, trainable, ema_decay)
            total += float(loss.detach().cpu())
            n_batches += 1
        losses.append(round(total / max(n_batches, 1), 4))
        console.step(f"step {step}/{steps} loss={losses[-1]} lr={lr_now:.2e}")

    # apply EMA weights before saving
    if ema:
        trainable = [p for p in model.decoder.parameters() if p.requires_grad]
        with torch.no_grad():
            for p, e in zip(trainable, ema_params):
                p.copy_(e)

    # #28 CFG sweep: log candidate guidance scales to MLflow for later sweep
    cfg_points: List[float] = []
    if cfg_sweep:
        cfg_points = guidance_candidates(cfg_base, cfg_sweep)
        try:
            from .experiments import log_metric

            for i, g in enumerate(cfg_points):
                log_metric(cfg, "cfg_sweep_point", g, step=i)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not log CFG sweep points to MLflow: %s", exc)

    out_dir = Path(out_dir) if out_dir else cfg.project_root / "adapters"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.decoder.save_pretrained(out_dir)
    console.ok(f"LoRA adapters -> {out_dir.relative_to(cfg.project_root)}")

    return {
        "out_dir": str(out_dir), "steps": steps, "n_pairs": len(pairs),
        "losses": losses, "final_loss": losses[-1] if losses else None, "r": r,
        "gradient_checkpointing": gradient_checkpointing, "bf16": use_bf16,
        "ema": ema, "ddp": ddp, "curriculum": curriculum,
        "cfg_sweep": cfg_points, "lr_schedule": {"mode": lr_mode, "warmup_steps": warmup_steps},
    }
