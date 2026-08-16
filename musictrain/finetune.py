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


def _encode_batch(model, processor, batch: List[Tuple[Path, str]], device: str,
                  use_bf16: bool):
    """Encode one batch to EnCodec codes + text-conditioning hidden states."""
    import soundfile as sf
    import torch

    input_values = []
    text_inputs = None
    for path, desc in batch:
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
        enc = model.encodec(iv)
        codes = enc.audio_codes[:, 0, :].to(device)  # [B, T] single codebook
        hidden = model.text_encoder(**text_inputs).last_hidden_state
        hidden = model.encoder_proj(hidden) if hasattr(model, "encoder_proj") else hidden
    return codes, hidden


def _decoder_loss(model, codes, hidden, use_bf16: bool):
    """Next-token cross-entropy on the decoder, with optional bf16 autocast."""
    import torch

    if use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.decoder(
                input_ids=codes[:, :-1], encoder_hidden_states=hidden, labels=codes[:, 1:]
            )
            return out.loss
    out = model.decoder(
        input_ids=codes[:, :-1], encoder_hidden_states=hidden, labels=codes[:, 1:]
    )
    return out.loss


def _set_lr(optim, value: float) -> None:
    for g in optim.param_groups:
        g["lr"] = value


def _save_checkpoint(out_dir: Path, optim, opt_step: int, step: int,
                     losses: List[float], val_losses: List[float],
                     ema: bool, ema_params: List, meta: dict) -> None:
    """Persist optimizer + LR-scheduler counters so training can resume (#3)."""
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": optim.state_dict(),
            "opt_step": opt_step,
            "step": step,
            "losses": losses,
            "val_losses": val_losses,
            "ema": ema_params if ema else None,
        },
        out_dir / "trainer_state.pt",
    )
    (out_dir / "trainer_state.json").write_text(json.dumps(meta, indent=2))


def _load_checkpoint_state(resume_dir: Path, optim, ema_params: List, ema: bool) -> dict:
    """Restore optimizer/scheduler state + counters from a prior run (#3)."""
    import torch

    resume_dir = Path(resume_dir)
    pt = resume_dir / "trainer_state.pt"
    if not pt.exists():
        return {"opt_step": 0, "step": 0, "losses": [], "val_losses": []}
    try:
        state = torch.load(pt, map_location="cpu")
        optim.load_state_dict(state.get("optimizer", {}))
    except Exception as exc:  # noqa: BLE001 - best-effort resume
        log.warning("optimizer state restore failed: %s", exc)
    if ema and ema_params and state.get("ema"):
        with torch.no_grad():
            for e, s in zip(ema_params, state["ema"]):
                e.copy_(s)
    return {
        "opt_step": state.get("opt_step", 0),
        "step": state.get("step", 0),
        "losses": list(state.get("losses", [])),
        "val_losses": list(state.get("val_losses", [])),
    }


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
    # gap #3/#4/#5 — resume, gradient accumulation, validation, full fine-tune
    accum: int = 1,
    val_split: float = 0.0,
    full: bool = False,
    resume: str = "",
    check_leakage: bool = True,
) -> dict:
    """Train the decoder — LoRA by default, or full fine-tune — on
    (audio, description) pairs.

    * ``steps`` is the number of full passes over the data (epochs).
    * ``accum`` > 1 accumulates gradients over micro-batches (#4).
    * ``val_split`` > 0 holds out a fraction for a per-epoch validation loss (#4).
    * ``full`` trains every decoder weight instead of LoRA adapters (#5).
    * ``resume`` restores weights + optimizer/scheduler state from a prior run (#3).
    """
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
        console.ok(
            f"Dry run: data prepared, {'full' if full else 'LoRA'} would train "
            f"on {len(pairs)} pair(s)"
        )
        return {"dry_run": True, "n_pairs": len(pairs), "steps": 0, "full": full}

    # gap #8 — content-hash leakage guard before training
    if check_leakage:
        from .split import check_split_leakage

        leak = check_split_leakage(cfg.project_root)
        if not leak["clean"]:
            console.warn(
                f"Split leakage: {leak['n_overlaps']} content hash(es) appear in "
                f"more than one split — {leak['overlapping_files'][:5]}"
            )
        else:
            console.ok(f"Leakage guard: no overlapping content across splits "
                       f"{leak['checked']}")

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        console.error(f"torch unavailable: {exc}")
        return {}

    if not full:
        try:
            import peft  # noqa: F401
        except Exception:  # noqa: BLE001
            console.error("LoRA fine-tuning needs `pip install peft` (or pass --full).")
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

    resume_dir = Path(resume) if resume else None

    # -- mode: full fine-tune (#5) vs LoRA ---------------------------------
    if full:
        for p in model.decoder.parameters():
            p.requires_grad = True
        if resume_dir and (resume_dir / "decoder_state.pt").exists():
            try:
                model.decoder.load_state_dict(
                    torch.load(resume_dir / "decoder_state.pt", map_location=device)
                )
                console.ok(f"Resumed full decoder weights from {resume_dir}")
            except Exception as exc:  # noqa: BLE001
                console.warn(f"decoder state restore failed ({exc}) — starting fresh")
    else:
        from peft import LoraConfig, PeftModel, get_peft_model

        lora_cfg = LoraConfig(
            r=r, lora_alpha=2 * r, target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        if resume_dir and (resume_dir / "adapter_config.json").exists():
            model.decoder = PeftModel.from_pretrained(model.decoder, resume_dir)
            console.ok(f"Resumed LoRA adapter from {resume_dir}")
        else:
            model.decoder = get_peft_model(model.decoder, lora_cfg)

    # #30 multi-GPU
    if ddp:
        model.decoder = maybe_ddp(model.decoder, device)

    if use_bf16:
        model.decoder = model.decoder.to(dtype=torch.bfloat16)
    try:
        model.decoder.print_trainable_parameters()
    except Exception:  # noqa: BLE001 - full mode has no PEFT summary
        pass

    optim = torch.optim.AdamW(
        (p for p in model.decoder.parameters() if p.requires_grad), lr=lr
    )

    # #29 EMA buffer
    ema_params: List = []
    if ema:
        ema_params = [p.detach().clone().float() for p in model.decoder.parameters() if p.requires_grad]

    # -- train/val split (#4 validation) ----------------------------------
    train_pairs = pairs
    val_pairs: List[Tuple[Path, str]] = []
    if 0.0 < val_split < 1.0 and len(pairs) > 1:
        n_val = max(1, int(len(pairs) * val_split))
        val_pairs = pairs[-n_val:]
        train_pairs = pairs[:-n_val]
        console.info(f"Validation hold-out: {len(val_pairs)} pair(s)")

    # -- resume optimizer/counters (#3) -----------------------------------
    opt_step = 0
    start_step = 0
    losses: List[float] = []
    val_losses: List[float] = []
    if resume_dir:
        st = _load_checkpoint_state(resume_dir, optim, ema_params, ema)
        opt_step = st["opt_step"]
        start_step = st["step"]
        losses = st["losses"]
        val_losses = st["val_losses"]
        if start_step:
            console.ok(f"Resumed at step {start_step} (opt_step {opt_step})")

    n_train_batches = max(len(train_pairs) // batch_size, 1)
    total_opt_steps = steps * max((n_train_batches + accum - 1) // accum, 1)

    for step in range(start_step + 1, steps + 1):
        total = 0.0
        n_micro = 0
        batches = iter_batches(train_pairs, batch_size) if stream else \
            [train_pairs[i:i + batch_size] for i in range(0, len(train_pairs), batch_size)]
        optim.zero_grad()
        for batch in batches:
            codes, hidden = _encode_batch(model, processor, batch, device, use_bf16)
            loss = _decoder_loss(model, codes, hidden, use_bf16)
            (loss / accum).backward()
            n_micro += 1
            total += float(loss.detach().cpu())
            if n_micro % accum == 0:
                _set_lr(optim, lr_schedule(opt_step, lr, warmup_steps, total_opt_steps, lr_mode))
                optim.step()
                optim.zero_grad()
                opt_step += 1
                if ema:
                    trainable = [p for p in model.decoder.parameters() if p.requires_grad]
                    ema_update(ema_params, trainable, ema_decay)
        # flush a trailing partial accumulation group
        if n_micro % accum != 0:
            _set_lr(optim, lr_schedule(opt_step, lr, warmup_steps, total_opt_steps, lr_mode))
            optim.step()
            optim.zero_grad()
            opt_step += 1
            if ema:
                trainable = [p for p in model.decoder.parameters() if p.requires_grad]
                ema_update(ema_params, trainable, ema_decay)
        train_loss = round(total / max(n_micro, 1), 4)
        losses.append(train_loss)
        console.step(f"step {step}/{steps} loss={train_loss} lr={optim.param_groups[0]['lr']:.2e}")

        # -- per-epoch validation (#4) ------------------------------------
        if val_pairs:
            model.eval()
            vtotal = 0.0
            vn = 0
            with torch.no_grad():
                for vb in [val_pairs[i:i + batch_size] for i in range(0, len(val_pairs), batch_size)]:
                    codes, hidden = _encode_batch(model, processor, vb, device, use_bf16)
                    vloss = _decoder_loss(model, codes, hidden, use_bf16)
                    vtotal += float(vloss.detach().cpu())
                    vn += 1
            val_losses.append(round(vtotal / max(vn, 1), 4))
            console.info(f"  val loss={val_losses[-1]}")
            model.train()

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
    if full:
        torch.save(model.decoder.state_dict(), out_dir / "decoder_state.pt")
        console.ok(f"Full decoder weights -> {out_dir.relative_to(cfg.project_root)}")
    else:
        model.decoder.save_pretrained(out_dir)
        console.ok(f"LoRA adapters -> {out_dir.relative_to(cfg.project_root)}")

    _save_checkpoint(
        out_dir, optim, opt_step, steps, losses, val_losses, ema, ema_params,
        {
            "mode": "full" if full else "lora",
            "steps": steps,
            "lr": lr,
            "r": r,
            "batch_size": batch_size,
            "accum": accum,
            "val_split": val_split,
            "n_pairs": len(pairs),
            "lr_mode": lr_mode,
            "warmup_steps": warmup_steps,
        },
    )

    return {
        "out_dir": str(out_dir), "steps": steps, "n_pairs": len(pairs),
        "losses": losses, "val_losses": val_losses,
        "final_loss": losses[-1] if losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "r": r, "full": full, "accum": accum,
        "gradient_checkpointing": gradient_checkpointing, "bf16": use_bf16,
        "ema": ema, "ddp": ddp, "curriculum": curriculum,
        "cfg_sweep": cfg_points, "lr_schedule": {"mode": lr_mode, "warmup_steps": warmup_steps},
    }
