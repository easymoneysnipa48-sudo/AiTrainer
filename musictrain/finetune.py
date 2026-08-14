"""LoRA fine-tuning of the MusicGen decoder (advanced #12).

A minimal, honest training loop: encode each segment's audio to EnCodec
tokens, embed the description with the text encoder, and run next-token
cross-entropy on the decoder LM head with LoRA adapters applied via `peft`.

Requires: ``pip install peft`` (lazy-imported) and at least one (audio,
description) pair from data/segments + metadata/labels.csv (falling back to
data/clean + manifest.jsonl). The default is intentionally tiny (a few steps,
one codebook) — it is a starting point, not a production trainer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import console
from .config import Config


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


def train(
    cfg: Config,
    steps: int = 5,
    lr: float = 1e-4,
    batch_size: int = 1,
    limit: int = 0,
    out_dir: Optional[Path] = None,
    r: int = 8,
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
    console.step(f"Training data: {len(pairs)} pair(s) — first: {pairs[0][0].name}")

    if steps <= 0:
        console.ok(f"Dry run: data prepared, LoRA would train on {len(pairs)} pair(s)")
        return {"dry_run": True, "n_pairs": len(pairs), "steps": 0}

    try:
        import peft  # noqa: F401
    except Exception:  # noqa: BLE001
        console.error("LoRA fine-tuning needs `pip install peft`.")
        return {}

    from transformers import AutoModelForTextToWaveform, AutoProcessor

    from .inference import load_model, resolve_device

    device = resolve_device(cfg.inference.device)
    processor, model, device = load_model(cfg.inference)
    model.train()

    # LoRA on the decoder's attention projections
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=r, lora_alpha=2 * r, target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model.decoder = get_peft_model(model.decoder, lora_cfg)
    model.decoder.print_trainable_parameters()

    optim = torch.optim.AdamW(
        (p for p in model.decoder.parameters() if p.requires_grad), lr=lr
    )
    losses: List[float] = []
    sr = getattr(model.config.audio_encoder, "sampling_rate", 32000)

    for step in range(1, steps + 1):
        total = 0.0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            input_values = []
            text_inputs = None
            for path, desc in batch:
                import soundfile as sf

                audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
                audio = audio.mean(axis=1)
                import numpy as np

                input_values.append(torch.from_numpy(audio)[None, None].to(device))
                ti = processor(text=desc, return_tensors="pt", padding=True)
                ti = {k: v.to(device) for k, v in ti.items()}
                if text_inputs is None:
                    text_inputs = ti
            iv = torch.cat(input_values, dim=0)

            with torch.no_grad():
                enc = model.encodec(iv)  # EnCodec tokens
                codes = enc.audio_codes[:, 0, :].to(device)  # [B, T] single codebook
                hidden = model.text_encoder(**text_inputs).last_hidden_state
                hidden = model.encoder_proj(hidden) if hasattr(model, "encoder_proj") else hidden

            optim.zero_grad()
            out = model.decoder(
                input_ids=codes[:, :-1], encoder_hidden_states=hidden, labels=codes[:, 1:]
            )
            loss = out.loss
            loss.backward()
            optim.step()
            total += float(loss.detach().cpu())
        losses.append(round(total / max(len(pairs) // batch_size, 1), 4))
        console.step(f"step {step}/{steps} loss={losses[-1]}")

    out_dir = Path(out_dir) if out_dir else cfg.project_root / "adapters"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.decoder.save_pretrained(out_dir)
    console.ok(f"LoRA adapters -> {out_dir.relative_to(cfg.project_root)}")

    return {
        "out_dir": str(out_dir), "steps": steps, "n_pairs": len(pairs),
        "losses": losses, "final_loss": losses[-1] if losses else None, "r": r,
    }
