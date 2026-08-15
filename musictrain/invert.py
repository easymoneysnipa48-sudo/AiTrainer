"""Audio-to-prompt inversion (Advanced #22).

Given an audio clip, produce a text prompt that describes it, so you can
reverse-engineer prompts from reference tracks or verify prompt adherence.

Two signals are combined:

1. **Feature template** — from ``audio/analysis`` output (key, BPM, energy,
   vocal/instrumental, onset density) we synthesize a descriptive sentence.
2. **CLAP retrieval** — embed the audio and return the closest prompts from a
   prompt library (the eval prompt set, if present) so you get an exact,
   reusable prompt string.

Writes ``metadata/inverted_prompts.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import console
from .config import Config

_SECTION_WORDS = ["intro", "verse", "pre-chorus", "chorus", "bridge", "outro", "drop"]


def _analyze_features(cfg: Config, path: Path) -> dict:
    """Best-effort features using analysis if present, else quick librosa pass."""
    from .audio.analysis import analyze_file

    try:
        return analyze_file(cfg, path, path.parents[2] if len(path.parts) >= 3 else path.parent)
    except Exception as exc:  # noqa: BLE001
        console.warn(f"Analysis failed for {path.name}: {exc}")
        return {}


def template_prompt(feats: dict) -> str:
    """Synthesize a descriptive prompt from analysis features."""
    parts: List[str] = []

    bg = feats.get("beat_grid") or {}
    bpm = bg.get("tempo")
    if bpm:
        if bpm >= 150:
            parts.append("fast, high-energy")
        elif bpm >= 120:
            parts.append("uptempo")
        elif bpm >= 90:
            parts.append("mid-tempo")
        else:
            parts.append("slow")

    key = (feats.get("key") or {}).get("key")
    if key:
        parts.append(f"in {key}")

    sw = (feats.get("swing") or {}).get("feel")
    if sw in ("swung", "moderate"):
        parts.append("with a swung, groovy feel")

    vocal = (feats.get("vocal") or {}).get("verdict")
    if vocal == "vocal":
        parts.append("with vocals")
    elif vocal == "instrumental":
        parts.append("instrumental")

    ons = (feats.get("onsets") or {}).get("onset_density")
    if ons is not None:
        parts.append("dense percussion" if ons > 6.0 else "sparse percussion")

    prompt = ", ".join(parts) if parts else "a clean, balanced music track"
    return prompt


def _load_prompt_library(root: Path) -> List[str]:
    """Collect candidate prompts from the eval set if it exists."""
    from .evalset import load

    try:
        prompts = load(root)
    except Exception:  # noqa: BLE001
        return []
    return [p["prompt"] for p in prompts if p.get("prompt")]


def _clap_retrieve(cfg: Config, path: Path, library: List[str], top_k: int) -> List[dict]:
    if not library or not cfg.clap.enabled:
        return []
    from .similarity import load_clap, resolve_device
    from .embeddings import embed_audio

    device = resolve_device(cfg.clap.device)
    _fe, tok, model, device = load_clap(cfg.clap.model_name, device)

    aemb = embed_audio(cfg, path)
    aemb = aemb / (np.linalg.norm(aemb) + 1e-12)

    import torch

    scored: List[tuple] = []
    with torch.inference_mode():
        for i in range(0, len(library), 32):
            batch = library[i : i + 32]
            inputs = tok(batch, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
            tembs = model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            if isinstance(tembs, tuple):
                tembs = tembs[0].pooler_output
            else:
                tembs = tembs.pooler_output
            for j, t in enumerate(tembs):
                t = t.detach().cpu().numpy()
                t = t / (np.linalg.norm(t) + 1e-12)
                scored.append((float(aemb @ t), batch[j]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"prompt": prompt, "similarity": round(sim, 4)} for sim, prompt in scored[:top_k]
    ]


def invert(
    root: Path,
    cfg: Config,
    path: Path,
    top_k: int = 5,
    include_template: bool = True,
) -> dict:
    console.step(f"Inverting {path.name} -> prompt")

    feats = _analyze_features(cfg, path)
    template = template_prompt(feats) if include_template else None

    library = _load_prompt_library(root)
    retrieved = _clap_retrieve(cfg, path, library, top_k)

    report: dict = {
        "audio": str(path),
        "template_prompt": template,
        "retrieved": retrieved,
        "features_summary": {
            k: feats.get(k)
            for k in ("beat_grid", "key", "swing", "onsets", "vocal")
        },
        "at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }

    out = root / "metadata" / "inverted_prompts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    console.ok("Wrote -> metadata/inverted_prompts.json")
    if template:
        console.info(f"template: {template}")
    for r in retrieved:
        console.info(f"retrieved ({r['similarity']:.3f}): {r['prompt']}")
    return report
