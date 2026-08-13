"""CLAP audio-text similarity scoring (prompt adherence).

Encodes the generated audio and its conditioning prompt with LAION's CLAP
(laion/clap-htsat-unfused) and reports their cosine similarity as an automated
prompt-adherence score. The model is cached in-process so batch eval only loads
it once.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import console
from .config import Config

_cache: dict = {}


def resolve_device(preferred: str) -> str:
    import torch

    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def load_clap(model_name: str, device: str) -> Tuple:
    device = resolve_device(device)
    key = (model_name, device)
    if key in _cache:
        return _cache[key]

    from transformers import AutoFeatureExtractor, AutoTokenizer, ClapModel

    console.info(f"Loading CLAP {model_name} on {device}…")
    model = ClapModel.from_pretrained(model_name).to(device)
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    _cache[key] = (fe, tok, model, device)
    return _cache[key]


def _pooler(result):
    return result[0].pooler_output if isinstance(result, tuple) else result.pooler_output


def score(cfg: Config, audio_path: Path, text: str) -> Optional[float]:
    """Return the CLAP cosine similarity between `audio_path` and `text`."""
    if not cfg.clap.enabled:
        return None
    import librosa
    import soundfile as sf
    import torch

    fe, tok, model, device = load_clap(cfg.clap.model_name, cfg.clap.device)

    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        sr = 48000

    audio_inputs = fe(raw_speech=[audio], sampling_rate=sr, return_tensors="pt")
    text_inputs = tok([text], padding=True, return_tensors="pt")
    audio_inputs = {k: v.to(device) for k, v in audio_inputs.items() if hasattr(v, "to")}
    text_inputs = {k: v.to(device) for k, v in text_inputs.items() if hasattr(v, "to")}

    with torch.inference_mode():
        audio_emb = _pooler(
            model.get_audio_features(
                input_features=audio_inputs["input_features"],
                is_longer=audio_inputs.get("is_longer"),
            )
        )
        text_emb = _pooler(
            model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
        )

    sim = float((audio_emb @ text_emb.T)[0, 0].item())
    return round(sim, 4)
