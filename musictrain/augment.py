"""Audio augmentation (Advanced #24).

Generates training variants from clean audio — pitch shift, time stretch,
background noise, EQ/spectral tilt, and gain variation — so the fine-tune set
generalizes beyond the exact source files.

Augmented clips are written to ``data/augmented/<track>__<op>.wav`` with a
manifest at ``metadata/augmented.json`` describing each transform.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from . import console
from .config import Config

OPS = ("pitch_up", "pitch_down", "stretch", "noise", "eq", "quiet")


def _load(path: Path):
    import soundfile as sf

    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def _save(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr)


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    import librosa

    return librosa.resample(audio, orig_sr=sr_in, target_sr=sr_out)


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    import librosa

    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    import librosa

    return librosa.effects.time_stretch(audio, rate=rate)


def add_noise(audio: np.ndarray, level: float = 0.004, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(audio.shape).astype(np.float32)
    return audio + level * noise * (np.abs(audio).max() + 1e-6)


def spectral_tilt(audio: np.ndarray, sr: int, tilt_db: float = 3.0) -> np.ndarray:
    """Apply a one-pole spectral tilt (simple EQ color)."""
    import scipy.signal

    alpha = 1.0 - np.abs(tilt_db) / 12.0
    if tilt_db >= 0:  # brighten: high-pass-ish one-pole
        b = [1.0, -alpha]
        a = [1.0]
    else:  # darken: low-pass one-pole
        b = [1.0 - alpha]
        a = [1.0, -alpha]
    return scipy.signal.lfilter(b, a, audio).astype(np.float32)


def _default_params(op: str) -> dict:
    if op == "pitch_up":
        return {"semitones": 2.0}
    if op == "pitch_down":
        return {"semitones": -2.0}
    if op == "stretch":
        return {"rate": 1.1}
    if op == "noise":
        return {"level": 0.004}
    if op == "eq":
        return {"tilt_db": 3.0}
    if op == "quiet":
        return {"gain": 0.7}
    return {}


def _rand_params(op: str, rng) -> dict:
    """Draw a random parameter set for an op (pitch ±1-3 st, stretch 0.9-1.1x, ...)."""
    if op == "pitch_up":
        return {"semitones": round(float(rng.uniform(1.0, 3.0)), 2)}
    if op == "pitch_down":
        return {"semitones": round(-float(rng.uniform(1.0, 3.0)), 2)}
    if op == "stretch":
        return {"rate": round(float(rng.uniform(0.9, 1.1)), 4)}
    if op == "noise":
        return {"level": round(float(rng.uniform(0.002, 0.008)), 5)}
    if op == "eq":
        return {"tilt_db": round(float(rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 6.0)), 2)}
    if op == "quiet":
        return {"gain": round(float(rng.uniform(0.6, 0.9)), 3)}
    return {}


def _apply_op(op: str, audio: np.ndarray, sr: int, p: dict, rng) -> np.ndarray:
    if op == "pitch_up":
        return pitch_shift(audio, sr, p["semitones"])
    if op == "pitch_down":
        return pitch_shift(audio, sr, p["semitones"])
    if op == "stretch":
        return time_stretch(audio, sr, p["rate"])
    if op == "noise":
        return add_noise(audio, p["level"], seed=int(rng.integers(0, 1 << 30)))
    if op == "eq":
        return spectral_tilt(audio, sr, p["tilt_db"])
    if op == "quiet":
        return audio * p["gain"]
    return audio


def augment(
    root: Path,
    cfg: Config,
    which: str = "segments",
    ops: List[str] | None = None,
    variants: int = 1,
    limit: int = 0,
    seed: int = 0,
) -> List[dict]:
    from .audio.inventory import AUDIO_GLOB

    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return []

    files: List[Path] = []
    for pattern in AUDIO_GLOB:
        files.extend(sorted(target.glob(pattern)))
    files = sorted(set(files))
    if not files:
        console.warn(f"No audio under {target}")
        return []

    selected = ops or list(OPS)
    out_root = root / "data" / "augmented"
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    rng = np.random.default_rng(seed)
    console.step(f"Augmenting {len(files)} file(s) x {variants} variant(s) with {selected} (-> data/augmented)")
    for i, path in enumerate(files, 1):
        if limit and i > limit:
            break
        try:
            audio, sr = _load(path)
        except Exception as exc:  # noqa: BLE001
            console.error(f"Load failed {path.name}: {exc}")
            continue

        made: List[dict] = []
        for op in selected:
            for v in range(max(1, variants)):
                try:
                    p = _rand_params(op, rng) if variants > 1 else _default_params(op)
                    out = _apply_op(op, audio, sr, p, rng)
                    name = f"{path.stem}__{op}.wav" if variants == 1 else f"{path.stem}__{op}_{v}.wav"
                    _save(out_root / name, out, sr)
                    made.append({
                        "op": op,
                        "path": str((out_root / name).relative_to(root)),
                        "params": p,
                    })
                except Exception as exc:  # noqa: BLE001
                    console.error(f"Augment {op} failed {path.name}: {exc}")

        results.append({"source": str(path.relative_to(root)), "variants": made})
        console.info(f"[{i}/{len(files)}] {path.name} -> {len(made)} variant(s)")

    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "augmented.json").write_text(json.dumps(results, indent=2))
    n = sum(len(r["variants"]) for r in results)
    console.ok(f"Augmented {len(results)} track(s) into {n} variant(s) -> data/augmented/")
    return results
