"""Audio-quality scoring: flag low bitrate, clipping, silence, DC offset, and
lowpass artifacts so poor sources never reach training.

Phase 1 dataset-hygiene feature #4.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from .. import console
from ..config import Config, QualityCfg

AUDIO_GLOB = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.aiff", "*.aif", "*.ogg")


def _scan(dir_path: Path) -> List[Path]:
    found: List[Path] = []
    for pattern in AUDIO_GLOB:
        found.extend(sorted(dir_path.glob(pattern)))
    return sorted(set(found))


def analyze_file(path: Path, cfg: QualityCfg) -> dict:
    """Return raw quality metrics + a 0-100 score + flags for one file."""
    import librosa
    import soundfile as sf

    info = sf.info(path)
    sr = int(info.samplerate)
    y, sr = librosa.load(path, sr=None, mono=True)
    duration = len(y) / sr

    bitrate_kbps = (path.stat().st_size * 8) / (duration * 1000) if duration > 0 else 0.0

    clipping_ratio = float(np.mean(np.abs(y) >= 0.999))
    dc_offset = float(np.abs(np.mean(y)))

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    silence_ratio = float(np.mean(rms < (10 ** (-60.0 / 20.0))))

    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))
    # spectral flatness: geometric/arithmetic mean of the power spectrum (noise proxy)
    S = np.abs(librosa.stft(y, hop_length=hop))
    with np.errstate(divide="ignore", invalid="ignore"):
        flatness = np.exp(np.mean(np.log(S + 1e-12))) / (np.mean(S) + 1e-12)
    spectral_flatness = float(np.mean(flatness))

    flags: List[str] = []
    score = 100.0

    if bitrate_kbps and bitrate_kbps < cfg.min_bitrate_kbps:
        flags.append(f"low bitrate ({bitrate_kbps:.0f} kbps)")
        score -= min(30.0, (cfg.min_bitrate_kbps - bitrate_kbps) / cfg.min_bitrate_kbps * 60.0)
    if sr < cfg.min_sample_rate:
        flags.append(f"low sample rate ({sr} Hz)")
        score -= 25.0
    if clipping_ratio > cfg.max_clipping_ratio:
        flags.append(f"clipping {clipping_ratio * 100:.2f}%")
        score -= min(25.0, clipping_ratio / max(cfg.max_clipping_ratio, 1e-9) * 12.0)
    if silence_ratio > cfg.max_silence_ratio:
        flags.append(f"silence {silence_ratio * 100:.0f}%")
        score -= 15.0
    if dc_offset > cfg.max_dc_offset:
        flags.append(f"DC offset {dc_offset:.4f}")
        score -= 15.0
    if rolloff < cfg.min_rolloff_hz:
        flags.append(f"lowpass rolloff {rolloff / 1000:.1f} kHz")
        score -= 20.0
    if spectral_flatness > 0.15:
        flags.append(f"noisy (flatness {spectral_flatness:.2f})")
        score -= 10.0

    score = round(max(0.0, min(100.0, score)), 1)
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "F"

    return {
        "path": str(path),
        "duration": round(duration, 3),
        "sample_rate": sr,
        "channels": int(info.channels),
        "bitrate_kbps": round(bitrate_kbps, 1),
        "clipping_ratio": round(clipping_ratio, 5),
        "silence_ratio": round(silence_ratio, 4),
        "dc_offset": round(dc_offset, 5),
        "rolloff_hz": round(rolloff, 1),
        "spectral_flatness": round(spectral_flatness, 4),
        "quality_score": score,
        "grade": grade,
        "flags": flags,
    }


def quality(root: Path, cfg: Config, which: str = "clean", limit: int = 0) -> List[dict]:
    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return []

    files = _scan(target)
    if not files:
        console.warn(f"No audio files under {target}")
        return []

    results: List[dict] = []
    console.step(f"Scoring quality of {len(files)} files (data/{which})")
    for i, path in enumerate(files, 1):
        if limit and i > limit:
            break
        try:
            rec = analyze_file(path, cfg.quality)
        except Exception as exc:  # noqa: BLE001
            console.error(f"Failed {path.name}: {exc}")
            rec = {"path": str(path), "error": str(exc), "quality_score": 0, "grade": "F", "flags": ["decode failure"]}
        rec["path"] = str(path.relative_to(root))
        results.append(rec)
        console.info(f"[{i}/{len(files)}] {path.name}: score={rec.get('quality_score')} grade={rec.get('grade')}")

    out = root / "metadata" / "quality_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    bad = [r for r in results if r.get("grade") in ("C", "F")]
    console.ok(f"Wrote {len(results)} scores -> metadata/quality_report.json ({len(bad)} flagged C/F)")
    return results
