"""Signal-level feature extraction: BPM, key, loudness, peak, silence, clipping."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..config import FeaturesCfg

# Krumhansl-Schmuckler key profiles (major / minor), standard music theory
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load_audio(path: Path, sr: int = 32000):
    """Load any supported format (wav/flac/mp3/m4a/aiff/ogg) -> (y, sr).

    Uses the universal decoder, which falls back to an ffmpeg conversion for
    AAC-family files (m4a) that librosa/soundfile can't open.
    """
    from .decode import load_any

    y, sr = load_any(path, sr=sr, mono=True)
    return y, sr


def estimate_bpm(y: np.ndarray, sr: int) -> Optional[float]:
    import librosa

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        if tempo <= 0:
            return None
        return round(tempo, 2)
    except Exception:  # noqa: BLE001
        return None


def estimate_key(y: np.ndarray, sr: int) -> Optional[str]:
    import librosa

    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        mean = chroma.mean(axis=1)
        best, best_corr = None, -np.inf
        for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
            for i in range(12):
                rolled = np.roll(np.array(profile, dtype=float), i)
                corr = np.corrcoef(mean, rolled)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best = f"{_PC_NAMES[i]} {mode}"
        return best
    except Exception:  # noqa: BLE001
        return None


def estimate_lufs(y: np.ndarray, sr: int) -> Optional[float]:
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        return round(float(meter.integrated_loudness(y)), 2)
    except Exception:  # noqa: BLE001
        return None


def compute_features(y: np.ndarray, sr: int, cfg: FeaturesCfg) -> dict:
    import librosa

    duration = len(y) / sr

    rms = librosa.feature.rms(y=y, hop_length=cfg.hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_db = float(np.mean(librosa.amplitude_to_db(rms, ref=1.0)))

    peak = float(np.max(np.abs(y)))
    peak_db = float(20 * np.log10(peak + 1e-12))

    # energy proxy: map mean RMS in dBFS from [-60, 0] onto [0, 1]
    energy = float(np.clip((rms_db + 60.0) / 60.0, 0.0, 1.0))

    silence_thresh = 10 ** (cfg.silence_threshold_db / 20.0)
    silence_ratio = float(np.mean(rms < silence_thresh))
    clipping_ratio = float(np.mean(np.abs(y) >= cfg.clip_threshold))

    return {
        "duration": round(duration, 3),
        "sample_rate": int(sr),
        "channels": 1,
        "bpm": estimate_bpm(y, sr),
        "key": estimate_key(y, sr),
        "rms": round(rms_mean, 6),
        "rms_db": round(rms_db, 2),
        "peak": round(peak, 6),
        "peak_db": round(peak_db, 2),
        "lufs": estimate_lufs(y, sr),
        "silence_ratio": round(silence_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 4),
        "energy": round(energy, 3),
    }


def extract_file(path: Path, cfg: FeaturesCfg) -> dict:
    y, sr = load_audio(path, sr=cfg.sr)
    return compute_features(y, sr, cfg)
