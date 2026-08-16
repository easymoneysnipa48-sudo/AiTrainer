"""Deep signal analysis (advanced eval batch #13-#20).

Signal-level analyses that go beyond the Phase-2 ``analysis`` pipeline
(chords/key/structure). Everything is librosa/numpy — no extra models.

* **Tempo drift (#13)** — local BPM over sliding windows; flags rushing or
  dragging within a clip (the tempo is steady vs accelerating/decelerating).
* **Groove (#14)** — swing ratio + microtiming deviation from the beat grid.
* **Loudness (#15)** — LUFS + RMS envelope + dynamic-range percentiles.
* **Stereo width (#16)** — mid/side energy ratio and phase correlation
  (mono-compatibility).
* **Artifacts (#17)** — clicks/pops, clipping, DC offset, dropouts.
* **Spectral profile (#18)** — centroid, rolloff, bandwidth, flatness, crest.
* **Onset density map (#19)** — busy vs sparse regions over time.
* **Frequency masking (#20)** — dominant spectral peaks that collide.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .. import console
from ..config import Config
from .features import estimate_lufs
from .inventory import AUDIO_GLOB


# --------------------------------------------------------------------------- #
# #13 tempo drift
# --------------------------------------------------------------------------- #
def tempo_drift(y: np.ndarray, sr: int, hop_length: int = 512,
                window_seconds: float = 4.0) -> Optional[dict]:
    """Local BPM per sliding window; verdict on rushing/dragging."""
    import librosa

    duration = len(y) / sr
    win = int(window_seconds * sr)
    step = max(win // 2, sr)
    if duration < window_seconds * 2:
        return None

    bpms: List[float] = []
    starts: List[float] = []
    for t0 in range(0, len(y) - win + 1, step):
        seg = y[t0:t0 + win]
        tempo, _ = librosa.beat.beat_track(y=seg, sr=sr, hop_length=hop_length)
        tempo = float(np.atleast_1d(tempo)[0])
        if tempo > 0:
            bpms.append(tempo)
            starts.append(round(t0 / sr, 2))

    if len(bpms) < 2:
        return None
    arr = np.asarray(bpms, dtype=float)
    mean = float(arr.mean())
    # drift rate: BPM change per second (linear slope of tempo vs time)
    ts = np.asarray(starts, dtype=float)
    slope = float(np.polyfit(ts, arr, 1)[0])
    total_drift = slope * duration

    if mean <= 0:
        verdict = "unknown"
    elif abs(total_drift) / mean < 0.02:
        verdict = "steady"
    elif slope > 0:
        verdict = "rushing"   # speeding up
    else:
        verdict = "dragging"  # slowing down

    return {
        "windows": [{"t": s, "bpm": round(b, 2)} for s, b in zip(starts, bpms, strict=True)],
        "mean_bpm": round(mean, 2),
        "std_bpm": round(float(arr.std()), 2),
        "drift_bpm_per_sec": round(slope, 4),
        "total_drift": round(total_drift, 4),
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# #14 groove (swing + microtiming)
# --------------------------------------------------------------------------- #
def groove(y: np.ndarray, sr: int, hop_length: int = 512) -> Optional[dict]:
    """Swing ratio (off/on-beat energy) plus microtiming deviation."""
    import librosa

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo = float(np.atleast_1d(tempo)[0])
    if tempo <= 0 or len(beats) < 2:
        return None

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    period = 60.0 / tempo

    on, off = [], []
    for bt in beat_times[:-1]:
        on.append(float(np.interp(bt, times, onset)))
        off.append(float(np.interp(bt + period / 2.0, times, onset)))
    on_mean = float(np.mean(on)) if on else 0.0
    off_mean = float(np.mean(off)) if off else 0.0
    swing = off_mean / on_mean if on_mean > 1e-9 else None

    # microtiming: median deviation of the nearest onset to each beat (ms)
    if len(onset) == 0:
        microtiming_ms = None
    else:
        devs = []
        for bt in beat_times:
            if not len(times):
                break
            nearest = times[int(np.argmin(np.abs(times - bt)))]
            devs.append((nearest - bt) * 1000.0)
        microtiming_ms = round(float(np.median(devs)), 2)

    if swing is None:
        feel = "unknown"
    elif swing < 0.5:
        feel = "straight"
    elif swing < 0.9:
        feel = "moderate"
    else:
        feel = "swung"

    return {
        "swing_ratio": round(swing, 4) if swing is not None else None,
        "feel": feel,
        "microtiming_median_ms": microtiming_ms,
        "tempo": round(tempo, 2),
    }


# --------------------------------------------------------------------------- #
# #15 loudness profile (LUFS + RMS envelope + dynamic range)
# --------------------------------------------------------------------------- #
def loudness_profile(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    import librosa

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    rms_db = rms_db[rms_db > -120]  # drop pure silence for stable percentiles
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # envelope: downsample to <= 64 points
    n_env = min(len(times), 64)
    idx = np.linspace(0, len(rms) - 1, n_env).astype(int)
    envelope = [
        {"t": round(float(times[i]), 2), "rms_db": round(float(rms_db[i]), 2)}
        for i in idx
    ]

    def _pct(p: float) -> Optional[float]:
        return round(float(np.percentile(rms_db, p)), 2) if rms_db.size else None

    return {
        "lufs": estimate_lufs(y, sr),
        "peak_db": round(float(20 * np.log10(np.max(np.abs(y)) + 1e-12)), 2),
        "rms_p10_db": _pct(10),
        "rms_p90_db": _pct(90),
        "dynamic_range_db": round(_pct(90) - _pct(10), 2) if rms_db.size else None,
        "envelope": envelope,
    }


# --------------------------------------------------------------------------- #
# #16 stereo width + phase correlation
# --------------------------------------------------------------------------- #
def stereo_profile(path: Path) -> Optional[dict]:
    """Mid/side width and inter-channel phase correlation (mono-compatibility)."""
    import soundfile as sf

    try:
        audio, _sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001
        return None
    if audio.shape[1] < 2:
        return {"channels": 1, "note": "mono source"}

    left, right = audio[:, 0], audio[:, 1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    mid_e = float((mid ** 2).mean())
    side_e = float((side ** 2).mean())
    width = side_e / (mid_e + side_e + 1e-12)  # 0..1

    # phase correlation: normalized cross-correlation at zero lag
    lch = left - left.mean()
    r = right - right.mean()
    denom = (np.sqrt((lch ** 2).sum()) * np.sqrt((r ** 2).sum())) + 1e-12
    phase_corr = float((lch * r).sum() / denom)

    mono_ok = bool(phase_corr > 0.0)
    return {
        "channels": int(audio.shape[1]),
        "stereo_width": round(width, 4),
        "phase_correlation": round(phase_corr, 4),
        "mono_compatible": mono_ok,
        "note": "out-of-phase -> mono downmix cancels" if phase_corr < -0.3 else "",
    }


# --------------------------------------------------------------------------- #
# #17 artifact detection
# --------------------------------------------------------------------------- #
def detect_artifacts(y: np.ndarray, sr: int) -> dict:
    """Clicks/pops (impulsive spikes), clipping, DC offset, dropouts."""
    if len(y) < 4:
        return {"n_clicks": 0, "clipping_ratio": 0.0, "dc_offset": 0.0, "n_dropouts": 0}

    # clicks: first-difference spikes many sigma above the local median
    d = np.abs(np.diff(y))
    med = float(np.median(d)) + 1e-12
    clicks = int(np.sum(d > 20.0 * med))
    clicks = int(min(clicks, 1000))  # cap pathological counts

    clipping_ratio = float(np.mean(np.abs(y) >= 0.999))
    dc_offset = float(np.abs(np.mean(y)))

    # dropouts: sudden near-silent bursts inside otherwise non-silent audio
    hop = 512
    import librosa

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    # a dropout is a short region < -50 dB surrounded by louder content
    silent = rms_db < -50.0
    n_dropouts = 0
    run = 0
    for s in silent:
        if s:
            run += 1
        else:
            if 2 <= run <= 20:  # between ~23ms and ~230ms
                n_dropouts += 1
            run = 0
    if silent[-1] and 2 <= run <= 20:
        n_dropouts += 1

    return {
        "n_clicks": clicks,
        "clipping_ratio": round(clipping_ratio, 5),
        "dc_offset": round(dc_offset, 5),
        "n_dropouts": n_dropouts,
        "clean": clicks == 0 and clipping_ratio < 0.001 and dc_offset < 0.01 and n_dropouts == 0,
    }


# --------------------------------------------------------------------------- #
# #18 spectral profile
# --------------------------------------------------------------------------- #
def spectral_profile(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    import librosa

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    crest = float(np.max(np.abs(y)) / (np.sqrt((y ** 2).mean()) + 1e-12))
    return {
        "centroid_hz": round(float(centroid.mean()), 1),
        "rolloff_hz": round(float(rolloff.mean()), 1),
        "bandwidth_hz": round(float(bandwidth.mean()), 1),
        "flatness": round(float(flatness.mean()), 4),
        "crest_factor": round(crest, 2),
        "rms_mean": round(float(rms.mean()), 6),
    }


# --------------------------------------------------------------------------- #
# #19 onset density map
# --------------------------------------------------------------------------- #
def onset_density_map(y: np.ndarray, sr: int, hop_length: int = 512,
                      bin_seconds: float = 2.0) -> dict:
    import librosa

    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    duration = len(y) / sr

    bins = max(1, int(np.ceil(duration / bin_seconds)))
    counts = np.zeros(bins)
    for t in onset_times:
        counts[min(bins - 1, int(t // bin_seconds))] += 1

    density = counts / bin_seconds
    overall = len(onset_times) / duration if duration > 0 else 0.0
    peak_bin = int(np.argmax(density))
    return {
        "overall_onset_density": round(overall, 4),
        "bin_seconds": bin_seconds,
        "map": [
            {"t": round(i * bin_seconds, 1), "density": round(float(density[i]), 4)}
            for i in range(bins)
        ],
        "busiest_region_t": round(float(peak_bin * bin_seconds), 1),
        "sparsest_region_t": round(float(int(np.argmin(density)) * bin_seconds), 1),
    }


# --------------------------------------------------------------------------- #
# #20 frequency masking
# --------------------------------------------------------------------------- #
def frequency_masking(y: np.ndarray, sr: int, n_fft: int = 2048,
                      max_peaks: int = 6) -> dict:
    """Detect dominant spectral peaks that collide (masking candidates)."""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=n_fft))
    mag = S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # top peaks (local maxima)
    peak_idx = []
    for i in range(1, len(mag) - 1):
        if mag[i] > mag[i - 1] and mag[i] >= mag[i + 1]:
            peak_idx.append(i)
    peak_idx.sort(key=lambda i: -mag[i])
    peaks = [(freqs[i], float(mag[i])) for i in peak_idx[:max_peaks]]
    peaks.sort()

    collisions = []
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            f1, a1 = peaks[i]
            f2, a2 = peaks[j]
            if f1 <= 0:
                continue
            # within ~1 critical band (approx 1/3 octave) -> mask candidate
            if f2 / f1 < 1.26:
                louder, quieter = max(a1, a2), min(a1, a2)
                if quieter > 0 and louder / quieter > 4.0:
                    collisions.append({
                        "f1_hz": round(f1, 1), "f2_hz": round(f2, 1),
                        "amplitude_ratio": round(louder / quieter, 2),
                    })

    return {
        "dominant_peaks_hz": [round(f, 1) for f, _ in peaks],
        "n_collisions": len(collisions),
        "collisions": collisions,
        "masking_risk": len(collisions) > 0,
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def analyze_deep(path: Path) -> dict:
    """All signal-level analyses for one audio file."""
    from .features import load_audio

    y, sr = load_audio(path, sr=None)
    rec = {
        "path": str(path),
        "duration": round(len(y) / sr, 3),
        "tempo_drift": tempo_drift(y, sr),
        "groove": groove(y, sr),
        "loudness": loudness_profile(y, sr),
        "stereo": stereo_profile(path),
        "artifacts": detect_artifacts(y, sr),
        "spectral": spectral_profile(y, sr),
        "onset_density": onset_density_map(y, sr),
        "masking": frequency_masking(y, sr),
    }
    return rec


def deep(root: Path, cfg: Config, which: str = "clean", limit: int = 0,
         path: Optional[Path] = None) -> List[dict]:
    if path is not None:
        files = [path]
    else:
        target = root / "data" / which
        if not target.exists():
            console.error(f"Directory not found: {target}")
            return []
        files: List[Path] = []
        for pattern in AUDIO_GLOB:
            files.extend(sorted(target.glob(pattern)))
        files = sorted(set(files))

    if not files:
        console.warn("No audio files to deep-analyze.")
        return []

    records: List[dict] = []
    console.step(f"Deep-analyzing {len(files)} file(s) (drift, groove, loudness, stereo, artifacts…)")
    for i, p in enumerate(files, 1):
        if limit and i > limit:
            break
        try:
            rec = analyze_deep(p)
        except Exception as exc:  # noqa: BLE001
            console.error(f"Deep analysis failed {p.name}: {exc}")
            continue
        rec["path"] = str(p.relative_to(root)) if not path else str(p)
        records.append(rec)
        drift = (rec.get("tempo_drift") or {}).get("verdict", "—")
        artifacts = rec["artifacts"].get("n_clicks", 0)
        console.info(f"[{i}/{len(files)}] {p.name}: drift={drift} clicks={artifacts}")

    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "deep_analysis.json").write_text(json.dumps(records, indent=2, default=str))
    with (meta / "deep_analysis.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")
    console.ok(f"Wrote {len(records)} records -> metadata/deep_analysis.json(l)")
    return records
