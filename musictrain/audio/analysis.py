"""Deep audio analysis (Phase 2): chords, beat/downbeat grid, key confidence,
onset density, tempo curve, swing, structure detection, and vocal/instrumental.

Everything here is librosa-based (already a dependency), except vocal detection
(#14) and timbre embeddings (#17), which reuse the CLAP model already used for
prompt adherence — so no new heavy model downloads are required.

The entry point is :func:`analyze`, which scans a directory (or a single file)
and writes ``metadata/analysis.jsonl`` + ``metadata/analysis.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import console
from ..config import AnalysisCfg, Config
from .features import _MAJOR, _MINOR, _PC_NAMES, load_audio

# --------------------------------------------------------------------------- #
# Key confidence (#13) — Krumhansl-Schmuckler with softmax confidence
# --------------------------------------------------------------------------- #


def key_candidates(y: np.ndarray, sr: int, top_k: int = 3) -> dict:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean = chroma.mean(axis=1)

    scored: List[Tuple[str, float]] = []
    for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
        for i in range(12):
            rolled = np.roll(np.array(profile, dtype=float), i)
            r = float(np.corrcoef(mean, rolled)[0, 1])
            scored.append((f"{_PC_NAMES[i]} {mode}", r))
    scored.sort(key=lambda t: t[1], reverse=True)

    # softmax over correlations -> probabilities that sum to 1
    rs = np.array([r for _, r in scored])
    rs = rs - rs.max()
    exps = np.exp(rs * 6.0)  # temperature sharpens differences
    probs = exps / exps.sum()

    top = scored[0]
    return {
        "key": top[0],
        "confidence": round(float(probs[0]), 4),
        "candidates": [
            {"key": name, "prob": round(float(probs[i]), 4)}
            for i, (name, _) in enumerate(scored[:top_k])
        ],
    }


# --------------------------------------------------------------------------- #
# Chords (#11) — chroma template matching against major/minor triads
# --------------------------------------------------------------------------- #

_CHORD_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _chord_templates() -> Tuple[List[str], np.ndarray]:
    """24 templates: 12 roots x {major, minor} triads."""
    names: List[str] = []
    tmpl: List[np.ndarray] = []
    for i, root in enumerate(_CHORD_ROOTS):
        for quality, offsets in (("", [0, 4, 7]), ("m", [0, 3, 7])):
            v = np.zeros(12)
            for off in offsets:
                v[(i + off) % 12] = 1.0
            v = v / (np.linalg.norm(v) + 1e-12)
            names.append(f"{root}{quality}")
            tmpl.append(v)
    return names, np.stack(tmpl)


def extract_chords(
    y: np.ndarray, sr: int, hop_length: int = 512, frame_seconds: float = 0.5
) -> List[dict]:
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-12)
    names, tmpl = _chord_templates()

    hop = hop_length / sr
    frames_per_label = max(1, int(round(frame_seconds / hop)))
    n_frames = chroma.shape[1]

    chords: List[dict] = []
    for start in range(0, n_frames, frames_per_label):
        end = min(start + frames_per_label, n_frames)
        seg = chroma[:, start:end].mean(axis=1)
        seg = seg / (np.linalg.norm(seg) + 1e-12)
        sims = tmpl @ seg
        best = int(np.argmax(sims))
        t = round(start * hop, 3)
        chords.append(
            {"t": t, "chord": names[best], "confidence": round(float(sims[best]), 4)}
        )
    return chords


# --------------------------------------------------------------------------- #
# Beat / downbeat grid (#12)
# --------------------------------------------------------------------------- #


def beat_grid(y: np.ndarray, sr: int, hop_length: int = 512, beats_per_bar: int = 4) -> dict:
    import librosa

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    # downbeat phase: pick the offset (0..beats_per_bar-1) whose beats carry the
    # most onset energy — the loudest accent of each bar is usually the downbeat.
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    phase_scores = []
    for p in range(beats_per_bar):
        idx = beats[p::beats_per_bar]
        idx = idx[idx < len(onset)]
        phase_scores.append(float(onset[idx].mean()) if len(idx) else 0.0)
    downbeat_phase = int(np.argmax(phase_scores))
    downbeats = beats[downbeat_phase::beats_per_bar]
    downbeat_times = librosa.frames_to_time(downbeats, sr=sr, hop_length=hop_length)

    return {
        "tempo": round(tempo, 2),
        "beats_per_bar": beats_per_bar,
        "n_beats": int(len(beats)),
        "beat_times": [round(float(t), 3) for t in beat_times],
        "downbeat_phase": downbeat_phase,
        "downbeat_times": [round(float(t), 3) for t in downbeat_times],
    }


# --------------------------------------------------------------------------- #
# Onset density / rhythmic complexity (#15)
# --------------------------------------------------------------------------- #


def onset_stats(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    import librosa

    duration = len(y) / sr
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)

    density = len(onset_times) / duration if duration > 0 else 0.0
    ioi = np.diff(onset_times)
    mean_ioi = float(np.mean(ioi)) if len(ioi) else None
    cv_ioi = float(np.std(ioi) / (np.mean(ioi) + 1e-12)) if len(ioi) else None

    # rhythmic complexity: CV of inter-onset intervals (higher = more irregular)
    complexity = round(cv_ioi, 4) if cv_ioi is not None else None
    return {
        "n_onsets": int(len(onset_times)),
        "onset_density": round(density, 4),
        "mean_ioi": round(mean_ioi, 4) if mean_ioi is not None else None,
        "rhythmic_complexity": complexity,
    }


# --------------------------------------------------------------------------- #
# Tempo curve (#19) — per-frame tempo downsampled into bins
# --------------------------------------------------------------------------- #


def tempo_curve(y: np.ndarray, sr: int, hop_length: int = 512, bin_seconds: float = 2.0) -> dict:
    import librosa

    try:
        # librosa 1.x moved tempo to librosa.feature.tempo
        if hasattr(librosa.feature, "tempo"):
            per_frame = librosa.feature.tempo(y=y, sr=sr, hop_length=hop_length, aggregate=None)
        else:
            per_frame = librosa.beat.tempo(y=y, sr=sr, hop_length=hop_length, aggregate=None)
        per_frame = np.asarray(per_frame).reshape(-1)
    except Exception:  # noqa: BLE001
        per_frame = np.asarray([])

    if per_frame.size == 0:
        return {"mean": None, "median": None, "std": None, "curve": []}

    valid = per_frame[np.isfinite(per_frame)]
    if valid.size == 0:
        return {"mean": None, "median": None, "std": None, "curve": []}

    # tempo per frame -> per bin (median within each bin)
    frame_sec = hop_length / sr
    duration = len(y) / sr
    bins = max(1, int(np.ceil(duration / bin_seconds)))
    frames_per_bin = max(1, int(round(bin_seconds / frame_sec)))

    curve = []
    for b in range(bins):
        seg = valid[b * frames_per_bin : (b + 1) * frames_per_bin]
        if seg.size:
            curve.append({"t": round(b * bin_seconds, 1), "bpm": round(float(np.median(seg)), 2)})

    return {
        "mean": round(float(np.mean(valid)), 2),
        "median": round(float(np.median(valid)), 2),
        "std": round(float(np.std(valid)), 2),
        "curve": curve,
    }


# --------------------------------------------------------------------------- #
# Swing / groove (#20) — off-beat vs on-beat onset energy
# --------------------------------------------------------------------------- #


def swing_ratio(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    import librosa

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo = float(np.atleast_1d(tempo)[0])
    if tempo <= 0 or len(beats) < 2:
        return {"ratio": None, "feel": "unknown"}

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    beat_period = 60.0 / tempo
    on, off = [], []
    for bt in beat_times[:-1]:
        on.append(float(np.interp(bt, times, onset)))
        off.append(float(np.interp(bt + beat_period / 2.0, times, onset)))

    on_mean = float(np.mean(on)) if on else 0.0
    off_mean = float(np.mean(off)) if off else 0.0
    ratio = off_mean / on_mean if on_mean > 1e-9 else None

    if ratio is None:
        feel = "unknown"
    elif ratio < 0.5:
        feel = "straight"       # strong downbeats, weak off-beats
    elif ratio < 0.9:
        feel = "moderate"
    else:
        feel = "swung"          # pronounced off-beat / shuffle energy

    return {"ratio": round(ratio, 4) if ratio is not None else None, "feel": feel}


# --------------------------------------------------------------------------- #
# Structure detection (#18) — chroma clustering into labelled segments
# --------------------------------------------------------------------------- #


def detect_structure(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    min_segments: int = 2,
    max_segments: int = 8,
    segment_seconds: float = 10.0,
) -> dict:
    import librosa

    duration = len(y) / sr
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    n_frames = chroma.shape[1]
    if n_frames < 4:
        return {"n_segments": 1, "segments": [], "note": "audio too short for structure"}

    k = int(np.clip(round(duration / segment_seconds), min_segments, max_segments))
    k = min(k, n_frames)
    if k < 2:
        k = 2

    # librosa >=1.0 agglomerative returns boundary frame indices (not labels)
    boundaries = librosa.segment.agglomerative(chroma, k)
    boundaries = np.asarray(boundaries, dtype=int)
    boundaries = boundaries[boundaries < n_frames]
    boundaries = sorted(set([0, *boundaries.tolist(), n_frames]))

    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms = np.interp(np.linspace(0, 1, n_frames), np.linspace(0, 1, len(rms)), rms)

    # build segments from consecutive boundary pairs
    segments: List[dict] = []
    for idx, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if b <= a:
            continue
        seg_energy = float(rms[a:b].mean()) if b > a else 0.0
        segments.append(
            {
                "label": chr(65 + (idx % 26)),  # A, B, C, ...
                "start": round(float(times[a]), 3),
                "end": round(float(times[min(b, n_frames) - 1]), 3),
                "energy": round(seg_energy, 4),
            }
        )

    # heuristic role mapping (coarse, for guidance only)
    roles = _assign_roles(segments)
    for seg, role in zip(segments, roles):
        seg["role"] = role

    return {
        "n_segments": len(segments),
        "segments": segments,
        "note": "roles are energy/position heuristics, not ground truth",
    }


def _assign_roles(segments: List[dict]) -> List[str]:
    n = len(segments)
    if n == 1:
        return ["full-song"]
    energies = [s["energy"] for s in segments]
    loudest = int(np.argmax(energies))

    roles = ["verse"] * n
    roles[0] = "intro"
    roles[-1] = "outro"
    if n > 2:
        roles[loudest] = "chorus"
        # a quieter mid segment after a chorus is often a bridge
        if loudest > 0 and loudest < n - 1:
            for idx in range(1, n - 1):
                if idx != loudest and energies[idx] < 0.6 * energies[loudest]:
                    roles[idx] = "bridge"
                    break
    else:
        roles[loudest] = "chorus"
    return roles


# --------------------------------------------------------------------------- #
# Vocal vs instrumental (#14) — CLAP score against two text anchors
# --------------------------------------------------------------------------- #


def vocal_instrumental(cfg: Config, audio_path: Path) -> Optional[dict]:
    """Score audio against vocal vs instrumental anchors via CLAP."""
    if not cfg.clap.enabled or not cfg.analysis.vocal_enabled:
        return None

    from ..similarity import load_clap, resolve_device

    device = resolve_device(cfg.clap.device)
    fe, tok, model, device = load_clap(cfg.clap.model_name, device)

    import soundfile as sf
    import torch

    audio, asr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if asr != 48000:
        import librosa

        audio = librosa.resample(audio, orig_sr=asr, target_sr=48000)
        asr = 48000

    audio_inputs = fe(raw_speech=[audio], sampling_rate=asr, return_tensors="pt")
    audio_inputs = {k: v.to(device) for k, v in audio_inputs.items() if hasattr(v, "to")}

    with torch.inference_mode():
        aemb = model.get_audio_features(
            input_features=audio_inputs["input_features"],
            is_longer=audio_inputs.get("is_longer"),
        )
        if isinstance(aemb, tuple):
            aemb = aemb[0].pooler_output
        else:
            aemb = aemb.pooler_output

    anchors = {
        "vocal": "vocals, singing, rapping, voice",
        "instrumental": "instrumental, no vocals, no singing, instrumental music",
    }
    scores: Dict[str, float] = {}
    for name, text in anchors.items():
        text_inputs = tok([text], padding=True, return_tensors="pt")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items() if hasattr(v, "to")}
        with torch.inference_mode():
            temb = model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            if isinstance(temb, tuple):
                temb = temb[0].pooler_output
            else:
                temb = temb.pooler_output
        scores[name] = float((aemb @ temb.T)[0, 0].item())

    diff = scores["vocal"] - scores["instrumental"]
    if diff > 0.05:
        verdict = "vocal"
    elif diff < -0.05:
        verdict = "instrumental"
    else:
        verdict = "ambiguous"

    return {
        "verdict": verdict,
        "vocal_score": round(scores["vocal"], 4),
        "instrumental_score": round(scores["instrumental"], 4),
    }


# --------------------------------------------------------------------------- #
# Timbre embedding (#17) — reuse the existing CLAP audio embedding
# --------------------------------------------------------------------------- #


def timbre_embedding(cfg: Config, audio_path: Path) -> Optional[dict]:
    if not cfg.clap.enabled:
        return None
    from ..embeddings import embed_audio

    emb = embed_audio(cfg, audio_path)
    return {"dim": int(emb.shape[0]), "norm": round(float(np.linalg.norm(emb)), 4)}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def analyze_file(cfg: Config, path: Path, root: Path) -> dict:
    acfg = cfg.analysis
    y, sr = load_audio(path, sr=acfg.sr)
    rel = str(path.relative_to(root))

    rec: dict = {
        "path": rel,
        "duration": round(len(y) / sr, 3),
        "key": key_candidates(y, sr, top_k=acfg.key_top_k),
        "chords": extract_chords(y, sr, hop_length=acfg.hop_length, frame_seconds=acfg.chord_frame),
        "beat_grid": beat_grid(
            y, sr, hop_length=acfg.hop_length, beats_per_bar=acfg.beats_per_bar
        ),
        "onsets": onset_stats(y, sr, hop_length=acfg.hop_length),
        "tempo_curve": tempo_curve(y, sr, hop_length=acfg.hop_length),
        "swing": swing_ratio(y, sr, hop_length=acfg.hop_length),
        "structure": detect_structure(
            y,
            sr,
            hop_length=acfg.hop_length,
            min_segments=acfg.structure_min_segments,
            max_segments=acfg.structure_max_segments,
            segment_seconds=acfg.structure_segment_seconds,
        ),
    }
    if acfg.vocal_enabled and cfg.clap.enabled:
        rec["vocal"] = vocal_instrumental(cfg, path)
        rec["timbre"] = timbre_embedding(cfg, path)
    return rec


def analyze(
    root: Path,
    cfg: Config,
    which: str = "clean",
    limit: int = 0,
    path: Optional[Path] = None,
) -> List[dict]:
    from .inventory import AUDIO_GLOB

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
        console.warn("No audio files to analyze.")
        return []

    records: List[dict] = []
    console.step(f"Deep-analyzing {len(files)} file(s) (chords, grid, key, structure…)")
    for i, p in enumerate(files, 1):
        if limit and i > limit:
            break
        try:
            rec = analyze_file(cfg, p, root)
        except Exception as exc:  # noqa: BLE001
            console.error(f"Analysis failed {p.name}: {exc}")
            continue
        records.append(rec)
        key = rec["key"]["key"]
        tempo = rec["beat_grid"]["tempo"]
        nseg = rec["structure"]["n_segments"]
        console.info(f"[{i}/{len(files)}] {p.name}: key={key} bpm={tempo} segs={nseg}")

    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "analysis.json").write_text(json.dumps(records, indent=2))
    with (meta / "analysis.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    console.ok(f"Wrote {len(records)} records -> metadata/analysis.json(l)")
    return records
