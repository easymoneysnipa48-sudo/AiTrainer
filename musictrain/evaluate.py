"""Post-generation checks: BPM drift, time-stretch correction, bar alignment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import console
from .audio.features import estimate_bpm, load_audio
from .config import Config
from .util import now_stamp


def _bar_samples(target_bpm: float, beats_per_bar: int, sr: int) -> int:
    return int(round(beats_per_bar * 60.0 / target_bpm * sr))


def check(
    cfg: Config,
    audio_path: Path,
    target_bpm: Optional[float] = None,
    fix: bool = False,
    out_dir: Optional[Path] = None,
) -> dict:
    import soundfile as sf

    ccfg = cfg.check
    y, sr = load_audio(audio_path, sr=ccfg.sr)
    detected = estimate_bpm(y, sr)

    report: dict = {
        "path": str(audio_path),
        "sample_rate": sr,
        "duration": round(len(y) / sr, 3),
        "detected_bpm": detected,
        "target_bpm": target_bpm,
    }

    if detected is None:
        report["status"] = "undetected"
        console.warn(f"{audio_path.name}: could not detect BPM")
        return report

    if target_bpm is None:
        report["status"] = "measured"
        report["deviation"] = None
        console.info(f"{audio_path.name}: BPM = {detected}")
        return report

    raw_deviation = (detected - target_bpm) / target_bpm

    # Octave ambiguity: sparse or percussive content is frequently detected at
    # double/half/quadruple/quarter time. Fold by those ratios and see if any
    # matches the target (closest folds first).
    octave_note = None
    folded_bpm = None
    if abs(raw_deviation) > ccfg.bpm_tolerance:
        folds = (
            (0.5, "double-time (0.5x octave)"),
            (2.0, "half-time (2x octave)"),
            (0.25, "quadruple-time (0.25x octave)"),
            (4.0, "quarter-time (4x octave)"),
        )
        for factor, label in folds:
            folded = detected * factor
            if abs((folded - target_bpm) / target_bpm) <= ccfg.bpm_tolerance:
                octave_note = label
                folded_bpm = round(folded, 2)
                break

    # The reported deviation is the adherence-correct one: folded clips that
    # are within tolerance report the (small) folded deviation, so downstream
    # auto-reject / leaderboard / significance treat them as on-target.
    deviation = raw_deviation
    if octave_note and folded_bpm is not None:
        deviation = (folded_bpm - target_bpm) / target_bpm
    report["deviation"] = round(deviation, 4)
    report["raw_deviation"] = round(raw_deviation, 4)

    if abs(raw_deviation) <= ccfg.bpm_tolerance:
        report["status"] = "ok"
        console.ok(f"{audio_path.name}: {detected} BPM vs {target_bpm} (dev {deviation:+.2%})")
    elif octave_note:
        report["status"] = "ok"
        report["note"] = f"detected tempo is {octave_note} of target"
        report["folded_bpm"] = folded_bpm
        console.ok(
            f"{audio_path.name}: {detected} BPM ≈ {octave_note} of target {target_bpm} "
            f"(folded dev {deviation:+.2%})"
        )
    elif fix and abs(deviation) <= ccfg.max_time_stretch:
        rate = float(target_bpm) / detected
        report["status"] = "stretched"
        report["stretch_rate"] = round(rate, 4)
        console.info(f"{audio_path.name}: stretching by {rate:.4f} ({detected}->{target_bpm})")
        import librosa

        y2 = librosa.effects.time_stretch(y, rate=rate)
        # trim to whole bars
        bar = _bar_samples(target_bpm, ccfg.beats_per_bar, sr)
        if bar > 0:
            y2 = y2[: (len(y2) // bar) * bar]
        out_dir = Path(out_dir) if out_dir else audio_path.parent
        out_path = out_dir / f"{audio_path.stem}_fixed{audio_path.suffix}"
        sf.write(out_path, y2, sr)
        report["fixed_path"] = str(out_path)
        report["fixed_duration"] = round(len(y2) / sr, 3)
    else:
        report["status"] = "rejected"
        console.warn(
            f"{audio_path.name}: {detected} BPM vs {target_bpm} "
            f"(dev {deviation:+.2%} exceeds tolerance)"
        )

    return report


def check_dir(
    cfg: Config,
    dir_path: Path,
    target_bpm: Optional[float] = None,
    fix: bool = False,
) -> List[dict]:
    files = sorted(dir_path.glob("*.wav"))
    if not files:
        console.warn(f"No WAV files in {dir_path}")
        return []

    reports = [check(cfg, p, target_bpm=target_bpm, fix=fix, out_dir=dir_path) for p in files]

    report_path = dir_path / f"bpm_report_{now_stamp()}.json"
    report_path.write_text(json.dumps(reports, indent=2))
    console.ok(f"Report -> {report_path}")

    ok = sum(1 for r in reports if r["status"] in ("ok", "stretched"))
    console.ok(f"{ok}/{len(reports)} outputs within tolerance")
    return reports
