"""Segment normalized audio into fixed-length examples.

Two modes:
  * fast    — ffmpeg `-f segment` at a bar-rounded length (the default)
  * precise — explicit (start, end) windows when any of downbeat-alignment
              (#21), overlap (#24), or fades (#25) is requested, so cuts land
              on detected downbeats and boundaries are de-clicked.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .. import console
from ..config import Config
from ..util import sanitize_slug


def _bar_seconds(bpm: float, beats_per_bar: int) -> float:
    return beats_per_bar * 60.0 / bpm


def _segment_length(bpm: Optional[float], cfg: Config) -> float:
    scfg = cfg.segment
    target = scfg.segment_seconds
    if scfg.bar_aligned and bpm and bpm > 0:
        bar = _bar_seconds(float(bpm), scfg.beats_per_bar)
        if bar > 0:
            n_bars = max(1, int(target / bar))
            target = n_bars * bar
    return max(scfg.min_segment_seconds, target)


def _load_bpm_map(root: Path) -> Dict[str, Optional[float]]:
    """Read bpm keyed by song stem from metadata/manifest.jsonl if present."""
    manifest = root / "metadata" / "manifest.jsonl"
    if not manifest.exists():
        return {}
    out: Dict[str, Optional[float]] = {}
    try:
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            stem = sanitize_slug(Path(rec.get("path", "")).stem)
            bpm = rec.get("bpm")
            out[stem] = float(bpm) if bpm is not None else None
    except Exception:  # noqa: BLE001
        return {}
    return out


def detect_downbeats(y, sr: int, beats_per_bar: int = 4) -> List[float]:
    """Return downbeat timestamps (seconds) via librosa beat tracking."""
    import librosa

    from .analysis import beat_grid

    return beat_grid(y, sr, hop_length=512, beats_per_bar=beats_per_bar)["downbeat_times"]


def compute_windows(
    duration: float,
    length: float,
    overlap: float,
    min_seconds: float,
) -> List[Tuple[float, float]]:
    """Fixed-stride windows with optional overlap (#24)."""
    if overlap >= length:
        overlap = max(0.0, length - min_seconds)
    stride = length - overlap
    if stride <= 0:
        stride = length
    windows: List[Tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + length, duration)
        if end - start >= min_seconds:
            windows.append((round(start, 4), round(end, 4)))
        if end >= duration:
            break
        start += stride
    return windows


def downbeat_windows(
    downbeats: List[float],
    bars_per_seg: int,
    duration: float,
) -> List[Tuple[float, float]]:
    """Windows aligned to detected downbeats (#21)."""
    if not downbeats:
        return []
    step = max(1, bars_per_seg)
    windows: List[Tuple[float, float]] = []
    for i in range(0, len(downbeats) - 1, step):
        start = downbeats[i]
        end = downbeats[i + step] if i + step < len(downbeats) else duration
        windows.append((round(start, 4), round(end, 4)))
    return windows


def _slice_args(src: Path, dst: Path, start: float, end: float, cfg: Config, fade: float) -> List[str]:
    dur = end - start
    args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ss", f"{start:.4f}",
        "-t", f"{dur:.4f}",
        "-ac", str(cfg.normalize.channels),
        "-ar", str(cfg.normalize.sample_rate),
        "-c:a", cfg.normalize.codec,
    ]
    if fade and fade > 0 and dur > 2 * fade:
        f = min(fade, dur / 2.0)
        args += [
            "-af",
            f"afade=t=in:st=0:d={f:.4f},afade=t=out:st={dur - f:.4f}:d={f:.4f}",
        ]
    args.append(str(dst))
    return args


def segment(root: Path, cfg: Config, force: bool = False, dry_run: bool = False,
            progress: Optional[Callable[[int, int], None]] = None) -> List[dict]:
    scfg = cfg.segment
    clean_dir = root / "data" / "clean"
    seg_dir = root / "data" / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    bpm_map = _load_bpm_map(root)
    files = sorted(clean_dir.glob("*.wav"))
    if not files:
        console.warn(f"No WAV files under {clean_dir}")
        return []

    precise = scfg.downbeat_aligned or scfg.overlap_seconds > 0 or scfg.fade_seconds > 0

    manifest: List[dict] = []
    console.step(f"Segmenting {len(files)} files -> data/segments")

    for idx, src in enumerate(files, 1):
        if progress:
            progress(idx, len(files))
        song_id = sanitize_slug(src.stem)
        bpm = bpm_map.get(song_id) or bpm_map.get(src.stem)
        length = _segment_length(bpm, cfg)

        existing = sorted(seg_dir.glob(f"{song_id}_seg*.wav"))
        if existing and not force:
            console.info(f"skip (exists) {song_id}")
            continue
        if force and existing:
            for p in existing:
                p.unlink()

        if dry_run:
            console.info(f"[dry-run] {src.name} -> {song_id}_seg%03d.wav @ {length:.2f}s")
            continue

        # ---- precise mode (downbeat / overlap / fade) ---------------------
        if precise:
            import soundfile as sf

            from .features import load_audio

            info = sf.info(src)
            duration = float(info.duration)

            windows: List[Tuple[float, float]]
            if scfg.downbeat_aligned:
                y, sr = load_audio(src, sr=32000)
                dbs = detect_downbeats(y, sr, scfg.beats_per_bar)
                if dbs:
                    bar = _bar_seconds(bpm, scfg.beats_per_bar) if bpm else length / scfg.beats_per_bar
                    bars_per_seg = max(1, int(round(length / bar))) if bar > 0 else 1
                    windows = downbeat_windows(dbs, bars_per_seg, duration)
                    if not windows:
                        windows = compute_windows(duration, length, scfg.overlap_seconds, scfg.min_segment_seconds)
                else:
                    windows = compute_windows(duration, length, scfg.overlap_seconds, scfg.min_segment_seconds)
            else:
                windows = compute_windows(duration, length, scfg.overlap_seconds, scfg.min_segment_seconds)

            kept = 0
            for i, (start, end) in enumerate(windows):
                dst = seg_dir / f"{song_id}_seg{i:03d}.wav"
                args = _slice_args(src, dst, start, end, cfg, scfg.fade_seconds)
                res = subprocess.run(args, capture_output=True, text=True)
                if res.returncode != 0:
                    console.error(f"Slice failed for {src.name} [{start:.2f}-{end:.2f}]: {res.stderr.strip()[:300]}")
                    continue
                manifest.append(
                    {
                        "path": str(dst.relative_to(root)),
                        "song_id": song_id,
                        "source": str(src.relative_to(root)),
                        "segment_index": i,
                        "start_time": start,
                        "end_time": end,
                        "segment_seconds": round(end - start, 3),
                        "overlap_seconds": scfg.overlap_seconds,
                        "fade_seconds": scfg.fade_seconds,
                    }
                )
                kept += 1
            console.ok(f"{song_id}: {kept} segments (precise)")
            continue

        # ---- fast mode (ffmpeg -f segment) --------------------------------
        args = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-f", "segment",
            "-segment_time", f"{length:.4f}",
            "-reset_timestamps", "1",
            "-ac", str(cfg.normalize.channels),
            "-ar", str(cfg.normalize.sample_rate),
            "-c:a", cfg.normalize.codec,
            str(seg_dir / f"{song_id}_seg%03d.wav"),
        ]
        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0:
            console.error(f"Segment failed for {src.name}: {res.stderr.strip()[:400]}")
            continue

        import soundfile as sf

        produced = sorted(seg_dir.glob(f"{song_id}_seg*.wav"))
        kept = 0
        for i, p in enumerate(produced):
            try:
                dur = float(sf.info(p).duration)
            except Exception:  # noqa: BLE001
                dur = 0.0
            if dur < scfg.min_segment_seconds:
                p.unlink()
                continue
            start = round(i * length, 3)
            end = round(start + dur, 3)
            manifest.append(
                {
                    "path": str(p.relative_to(root)),
                    "song_id": song_id,
                    "source": str(src.relative_to(root)),
                    "segment_index": i,
                    "start_time": start,
                    "end_time": end,
                    "segment_seconds": round(dur, 3),
                }
            )
            kept += 1
        console.ok(f"{song_id}: {kept} segments @ {length:.2f}s")

    out = root / "metadata" / "segments.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    console.ok(f"Wrote {len(manifest)} segment records -> {out.relative_to(root)}")
    return manifest
