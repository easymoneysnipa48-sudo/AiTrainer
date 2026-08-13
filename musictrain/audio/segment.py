"""Segment normalized audio into fixed-length examples with optional bar alignment."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

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


def segment(root: Path, cfg: Config, force: bool = False, dry_run: bool = False) -> List[dict]:
    scfg = cfg.segment
    clean_dir = root / "data" / "clean"
    seg_dir = root / "data" / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    bpm_map = _load_bpm_map(root)
    files = sorted(clean_dir.glob("*.wav"))
    if not files:
        console.warn(f"No WAV files under {clean_dir}")
        return []

    manifest: List[dict] = []
    console.step(f"Segmenting {len(files)} files -> data/segments")

    for src in files:
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
