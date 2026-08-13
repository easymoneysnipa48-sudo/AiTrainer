"""Standalone loudness normalization via FFmpeg loudnorm (ITU-R BS.1770).

Normalizes already-clean audio in data/<which> to a target LUFS so every
training example sits at a consistent perceived loudness (Phase 1 feature #3).
Writes in place only with --force; otherwise dry-run.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from . import console
from .audio.inventory import AUDIO_GLOB
from .config import Config


def _scan(dir_path: Path) -> List[Path]:
    found: List[Path] = []
    for pattern in AUDIO_GLOB:
        found.extend(sorted(dir_path.glob(pattern)))
    return sorted(set(found))


def loudnorm(
    root: Path,
    cfg: Config,
    which: str = "clean",
    target_lufs: float = -14.0,
    force: bool = False,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return 0, 0, 0

    files = _scan(target)
    if not files:
        console.warn(f"No audio files under {target}")
        return 0, 0, 0

    converted = skipped = failed = 0
    console.step(f"Loudness-normalizing {len(files)} files -> {target_lufs:.1f} LUFS (data/{which})")
    for i, path in enumerate(files, 1):
        if dry_run:
            console.info(f"[dry-run] {path.name} -> {target_lufs} LUFS")
            continue
        fd, tmp = tempfile.mkstemp(suffix=".wav", dir=str(target))
        import os

        os.close(fd)
        args = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-ar", "32000", "-ac", "1", "-c:a", "pcm_s16le",
            tmp,
        ]
        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0:
            console.error(f"FFmpeg failed for {path.name}: {res.stderr.strip()[:300]}")
            failed += 1
            Path(tmp).unlink(missing_ok=True)
            continue
        if not force:
            # keep original, report only
            console.info(f"[{i}/{len(files)}] would write {path.name} ({target_lufs} LUFS)")
            skipped += 1
            Path(tmp).unlink(missing_ok=True)
            continue
        Path(tmp).replace(path)
        console.ok(f"[{i}/{len(files)}] {path.name} -> {target_lufs} LUFS")
        converted += 1

    console.ok(f"Done: {converted} normalized, {skipped} skipped, {failed} failed")
    return converted, skipped, failed
