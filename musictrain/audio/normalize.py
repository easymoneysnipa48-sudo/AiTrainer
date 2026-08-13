"""Audio normalization via FFmpeg: data/raw -> data/clean."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

from .. import console
from ..config import Config, NormalizeCfg
from ..util import unique_path


def find_audio(root: Path, extensions: List[str]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for ext in extensions:
        ext = ext.lower()
        for pattern in (f"*{ext}", f"*{ext.upper()}"):
            for p in sorted(root.rglob(pattern)):
                if p not in seen:
                    seen.add(p)
                    found.append(p)
    return found


def _ffmpeg_args(src: Path, dst: Path, cfg: NormalizeCfg) -> List[str]:
    args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
    ]
    if cfg.strip_metadata:
        args += ["-map_metadata", "-1"]
    args += ["-ac", str(cfg.channels), "-ar", str(cfg.sample_rate)]
    if cfg.target_lufs is not None:
        args += ["-af", f"loudnorm=I={cfg.target_lufs}:TP=-1.5:LRA=11"]
    args += ["-c:a", cfg.codec, str(dst)]
    return args


def _dst_path(src: Path, root: Path, clean_dir: Path, cfg: NormalizeCfg) -> Path:
    """Collision-safe destination path under data/clean."""
    rel = src.relative_to(root) if src.is_relative_to(root) else Path(src.name)
    name = rel.with_suffix(".wav").name if cfg.codec != "flac" else rel.with_suffix(".flac").name
    dst = clean_dir / name
    if dst == src:
        dst = unique_path(dst)
    return dst


def normalize(
    root: Path,
    cfg: Config,
    force: bool = False,
    dry_run: bool = False,
    limit: int = 0,
) -> Tuple[int, int, int]:
    ncfg = cfg.normalize
    raw_dir = root / "data" / "raw"
    clean_dir = root / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio(raw_dir, ncfg.extensions)
    if not files:
        console.warn(f"No audio found under {raw_dir}")
        return 0, 0, 0

    converted = skipped = failed = 0
    console.step(f"Normalizing {len(files)} files -> {clean_dir.relative_to(root)}")

    for i, src in enumerate(files, 1):
        if limit and i > limit:
            break
        dst = _dst_path(src, root, clean_dir, ncfg)
        if dst.exists() and not force:
            console.info(f"[{i}/{len(files)}] skip (exists) {src.name}")
            skipped += 1
            continue
        if dry_run:
            console.info(f"[dry-run] {src.name} -> {dst.name}")
            continue
        args = _ffmpeg_args(src, dst, ncfg)
        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0:
            console.error(f"FFmpeg failed for {src.name}: {res.stderr.strip()[:400]}")
            failed += 1
        else:
            console.ok(f"[{i}/{len(files)}] {dst.name}")
            converted += 1

    console.ok(f"Done: {converted} converted, {skipped} skipped, {failed} failed")
    return converted, skipped, failed
