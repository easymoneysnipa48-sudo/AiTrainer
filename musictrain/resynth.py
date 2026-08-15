"""Stems re-synthesis (Advanced #21).

Takes Demucs output in ``data/stems/<track>/`` and re-synthesizes a mix with
per-stem gain/EQ, so you can build training variants (e.g. "vocal-up" mixes,
instrumental-dominant versions) without re-running separation.

Reads the ``metadata/stems.json`` manifest written by :func:`stems.separate_stems`
and writes re-mixed files into ``data/resynth/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import console
from .config import Config


def _read_wav(path: Path):
    import soundfile as sf

    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, sr)
    console.ok(f"Wrote {path.name} ({len(audio) / sr:.1f}s)")


def _load_manifest(root: Path) -> List[dict]:
    p = root / "metadata" / "stems.json"
    if not p.exists():
        console.error("No metadata/stems.json — run `musictrain stems` first.")
        return []
    return json.loads(p.read_text())


def _mix(audios: Dict[str, np.ndarray], gains: Dict[str, float], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    for name, a in audios.items():
        g = gains.get(name, 1.0)
        if g == 0.0 or g is None:
            continue
        m = min(len(a), n)
        out[:m] += g * a[:m]
    return out


def resynth(
    root: Path,
    cfg: Config,
    which: str = "clean",
    gains: Optional[Dict[str, float]] = None,
    limit: int = 0,
    out_dir: str = "resynth",
) -> List[dict]:
    """Re-mix stems with per-stem gains.

    ``gains`` maps stem name -> linear gain (e.g. {"vocals": 1.2, "drums": 0.5}).
    Stems missing from ``gains`` keep their original level.
    """
    manifest = _load_manifest(root)
    if not manifest:
        return []

    out_root = root / "data" / out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    gains = gains or {}
    results: List[dict] = []
    for i, rec in enumerate(manifest, 1):
        if limit and i > limit:
            break
        track_rel = rec["track"]
        if not (root / track_rel).exists():
            continue
        # only re-mix tracks whose source lives under the requested dir
        if which != "clean" and not track_rel.startswith(f"data/{which}"):
            continue

        audios: Dict[str, np.ndarray] = {}
        sr = None
        for name, rel in rec["stems"].items():
            p = root / rel
            if not p.exists():
                continue
            a, s = _read_wav(p)
            audios[name] = a
            sr = s if sr is None else sr

        if not audios or sr is None:
            console.warn(f"Skip {track_rel}: no stem files found")
            continue

        n = max(len(a) for a in audios.values())
        effective = {name: gains.get(name, 1.0) for name in audios}
        if all(g <= 0 for g in effective.values()):
            console.warn(f"Skip {track_rel}: every present stem has gain 0")
            continue
        mix = _mix(audios, gains, n)

        stem = Path(track_rel).stem
        out = out_root / f"{stem}_resynth.wav"
        try:
            _write_wav(out, mix, sr)
        except Exception as exc:  # noqa: BLE001
            console.error(f"Failed {track_rel}: {exc}")
            continue

        results.append(
            {
                "source": track_rel,
                "output": str(out.relative_to(root)),
                "gains": dict(gains),
                "stems_used": list(audios.keys()),
            }
        )
        console.info(f"[{i}/{len(manifest)}] {stem} -> {out.name}")

    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "resynth.json").write_text(json.dumps(results, indent=2))
    console.ok(f"Re-synthesized {len(results)} mix(es) -> data/{out_dir}/ (metadata/resynth.json)")
    return results


def rebuild_instrumental(root: Path, cfg: Config, limit: int = 0) -> List[dict]:
    """Rebuild an instrumental mix by dropping the vocals stem entirely."""
    return resynth(
        root,
        cfg,
        gains={"vocals": 0.0},
        limit=limit,
        out_dir="resynth_instrumental",
    )
