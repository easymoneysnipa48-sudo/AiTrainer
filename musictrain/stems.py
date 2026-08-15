"""Stem separation via Demucs (Phase 1 #5).

Splits each track into vocals / drums / bass / other (htdemucs) — or vocals /
accompaniment with `stems.two_stems: true` — so you can train stem-conditioned
models or audit the mix. Demucs is imported lazily so the rest of the toolkit
never pays its import cost or requires it installed.

Install with:  uv pip install demucs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from . import console
from .audio.inventory import AUDIO_GLOB
from .config import Config


def _scan(dir_path: Path) -> List[Path]:
    found: List[Path] = []
    for pattern in AUDIO_GLOB:
        found.extend(sorted(dir_path.glob(pattern)))
    return sorted(set(found))


def _load_demucs():
    try:
        import demucs.pretrained  # noqa: F401
        import demucs.api

        return demucs
    except ImportError:
        console.error("demucs is not installed — run `uv pip install demucs` and retry.")
        return None


def separate_stems(root: Path, cfg: Config, which: str = "clean", limit: int = 0) -> List[dict]:
    demucs = _load_demucs()
    if demucs is None:
        return []

    scfg = cfg.stems
    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return []

    files = _scan(target)
    if not files:
        console.warn(f"No audio files under {target}")
        return []

    console.step(f"Separating stems for {len(files)} files (model={scfg.model})")
    out_root = root / "data" / "stems"
    out_root.mkdir(parents=True, exist_ok=True)

    # demucs.api.Separator handles model download + device internally
    device = None if scfg.device == "auto" else scfg.device
    separator = demucs.api.Separator(
        model=scfg.model,
        device=device,
        progress=False,
    )

    results: List[dict] = []
    for i, path in enumerate(files, 1):
        if limit and i > limit:
            break
        stem_dir = out_root / path.stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        try:
            _origin, stems = separator.separate_audio_file(str(path))
            rec = {"track": str(path.relative_to(root)), "stems": {}}
            for name, source in stems.items():
                demucs.api.save_audio(source, str(stem_dir / f"{name}.wav"), samplerate=separator.samplerate)
                rec["stems"][name] = str((stem_dir / f"{name}.wav").relative_to(root))
            if scfg.two_stems:
                non_vocal = [s for n, s in stems.items() if n != "vocals"]
                if non_vocal:
                    acc = non_vocal[0].clone()
                    for s in non_vocal[1:]:
                        acc = acc + s
                    demucs.api.save_audio(acc, str(stem_dir / "accompaniment.wav"), samplerate=separator.samplerate)
                    rec["stems"]["accompaniment"] = str((stem_dir / "accompaniment.wav").relative_to(root))
            results.append(rec)
            console.ok(f"[{i}/{len(files)}] {path.name} -> {len(rec['stems'])} stems")
        except Exception as exc:  # noqa: BLE001
            console.error(f"Failed {path.name}: {exc}")

    out = root / "metadata" / "stems.json"
    out.write_text(json.dumps(results, indent=2))
    console.ok(f"Wrote {len(results)} track(s) -> data/stems/ (manifest: metadata/stems.json)")
    return results
