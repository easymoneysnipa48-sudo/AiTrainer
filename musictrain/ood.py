"""Off-distribution (OOD) curation (Phase 1 #8).

Flags tracks that fall outside the target distribution — tempo outside
`ood.bpm_range`, or a genre/instrument in `ood.tag_exclude` — so you can keep an
explicit OOD/hard-negative set for evaluation and negative prompting.
Writes metadata/ood_tracks.json; `action: move` relocates them to data/ood/.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

import jsonlines

from . import console
from .config import Config


def curate_ood(root: Path, cfg: Config, which: str = "clean") -> List[dict]:
    manifest = root / "metadata" / "manifest.jsonl"
    if not manifest.exists():
        console.error("No manifest.jsonl — run `musictrain features` first.")
        return []

    lo, hi = cfg.ood.bpm_range
    exclude = set(cfg.ood.tag_exclude)
    flagged: List[dict] = []
    all_rows = list(jsonlines.open(manifest))

    for r in all_rows:
        reasons: List[str] = []
        bpm = r.get("bpm")
        if bpm is not None and not (lo <= bpm <= hi):
            reasons.append(f"bpm {bpm} outside [{lo}, {hi}]")
        for dim in ("genre", "mood", "instruments"):
            tags = r.get(dim) or []
            if not isinstance(tags, list):
                tags = [tags]
            hits = [t for t in tags if str(t) in exclude]
            if hits:
                reasons.append(f"{dim} {hits} in tag_exclude")
        if reasons:
            rec = dict(r)
            rec["ood_reasons"] = reasons
            flagged.append(rec)

    out = root / "metadata" / "ood_tracks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(flagged, indent=2))

    console.ok(f"{len(flagged)} OOD track(s) -> metadata/ood_tracks.json")
    for f in flagged:
        console.warn(f"{f.get('path', '?')}: " + "; ".join(f["ood_reasons"]))

    if cfg.ood.action == "move" and flagged:
        _move(root, flagged)
    return flagged


def _move(root: Path, flagged: List[dict]) -> None:
    ood_dir = root / "data" / "ood"
    ood_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for r in flagged:
        src = root / r.get("path", "")
        if not src.exists():
            continue
        dst = ood_dir / src.name
        if dst.exists():
            dst = ood_dir / f"{src.stem}_ood{src.suffix}"
        shutil.move(str(src), str(dst))
        moved += 1
    console.ok(f"Moved {moved} OOD track(s) -> data/ood/")
