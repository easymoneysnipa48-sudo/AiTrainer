"""Auto-generate ``metadata/labels.csv`` keyed to the *real* segment files.

The hand-written labels.csv used invented source_ids (``track_001``) that never
match the actual segment filenames, so ``musictrain finetune`` found **zero**
(audio, description) pairs. This module rebuilds labels.csv from:

* ``metadata/manifest.jsonl`` — the measured BPM/key of each source song
* ``data/segments/*.wav`` — the actual training units

Each segment gets a description like
``"verse, 156 BPM, G major, aggressive trap beat, 808 bass, trap hi-hats, high
energy"`` so the MusicGen fine-tune has real text conditioning. The old file is
preserved as ``labels.csv.manual.bak``.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from . import console
from .logging import get_logger

log = get_logger("beatlabels")

_COLUMNS = [
    "source_id", "song_id", "section", "section_type", "section_index",
    "start_time", "end_time", "genre", "mood", "instruments", "energy",
    "narrative_role", "license", "description",
]

_ROLE_CYCLE = ["intro", "verse", "hook", "verse", "hook", "verse", "bridge", "hook", "outro"]
_ENERGY_BY_ROLE = {"intro": 0.3, "verse": 0.6, "hook": 0.9, "bridge": 0.5, "outro": 0.25}
_NARRATIVE = {
    "intro": "sparse opening", "verse": "narrative build",
    "hook": "central hook", "bridge": "transition", "outro": "closing",
}
_ENERGY_WORD = {0.25: "low", 0.3: "low", 0.5: "medium", 0.6: "medium", 0.9: "high"}


def _norm(s: str) -> str:
    """Lowercase alnum runs for fuzzy name matching."""
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _vibe(song_stem: str) -> Dict[str, str]:
    """Genre/mood/instruments heuristic from the beat's filename.

    Only emits terms present in ``labels.VOCAB`` so the generated CSV always
    passes ``musictrain labels --check`` (enforced in CI smoke).
    """
    low = song_stem.lower()
    if "thug" in low or "aggress" in low or "hard" in low:
        return {
            "genre": "trap", "mood": "aggressive|dark",
            "instruments": "808 bass|trap hi-hats|piano",
        }
    if "soul" in low or "talk" in low:
        return {
            "genre": "melodic trap", "mood": "emotional|reflective",
            "instruments": "piano|808 bass|strings",
        }
    if "feel" in low or "kodak" in low:
        return {
            "genre": "melodic trap", "mood": "emotional|reflective",
            "instruments": "808 bass|bells|trap hi-hats",
        }
    if "click" in low:
        return {
            "genre": "trap", "mood": "dark",
            "instruments": "kick|percussion",
        }
    return {
        "genre": "trap", "mood": "dark|atmospheric",
        "instruments": "808 bass|trap hi-hats",
    }


def _load_song_analysis(root: Path) -> List[dict]:
    """(slug, bpm, key) per source song from manifest.jsonl."""
    manifest = root / "metadata" / "manifest.jsonl"
    out: List[dict] = []
    if not manifest.exists():
        return out
    for ln in manifest.open(encoding="utf-8"):
        if not ln.strip():
            continue
        row = json.loads(ln)
        path = row.get("path") or ""
        beat = row.get("beat_grid") or {}
        raw_key = row.get("key")
        key = raw_key.get("key", "") if isinstance(raw_key, dict) else str(raw_key or "")
        out.append({
            "path": path,
            "stem": Path(path).stem,
            "source_id": row.get("source_id") or "",
            "norm": _norm(Path(path).stem),
            "bpm": float(row.get("bpm") or beat.get("tempo") or 0.0),
            "key": key,
        })
    return out


def _match_song(seg_stem: str, songs: List[dict]) -> Optional[dict]:
    """Segment -> source song: exact source_id prefix, else fuzzy name match."""
    for s in songs:
        if s["source_id"] and seg_stem.startswith(s["source_id"]):
            return s
    norm = _norm(seg_stem)
    hits = [s for s in songs if s["norm"] and s["norm"] in norm]
    return hits[0] if hits else None


def generate_labels(root: Path, force: bool = False) -> int:
    """Write labels.csv with one row per real segment; returns row count."""
    root = Path(root)
    labels = root / "metadata" / "labels.csv"
    if labels.exists() and not force:
        console.error(
            f"{labels} exists — pass --force to overwrite "
            "(the current one is backed up to labels.csv.manual.bak)"
        )
        return 0

    seg_dir = root / "data" / "segments"
    segs = sorted(seg_dir.glob("*.wav")) if seg_dir.exists() else []
    if not segs:
        console.error("no segments in data/segments — run the segment step first")
        return 0

    songs = _load_song_analysis(root)
    if not songs:
        console.warn("no manifest entries found — descriptions will lack BPM/key")

    by_song: Dict[str, List[Path]] = {}
    for p in segs:
        song = _match_song(p.stem, songs)
        key = song["stem"] if song else p.stem.split("_seg")[0]
        by_song.setdefault(key, []).append(p)
    for k in by_song:
        by_song[k].sort()

    rows: List[dict] = []
    for song_key, files in sorted(by_song.items()):
        song = next((s for s in songs if s["stem"] == song_key), None)
        vibe = _vibe(song_key)
        n = len(files)
        for i, p in enumerate(files):
            role = _ROLE_CYCLE[i] if i < len(_ROLE_CYCLE) else "verse"
            if n > 1 and i == n - 1:
                role = "outro"
            energy = _ENERGY_BY_ROLE.get(role, 0.5)
            bpm = f"{song['bpm']:.0f}" if song and song["bpm"] else "?"
            key = song["key"] if song and song["key"] else "?"
            desc = (
                f"{role}, {bpm} BPM, {key}, {vibe['mood'].split('|')[0]} "
                f"{vibe['genre']} beat, {vibe['instruments']}, "
                f"{_ENERGY_WORD.get(energy, 'medium')} energy"
            )
            rows.append({
                "source_id": p.stem,
                "song_id": song_key,
                "section": role,
                "section_type": role,
                "section_index": i + 1,
                "start_time": f"{i * 30}.0",
                "end_time": f"{(i + 1) * 30}.0",
                "genre": vibe["genre"],
                "mood": vibe["mood"],
                "instruments": vibe["instruments"],
                "energy": f"{energy:.2f}",
                "narrative_role": _NARRATIVE.get(role, "section"),
                "license": "owned",
                "description": desc,
            })

    if labels.exists():
        labels.replace(labels.with_suffix(".csv.manual.bak"))
    with labels.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    console.ok(f"Wrote {len(rows)} segment labels -> metadata/labels.csv")
    for r in rows[:3]:
        console.info(f"  {r['source_id'][:38]:40s} {r['description']}")
    if len(rows) > 3:
        console.info(f"  … and {len(rows) - 3} more")
    return len(rows)
