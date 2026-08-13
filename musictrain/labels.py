"""Controlled-vocabulary label curation: template scaffolding + validation.

Keeps the natural-language prompts consistent so MusicGen learns a stable
text<->audio mapping. Edit VOCAB below to match your target style — it is the
single source of truth that `musictrain labels --check` enforces.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Set

from . import console

# --------------------------------------------------------------------------- #
# Controlled vocabulary — edit to suit your musical style.
VOCAB: Dict[str, Set[str]] = {
    "genre": {
        "melodic trap", "pain music", "emo rap", "trap", "drill",
        "southern trap", "hip hop", "cinematic hip hop", "lo-fi", "ambient",
        "electronic", "orchestral", "film score", "pop", "rnb", "jazz",
        "synthwave",
    },
    "mood": {
        "dark", "emotional", "determined", "energetic", "calm",
        "melancholic", "uplifting", "aggressive", "reflective", "tense",
        "epic", "mysterious", "nostalgic", "hopeful", "somber", "dreamy",
        "atmospheric",
    },
    "instruments": {
        "piano", "keys", "guitar loop", "acoustic guitar", "electric guitar",
        "808 bass", "sub bass", "bass", "808 slides", "strings", "pads",
        "synths", "synth lead", "drums", "percussion", "trap hi-hats",
        "snare", "clap", "snare roll", "brass", "choir", "vocals",
        "autotune vocals", "ad-libs", "organ", "flute", "harp", "bells",
        "vinyl crackle", "riser",
    },
    "section": {
        "intro", "verse", "pre-chorus", "chorus", "hook", "bridge", "outro",
        "instrumental-transition", "full-song",
    },
    "section_type": {
        "intro", "verse", "pre-chorus", "chorus", "hook", "bridge", "outro",
        "instrumental-transition", "full-song",
    },
}

COLUMNS = [
    "source_id", "song_id", "section", "section_type", "section_index",
    "start_time", "end_time", "genre", "mood", "instruments", "energy",
    "narrative_role", "license", "description",
]

# Example rows demonstrate the expected format. Replace with real tracks.
EXAMPLES = [
    {
        "source_id": "track_001", "song_id": "song_001",
        "section": "intro", "section_type": "intro",
        "section_index": "1", "start_time": "0.0", "end_time": "8.0",
        "genre": "melodic trap", "mood": "dark|atmospheric",
        "instruments": "piano|pads", "energy": "0.3",
        "narrative_role": "sparse opening", "license": "owned",
        "description": "intro, 75 BPM, A minor, dark piano loop, atmospheric pads, low energy",
    },
    {
        "source_id": "track_002", "song_id": "song_001",
        "section": "chorus", "section_type": "hook",
        "section_index": "3", "start_time": "48.0", "end_time": "78.0",
        "genre": "melodic trap", "mood": "dark|emotional|aggressive",
        "instruments": "piano|808 bass|autotune vocals|trap hi-hats", "energy": "0.9",
        "narrative_role": "central emotional hook", "license": "owned",
        "description": "hook, 78 BPM, A minor, heavy 808 bass, autotune vocals, trap hi-hats, high energy",
    },
    {
        "source_id": "track_003", "song_id": "song_002",
        "section": "verse", "section_type": "verse",
        "section_index": "2", "start_time": "8.0", "end_time": "48.0",
        "genre": "pain music", "mood": "melancholic|reflective",
        "instruments": "guitar loop|808 bass|snare", "energy": "0.55",
        "narrative_role": "narrative build", "license": "owned",
        "description": "verse, 72 BPM, E minor, melancholic guitar loop, deep 808, restrained snare, medium energy",
    },
]

ENFORCED_FIELDS = ("genre", "mood", "instruments", "section", "section_type")


def _split(value: str) -> List[str]:
    if not value or not value.strip():
        return []
    sep = "|" if "|" in value else ";" if ";" in value else ","
    return [p.strip() for p in value.split(sep) if p.strip()]


def scaffold(root: Path) -> Path:
    out = Path(root) / "metadata" / "labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        console.warn(f"{out.relative_to(root)} already exists; not overwriting")
        return out

    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in EXAMPLES:
            writer.writerow(row)
    console.ok(f"Wrote labels template -> {out.relative_to(root)}")
    console.info("Replace the example rows with your own tracks, then run:")
    console.info("  musictrain labels --check")
    return out


def check(path: Path) -> List[str]:
    path = Path(path)
    if not path.exists():
        return [f"labels file not found: {path}"]

    rows = list(csv.DictReader(path.open(newline="")))
    issues: List[str] = []
    seen: Set[str] = set()

    for i, row in enumerate(rows, start=2):  # header is line 1
        sid = (row.get("source_id") or "").strip()
        if not sid:
            issues.append(f"row {i}: missing source_id")
        elif sid in seen:
            issues.append(f"row {i}: duplicate source_id {sid!r}")
        else:
            seen.add(sid)

        if not (row.get("description") or "").strip():
            issues.append(f"row {i} ({sid}): missing description (critical for conditioning)")
        if not (row.get("license") or "").strip():
            issues.append(f"row {i} ({sid}): missing license")

        for field in ("genre", "section", "section_type"):
            val = (row.get(field) or "").strip()
            if val and val not in VOCAB.get(field, set()):
                issues.append(f"row {i} ({sid}): unknown {field} {val!r}")

        for field in ("mood", "instruments"):
            for item in _split(row.get(field) or ""):
                if item not in VOCAB.get(field, set()):
                    issues.append(f"row {i} ({sid}): unknown {field} item {item!r}")

        energy = row.get("energy")
        if energy not in (None, ""):
            try:
                e = float(energy)
                if not 0.0 <= e <= 1.0:
                    issues.append(f"row {i} ({sid}): energy {e} outside [0, 1]")
            except (TypeError, ValueError):
                issues.append(f"row {i} ({sid}): energy {energy!r} is not a number")

    return issues
