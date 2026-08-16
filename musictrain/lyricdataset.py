"""Lyric dataset import + prep for artist-style LLM fine-tuning.

Turns raw lyric files (the 1.4M-row ``updated_rappers.csv`` corpus, or any
JSON/CSV of songs) into a normalized, per-artist dataset:

* ``lyrics/<artist_id>/songs.jsonl`` — one song record per line
  ``{artist, artist_id, title, lines, mood, topic, source}``
* ``lyrics/index.json`` — per-artist summary
* ``metadata/lyrics_train|val|test.jsonl`` — deterministic song-level splits
* ``metadata/lyrics_train|val_instructions.jsonl`` — chat-formatted examples
  ready for ``train-lyrics`` (system = artist style, user = prompt, target =
  the actual lyrics)

Artist names are normalized onto the 22 rapper profiles in ``artists.py``
where possible (e.g. ``Jay Z`` -> ``jay-z``), otherwise a slug id is used.
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import console
from .artists import get_artist

# --------------------------------------------------------------------------- #
# Artist name normalization
# --------------------------------------------------------------------------- #

# corpus spellings -> profile id (the profiles are the canonical set)
_ALIASES: Dict[str, str] = {
    "drake": "drake",
    "future": "future",
    "lil durk": "lil-durk",
    "chief keef": "chief-keef",
    "meek mill": "meek-mill",
    "kendrick lamar": "kendrick-lamar",
    "gunna": "gunna",
    "lil baby": "lil-baby",
    "jay z": "jay-z",
    "jay-z": "jay-z",
    "kanye west": "kanye-west",
    "young thug": "young-thug",
    "juice wrld": "juice-wrld",
    "dababy": "dababy",
    "da baby": "dababy",
    "quavo": "quavo",
    "offset": "offset",
    "takeoff": "takeoff",
    "michael jackson": "michael-jackson",
    "nocap": "nocap",
    "quando rondo": "quando-rondo",
    "jackboy": "jackboy",
    "omb peezy": "omb-peezy",
    "lil gotit": "lil-gotit",
}


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "unknown"


def normalize_artist(name: str) -> Tuple[str, str]:
    """Return (artist_id, display_name) for a raw artist string."""
    key = (name or "").strip().lower()
    if key in _ALIASES:
        aid = _ALIASES[key]
        prof = get_artist(aid)
        return aid, (prof.name if prof else name.strip())
    prof = get_artist(key)
    if prof is not None:
        return prof.id, prof.name
    return _slug(key), name.strip() or key


# --------------------------------------------------------------------------- #
# Line cleaning
# --------------------------------------------------------------------------- #

_NOISE = {
    "produced by", "producer", "instrumental", "beat", "beat instrumental",
    "free beat", "type beat", "no copyright", "copyright free", "lyrics",
    "song lyrics", "genius", "genius.com", "embed", "share", "more on genius",
}
# substring matches ("produced by metro", "free type beat")
_NOISE_SUB = ("produced by", "type beat", "free beat", "no copyright",
              "copyright", "genius.com", "instrumental")
_TAG_RE = re.compile(r"^[\[(][^)\]]*[)\]]\s*$")          # "[Intro]" / "(chorus)"
_URL_RE = re.compile(r"https?://|www\.")


def clean_line(raw: str) -> str:
    line = (raw or "").strip()
    if not line or len(line) < 2:
        return ""
    low = line.lower()
    if _URL_RE.search(low):
        return ""
    if _TAG_RE.match(line):
        return ""
    if low in _NOISE or any(n in low for n in _NOISE_SUB):
        return ""
    # strip trailing punctuation-only noise
    return line


def _song_key(artist: str, song: str) -> str:
    return f"{_slug(artist)}::{_slug(song)}"


# --------------------------------------------------------------------------- #
# Importers
# --------------------------------------------------------------------------- #

def import_rap_csv(
    path: Path,
    artists: Optional[Iterable[str]] = None,
    limit: int = 0,
) -> List[dict]:
    """Import the ``artist, song, lyric, next lyric`` row-per-line corpus.

    Consecutive rows with the same (artist, song) are grouped into one song
    record with a ``lines`` list. ``artists`` filters to a subset of artist
    names/ids (case-insensitive substring).
    """
    want = [a.strip().lower() for a in (artists or []) if a and a.strip()]
    songs: Dict[str, dict] = {}
    order: List[str] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            artist = (row.get("artist") or "").strip()
            song = (row.get("song") or "").strip()
            lyric = clean_line(row.get("lyric") or "")
            if not artist or not song or not lyric:
                continue
            if want and not any(w in artist.lower() for w in want):
                continue
            key = _song_key(artist, song)
            rec = songs.get(key)
            if rec is None:
                rec = {
                    "artist": artist, "title": song, "lines": [],
                    "source": str(path),
                }
                songs[key] = rec
                order.append(key)
            rec["lines"].append(lyric)
            if limit and len(order) >= limit:
                break
    out = [songs[k] for k in order]
    return [r for r in out if len(r["lines"]) >= 4]


def import_records(path: Path, artists: Optional[Iterable[str]] = None) -> List[dict]:
    """Generic importer for user-provided JSON/CSV lyric files.

    JSON: a list of song objects with keys like ``artist`` (or ``artist_name``),
    ``title`` (or ``song``/``name``), and lyrics in ``lyrics``/``lines``/``text``
    (a string or list), plus optional ``mood``/``topic``.

    CSV: columns ``artist, title, lyrics`` where ``lyrics`` is the full text
    (newlines separate lines), plus optional ``mood``/``topic`` columns.
    """
    want = [a.strip().lower() for a in (artists or []) if a and a.strip()]
    suffix = path.suffix.lower()
    out: List[dict] = []
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("songs") or data.get("data") or []
        for item in data:
            if not isinstance(item, dict):
                continue
            artist = item.get("artist") or item.get("artist_name") or item.get("rapper") or ""
            title = item.get("title") or item.get("song") or item.get("name") or ""
            text = item.get("lyrics") or item.get("lines") or item.get("text") or ""
            if not artist or not text:
                continue
            if want and not any(w in artist.lower() for w in want):
                continue
            lines = text if isinstance(text, list) else str(text).splitlines()
            lines = [ln for ln in (clean_line(raw) for raw in lines) if ln]
            if len(lines) < 4:
                continue
            out.append({
                "artist": artist, "title": title or "Untitled",
                "lines": lines, "mood": item.get("mood", ""),
                "topic": item.get("topic", ""), "source": str(path),
            })
    elif suffix == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                artist = (row.get("artist") or "").strip()
                title = (row.get("title") or row.get("song") or "").strip()
                text = row.get("lyrics") or row.get("text") or ""
                if not artist or not text:
                    continue
                if want and not any(w in artist.lower() for w in want):
                    continue
                lines = [ln for ln in (clean_line(raw) for raw in str(text).splitlines()) if ln]
                if len(lines) < 4:
                    continue
                out.append({
                    "artist": artist, "title": title or "Untitled",
                    "lines": lines, "mood": row.get("mood", ""),
                    "topic": row.get("topic", ""), "source": str(path),
                })
    return out


def _is_rap_csv(path: Path) -> bool:
    """Row-per-line corpus (artist, song, lyric, next lyric) vs generic songs."""
    if "rappers" in path.name.lower() or path.name.lower().startswith("updated_rappers"):
        return True
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            header = f.readline().lower()
        return "next lyric" in header or ("lyric" in header and "song" in header)
    except OSError:
        return False


def import_sources(
    sources: List[Path],
    artists: Optional[Iterable[str]] = None,
    limit: int = 0,
) -> List[dict]:
    """Import many files; row-per-line rap CSVs get grouping, everything else generic."""
    out: List[dict] = []
    for p in sources:
        if not p.exists():
            console.warn(f"skip (missing) {p}")
            continue
        if _is_rap_csv(p):
            out.extend(import_rap_csv(p, artists=artists, limit=limit))
        else:
            out.extend(import_records(p, artists=artists))
    return out


# --------------------------------------------------------------------------- #
# Dataset build
# --------------------------------------------------------------------------- #

def build_dataset(
    root: Path,
    sources: List[Path],
    artists: Optional[Iterable[str]] = None,
    limit: int = 0,
) -> Dict[str, object]:
    """Normalize imported songs into ``lyrics/<artist_id>/songs.jsonl``."""
    songs = import_sources(sources, artists=artists, limit=limit)
    by_id: Dict[str, List[dict]] = {}
    for s in songs:
        aid, disp = normalize_artist(s["artist"])
        rec = dict(s)
        rec["artist_id"] = aid
        rec["artist"] = disp
        rec["n_lines"] = len(s["lines"])
        by_id.setdefault(aid, []).append(rec)

    out_root = root / "lyrics"
    out_root.mkdir(parents=True, exist_ok=True)
    index: Dict[str, object] = {}
    total_songs = total_lines = 0
    for aid, recs in sorted(by_id.items()):
        (out_root / aid).mkdir(parents=True, exist_ok=True)
        with (out_root / aid / "songs.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_lines = sum(r["n_lines"] for r in recs)
        index[aid] = {"display": recs[0]["artist"], "songs": len(recs), "lines": n_lines}
        total_songs += len(recs)
        total_lines += n_lines
    with (out_root / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    console.ok(f"Dataset: {total_songs} songs, {total_lines} lines, {len(by_id)} artists -> lyrics/")
    return {"songs": total_songs, "lines": total_lines, "artists": len(by_id), "index": index}


def _all_songs(root: Path) -> List[dict]:
    out: List[dict] = []
    for p in sorted((root / "lyrics").glob("*/songs.jsonl")):
        for ln in p.open(encoding="utf-8"):
            if ln.strip():
                out.append(json.loads(ln))
    return out


def split_dataset(root: Path, val: float = 0.1, test: float = 0.05, seed: int = 42) -> Dict[str, int]:
    """Deterministic song-level train/val/test split, written to metadata/."""
    rng = random.Random(seed)
    songs = _all_songs(root)
    by_artist: Dict[str, List[dict]] = {}
    for s in songs:
        by_artist.setdefault(s["artist_id"], []).append(s)
    splits = {"train": [], "val": [], "test": []}
    for aid, recs in by_artist.items():
        rng.shuffle(recs)
        n = len(recs)
        n_test = max(1, int(round(n * test))) if n >= 10 else 0
        n_val = max(1, int(round(n * val))) if n >= 10 else 0
        # tiny artists (<10 songs) get n_test = n_val = 0, so the train extend
        # below already keeps everything — no separate fallback needed
        splits["test"].extend(recs[:n_test])
        splits["val"].extend(recs[n_test:n_test + n_val])
        splits["train"].extend(recs[n_test + n_val:])
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for name, recs in splits.items():
        with (meta / f"lyrics_{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = len(recs)
    console.ok(f"Split: train={counts['train']} val={counts['val']} test={counts['test']}")
    return counts


# --------------------------------------------------------------------------- #
# Instruction format for fine-tuning
# --------------------------------------------------------------------------- #

_SYSTEM_TMPL = (
    "You are a ghostwriter in the style of {artist}. Match their flow, "
    "rhyme scheme, cadence, ad-libs, and slang. Write original lyrics only."
)
_USER_TMPL = "Write {style} lyrics in the style of {artist}."
_CHUNK = 8  # lines per chunk example (keeps each example inside the context window)


def _chunk_lines(lines: List[str]) -> List[List[str]]:
    return [lines[i:i + _CHUNK] for i in range(0, len(lines), _CHUNK)]


def build_instruction(rec: dict, style: str = "a verse") -> dict:
    """One chat example: system = artist style, target = the real lyrics."""
    artist = rec.get("artist") or rec.get("artist_id") or "unknown"
    target = "\n".join(rec["lines"])
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_TMPL.format(artist=artist)},
            {"role": "user", "content": _USER_TMPL.format(style=style, artist=artist)},
            {"role": "assistant", "content": target},
        ],
    }


def write_training_files(root: Path) -> Dict[str, int]:
    """Write chat-formatted train/val instruction JSONL from the split files."""
    meta = root / "metadata"
    counts: Dict[str, int] = {}
    for name in ("train", "val"):
        src = meta / f"lyrics_{name}.jsonl"
        dst = meta / f"lyrics_{name}_instructions.jsonl"
        if not src.exists():
            continue
        n = 0
        with dst.open("w", encoding="utf-8") as out:
            for ln in src.open(encoding="utf-8"):
                if not ln.strip():
                    continue
                rec = json.loads(ln)
                style = rec.get("mood") or rec.get("topic") or ""
                prompt_style = f"a {style}" if style else "a verse"
                ex = build_instruction(rec, style=prompt_style)
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                for chunk in _chunk_lines(rec["lines"]):
                    out.write(json.dumps({
                        "messages": [
                            {"role": "system", "content": _SYSTEM_TMPL.format(artist=rec["artist"])},
                            {"role": "user", "content": _USER_TMPL.format(style="a verse", artist=rec["artist"])},
                            {"role": "assistant", "content": "\n".join(chunk)},
                        ],
                    }, ensure_ascii=False) + "\n")
                    n += 1
        counts[name] = n
        console.ok(f"instructions {name}: {n} examples -> {dst.name}")
    return counts
