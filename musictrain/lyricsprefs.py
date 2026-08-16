"""Lyric-generation preferences: favorites, random recipes, weights, negatives.

Feature set (rap/lyrics pivot):
- #26 prompt library with favorites (save / reuse / star a recipe)
- #27 "surprise me" randomized recipe assembler
- #28 per-tag prompt weights (emphasis dials)
- #29 negative prompt (banned words / avoided topics)
- #30 generation history + pairwise diff

State lives under ``metadata/`` as plain JSON so it round-trips through git
and is easy to inspect:

- ``metadata/lyric_prefs.json``    — favorites, weights, negatives
- ``metadata/lyric_history.jsonl`` — one generation per line
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .artists import ARTISTS, GENRES, MOODS, get_artist, get_genre

_PREFS = "lyric_prefs.json"
_HISTORY = "lyric_history.jsonl"

# Default emphasis per dimension (feature #28). Higher = stronger pull.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "topic": 1.0,
    "mood": 1.0,
    "flow": 1.0,
    "ad_libs": 0.6,
    "slang": 0.6,
    "rhyme": 0.5,
    "energy": 0.4,
    "density": 0.4,
}

DEFAULT_NEGATIVES: List[str] = []  # e.g. ["profanity", "violence", "drugs"]


# --------------------------------------------------------------------------- #
# Prefs file I/O
# --------------------------------------------------------------------------- #
def _prefs_path(root: Path) -> Path:
    return Path(root) / "metadata" / _PREFS


def _history_path(root: Path) -> Path:
    return Path(root) / "metadata" / _HISTORY


def load_prefs(root: Path) -> Dict[str, Any]:
    p = _prefs_path(root)
    if not p.exists():
        return {"favorites": {}, "weights": dict(DEFAULT_WEIGHTS), "negatives": list(DEFAULT_NEGATIVES)}
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return {"favorites": {}, "weights": dict(DEFAULT_WEIGHTS), "negatives": list(DEFAULT_NEGATIVES)}
    data.setdefault("favorites", {})
    weights = dict(DEFAULT_WEIGHTS)
    weights.update(data.get("weights") or {})
    data["weights"] = weights
    data.setdefault("negatives", list(DEFAULT_NEGATIVES))
    return data


def save_prefs(root: Path, prefs: Dict[str, Any]) -> Path:
    p = _prefs_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prefs, indent=2, sort_keys=True))
    return p


# --------------------------------------------------------------------------- #
# Favorites (feature #26)
# --------------------------------------------------------------------------- #
def favorite_keys(root: Path) -> List[str]:
    return sorted(load_prefs(root)["favorites"].keys())


def add_favorite(root: Path, name: str, recipe: Dict[str, Any]) -> Path:
    prefs = load_prefs(root)
    prefs["favorites"][name] = recipe
    return save_prefs(root, prefs)


def get_favorite(root: Path, name: str) -> Optional[Dict[str, Any]]:
    return load_prefs(root)["favorites"].get(name)


def remove_favorite(root: Path, name: str) -> bool:
    prefs = load_prefs(root)
    if name not in prefs["favorites"]:
        return False
    del prefs["favorites"][name]
    save_prefs(root, prefs)
    return True


# --------------------------------------------------------------------------- #
# Random recipe (feature #27)
# --------------------------------------------------------------------------- #
def random_recipe(root: Path, seed: Optional[int] = None) -> Dict[str, Any]:
    """Assemble a random 'surprise me' recipe from the catalogs."""
    rng = random.Random(seed)
    artist = rng.choice(ARTISTS)
    genre = rng.choice(GENRES)
    mood = rng.choice(list(MOODS))
    topic = rng.choice(artist.topics) if artist.topics else "struggle"
    return {
        "artist": artist.id,
        "genre": genre.name,
        "mood": mood,
        "topic": topic,
        "energy": genre.default_energy,
        "density": genre.default_density,
        "seed": seed if seed is not None else rng.randint(0, 10**6),
    }


# --------------------------------------------------------------------------- #
# Weights (feature #28)
# --------------------------------------------------------------------------- #
def weights(root: Path) -> Dict[str, float]:
    return dict(load_prefs(root)["weights"])


def set_weight(root: Path, key: str, value: float) -> Path:
    prefs = load_prefs(root)
    prefs["weights"][key] = float(value)
    return save_prefs(root, prefs)


def reset_weights(root: Path) -> Path:
    prefs = load_prefs(root)
    prefs["weights"] = dict(DEFAULT_WEIGHTS)
    return save_prefs(root, prefs)


# --------------------------------------------------------------------------- #
# Negatives (feature #29)
# --------------------------------------------------------------------------- #
def negatives(root: Path) -> List[str]:
    return list(load_prefs(root)["negatives"])


def add_negative(root: Path, term: str) -> Path:
    prefs = load_prefs(root)
    t = term.strip()
    if t and t not in prefs["negatives"]:
        prefs["negatives"].append(t)
    return save_prefs(root, prefs)


def remove_negative(root: Path, term: str) -> Path:
    prefs = load_prefs(root)
    prefs["negatives"] = [t for t in prefs["negatives"] if t != term.strip()]
    return save_prefs(root, prefs)


def clear_negatives(root: Path) -> Path:
    prefs = load_prefs(root)
    prefs["negatives"] = []
    return save_prefs(root, prefs)


# --------------------------------------------------------------------------- #
# History + diff (feature #30)
# --------------------------------------------------------------------------- #
def record_history(root: Path, entry: Dict[str, Any]) -> Path:
    p = _history_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return p


def history(root: Path, limit: int = 0) -> List[Dict[str, Any]]:
    p = _history_path(root)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    if limit:
        out = out[-limit:]
    return out


def history_diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Return human-readable lines describing what changed between two runs."""
    lines: List[str] = []
    for field in ("artist", "mood", "topic", "genre", "seed", "bpm", "key"):
        va, vb = a.get(field), b.get(field)
        if va != vb:
            lines.append(f"{field}: {va!r} -> {vb!r}")
    if not lines:
        lines.append("(identical settings)")
    return lines


# --------------------------------------------------------------------------- #
# Helper: assemble a BeatContext-friendly recipe dict from loose CLI inputs.
# --------------------------------------------------------------------------- #
def normalize_recipe(
    artist: str = "",
    mood: str = "",
    topic: str = "",
    genre: str = "",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve names/ids to canonical values, applying genre defaults as fallback."""
    out: Dict[str, Any] = {"artist": "", "mood": "", "topic": "", "genre": ""}
    a = get_artist(artist)
    g = get_genre(genre)
    out["artist"] = a.id if a else (ARTISTS[0].id if ARTISTS else "")
    out["genre"] = g.name if g else ""
    out["mood"] = (mood or "").strip().lower() or (g.moods[0] if g and g.moods else "dark")
    out["topic"] = (topic or "").strip().lower()
    if not out["topic"]:
        if a and a.topics:
            out["topic"] = a.topics[0]
        elif g and g.topics:
            out["topic"] = g.topics[0]
        else:
            out["topic"] = "struggle"
    if seed is not None:
        out["seed"] = seed
    return out
