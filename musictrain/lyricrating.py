"""Lyric rating queue (#49) and personal style profile (#50).

A blind A/B / MOS rating loop over generated lyrics builds a *taste signal*:
the more you rate, the better the engine can bias future generations toward the
artists, moods, and topics you actually prefer.

State lives under ``metadata/``:

- ``metadata/lyric_ratings.jsonl`` — one rating per line
- ``metadata/lyric_profile.json``  — cached aggregated taste profile
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_RATINGS = "lyric_ratings.jsonl"
_PROFILE = "lyric_profile.json"


def _ratings_path(root: Path) -> Path:
    return Path(root) / "metadata" / _RATINGS


def _profile_path(root: Path) -> Path:
    return Path(root) / "metadata" / _PROFILE


# --------------------------------------------------------------------------- #
# Rating persistence
# --------------------------------------------------------------------------- #
def record_rating(root: Path, entry: Dict[str, Any]) -> Path:
    p = _ratings_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return p


def ratings(root: Path) -> List[Dict[str, Any]]:
    p = _ratings_path(root)
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
    return out


def build_queue(root: Path, items: List[Dict[str, Any]], n: int = 10, seed: int = 0) -> List[Dict[str, Any]]:
    """Return a blind A/B queue: shuffled pairs with ids A/B, choice hidden."""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    queue: List[Dict[str, Any]] = []
    for i in range(0, min(n, len(items) - 1), 2):
        a, b = items[i], items[i + 1]
        if rng.random() < 0.5:
            a, b = b, a
        queue.append({"A": a, "B": b})
    return queue


# --------------------------------------------------------------------------- #
# Style profile (feature #50)
# --------------------------------------------------------------------------- #
def _score_map(votes: Dict[str, float]) -> Dict[str, float]:
    total = sum(votes.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 4) for k, v in sorted(votes.items(), key=lambda kv: -kv[1])}


def build_profile(root: Path) -> Dict[str, Any]:
    """Aggregate ratings into a taste profile.

    Each rating contributes ``score`` (default 0.5 for a win, higher for MOS)
    to every dimension it names. The result is a set of normalized preference
    maps used by :func:`bias_recipe`.
    """
    rows = ratings(root)
    artist_votes: Counter = Counter()
    mood_votes: Counter = Counter()
    topic_votes: Counter = Counter()
    genre_votes: Counter = Counter()
    n = 0
    for r in rows:
        s = float(r.get("score", 0.5))
        if s <= 0:
            continue
        n += 1
        if r.get("artist"):
            artist_votes[str(r["artist"])] += s
        if r.get("mood"):
            mood_votes[str(r["mood"])] += s
        if r.get("topic"):
            topic_votes[str(r["topic"])] += s
        if r.get("genre"):
            genre_votes[str(r["genre"])] += s

    profile: Dict[str, Any] = {
        "n_ratings": n,
        "artists": _score_map(dict(artist_votes)),
        "moods": _score_map(dict(mood_votes)),
        "topics": _score_map(dict(topic_votes)),
        "genres": _score_map(dict(genre_votes)),
    }
    p = _profile_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, indent=2, sort_keys=True))
    return profile


def load_profile(root: Path) -> Dict[str, Any]:
    p = _profile_path(root)
    if not p.exists():
        return build_profile(root)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return build_profile(root)


def top_preference(profile: Dict[str, Any], dim: str) -> Optional[str]:
    """Return the highest-scoring value for a dimension, or None if no signal."""
    d = profile.get(dim) or {}
    return next(iter(d), None)


def bias_recipe(root: Path, base: Dict[str, Any], strength: float = 0.5) -> Dict[str, Any]:
    """Nudge a recipe toward the user's taste profile (feature #50)."""
    profile = load_profile(root)
    out = dict(base)
    if not profile.get("n_ratings"):
        return out
    strength = max(0.0, min(1.0, strength))
    for base_key, dim_key in (("artist", "artists"), ("mood", "moods"),
                              ("topic", "topics"), ("genre", "genres")):
        pref = top_preference(profile, dim_key)
        if pref and strength >= 0.5:
            # deterministic override once the signal is strong enough
            out[base_key] = pref
    out["_style_bias"] = strength
    return out
