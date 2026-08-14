"""Prompt-builder: assemble consistent MusicGen prompts (Phase 4 #30).

Turns controlled-vocabulary selections (section, genre, mood, instruments,
BPM, key, energy, narrative role) into the same natural-language shape the
eval set and the labels CSV use, so prompts stay consistent with training
text. Pure function — safe to unit-test, and shared by the CLI and the
dashboard's prompt-builder page.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union


def _as_list(value: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # split on any of the labels-CSV separators: | ; ,
        return [p.strip() for p in value.replace(";", ",").replace("|", ",").split(",") if p.strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v).strip() for v in value if str(v).strip()]
    # bare scalar (int/float/single tag) — treat as one item
    s = str(value).strip()
    return [s] if s else []


def _energy_phrase(energy: Optional[float]) -> Optional[str]:
    if energy is None:
        return None
    if energy < 0.33:
        return "low energy"
    if energy < 0.66:
        return "medium energy"
    return "high energy"


def build_prompt(
    section: Optional[Union[str, Sequence[str]]] = None,
    genre: Optional[str] = None,
    mood: Optional[Union[str, Sequence[str]]] = None,
    instruments: Optional[Union[str, Sequence[str]]] = None,
    bpm: Optional[float] = None,
    key: Optional[str] = None,
    energy: Optional[float] = None,
    role: Optional[str] = None,
) -> str:
    """Assemble a prompt in the canonical order: section, BPM, key, genre,
    mood, instruments, energy, narrative role. Duplicate tokens are dropped."""
    parts: List[str] = []

    sec = _as_list(section)
    if sec:
        parts.append(", ".join(sec))

    if bpm is not None:
        parts.append(f"{bpm:g} BPM")

    if key and str(key).strip():
        parts.append(str(key).strip())

    if genre and str(genre).strip():
        parts.append(str(genre).strip())

    moods = _as_list(mood)
    if moods:
        parts.append(", ".join(dict.fromkeys(moods)))  # dedupe, keep order

    instrs = _as_list(instruments)
    if instrs:
        parts.append(", ".join(dict.fromkeys(instrs)))

    energy_phrase = _energy_phrase(energy)
    if energy_phrase:
        parts.append(energy_phrase)

    if role and str(role).strip():
        parts.append(str(role).strip())

    return ", ".join(p for p in parts if p)


def apply_override(prompt: str, key: Optional[float] = None, bpm: Optional[float] = None) -> str:
    """Substitute BPM/key tokens in an existing prompt (used by eval re-runs)."""
    out = prompt
    if bpm is not None:
        import re

        out = re.sub(r"\b\d+(\.\d+)?\s*BPM\b", f"{bpm:g} BPM", out, flags=re.IGNORECASE)
    if key:
        import re

        out = re.sub(
            r"\b([A-G])([#b]?)\s+(major|minor)\b",
            lambda m: f"{key}",
            out,
            flags=re.IGNORECASE,
        )
    return out
