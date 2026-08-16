"""Lyric-quality utilities for the beat → lyrics engine.

Independent of :mod:`musictrain.lyrics` (operates on plain dicts / duck-typed
results), so the engine can import it without a circular dependency.

- :func:`count_syllables` — vowel-group heuristic for per-line flow
- :func:`syllable_target` — the syllable-per-line budget derived from tempo/flow
- :func:`annotate_section` / :func:`annotate` — rhyme + syllable annotations
- :func:`suggest_from_chords` — map detected key/chords to a mood + topic
- :func:`sheet` — a clean markdown studio sheet
- ``ARRANGEMENTS`` — named section layouts (hook-first, double-verse, …)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# Syllable counting (#1)
# --------------------------------------------------------------------------- #
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")


def count_syllables(text: str) -> int:
    """Approximate English syllable count via vowel groups (good enough for flow).

    Not phonetically exact — but deterministic, fast, and consistent, which is
    what matters for pacing lines against a tempo budget.
    """
    if not text:
        return 0
    cleaned = re.sub(r"[^a-z' ]", " ", text.lower())
    total = 0
    for word in cleaned.split():
        w = word.replace("'", "")
        if not w:
            continue
        # drop a silent trailing 'e' (roughly), but never below one syllable
        w2 = re.sub(r"e$", "", w) if len(w) > 2 else w
        groups = _VOWEL_GROUPS.findall(w2)
        total += max(1, len(groups))
    return total


def syllable_target(bpm: float, cadence: str = "medium", density: int = 3) -> int:
    """Syllables-per-line budget shaped by tempo, cadence and density.

    Slow laid-back beats take longer, denser storytelling lines; fast beats take
    shorter, punchier bars. Returns a target in [6, 16].
    """
    target = 9 + max(1, min(5, int(density)))  # density 1→10 … 5→14
    c = (cadence or "medium").lower()
    if c == "fast":
        target += 2
    elif c == "slow":
        target -= 2
    if bpm >= 140:
        target -= 1
    elif bpm <= 90:
        target += 1
    return max(6, min(16, target))


# --------------------------------------------------------------------------- #
# Rhyme + syllable annotations (#2)
# --------------------------------------------------------------------------- #
def annotate_section(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Annotate each line with its syllable count and rhyme group."""
    rhyme = section.get("rhyme", "free")
    out: List[Dict[str, Any]] = []
    for ln in section.get("lines", []):
        out.append({
            "line": ln,
            "syllables": count_syllables(ln),
            "rhyme": rhyme,
        })
    return out


def annotate(result: Any) -> Dict[str, Any]:
    """Full annotation of a :class:`~musictrain.lyrics.LyricsResult`."""
    sections = []
    for s in getattr(result, "sections", []):
        sections.append({
            "role": s.get("role"),
            "bars": s.get("bars"),
            "flow": s.get("flow"),
            "cadence": s.get("cadence"),
            "artist": s.get("artist"),
            "ad_libs": s.get("ad_libs", []),
            "lines": annotate_section(s),
        })
    return {
        "artist": getattr(result, "artist", ""),
        "bpm": getattr(result, "bpm", 0.0),
        "key": getattr(result, "key", ""),
        "swing": getattr(result, "swing", ""),
        "mood": getattr(result, "mood", ""),
        "topic": getattr(result, "topic", ""),
        "seed": getattr(result, "seed", 0),
        "sections": sections,
    }


# --------------------------------------------------------------------------- #
# Chord / key → mood + topic suggestion (#8)
# --------------------------------------------------------------------------- #
def suggest_from_chords(chords: List[Any], key: str = "") -> Dict[str, Any]:
    """Map a detected key + chord list to a suggested mood and topic."""
    k = (key or "").lower()
    is_minor = "minor" in k
    chord_names = [str(c.get("chord", "")) for c in chords if isinstance(c, dict)]
    minor_count = sum(1 for c in chord_names if c.endswith("m"))
    frac_minor = (minor_count / len(chord_names)) if chord_names else 0.0

    if is_minor or frac_minor >= 0.5:
        mood, topic = "melancholic", "pain"
        reason = "minor key / minor-heavy chords"
    elif frac_minor >= 0.25:
        mood, topic = "emotional", "heartbreak"
        reason = "mixed major/minor movement"
    else:
        mood, topic = "confident", "success"
        reason = "bright major harmony"

    return {
        "mood": mood,
        "topic": topic,
        "key": k,
        "minor_chord_frac": round(frac_minor, 3),
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# Studio sheet export (#9)
# --------------------------------------------------------------------------- #
def sheet(result: Any) -> str:
    """Render a clean markdown studio sheet with bar counts and ad-lib cues."""
    out: List[str] = []
    out.append(f"# {getattr(result, 'artist', '?')} — {getattr(result, 'mood', '')} / {getattr(result, 'topic', '')}")
    out.append(
        f"**{getattr(result, 'bpm', 0):.0f} BPM · {getattr(result, 'key', '?')} · "
        f"swing {getattr(result, 'swing', '?')} · seed {getattr(result, 'seed', 0)}**"
    )
    for s in getattr(result, "sections", []):
        artist = s.get("artist") or getattr(result, "artist", "")
        out.append(f"\n## [{s.get('role', '?').upper()}] {s.get('bars', '?')} bars "
                   f"· {s.get('flow', '')} @ {s.get('cadence', '')} · {artist}")
        for ann in annotate_section(s):
            out.append(f"- {ann['line']}  _({ann['syllables']} syll · rhyme {ann['rhyme']})_")
        if s.get("ad_libs"):
            out.append("> ad-libs: " + ", ".join(f"({a})" for a in s["ad_libs"]))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Arrangement presets (#5) — (role, bars) tuples; lyrics.py builds SectionSpecs
# --------------------------------------------------------------------------- #
ARRANGEMENTS: Dict[str, List[tuple]] = {
    "standard": [("intro", 4), ("verse", 16), ("hook", 8), ("verse", 16), ("hook", 8), ("outro", 4)],
    "hook-first": [("hook", 8), ("verse", 16), ("hook", 8), ("verse", 16), ("hook", 8), ("outro", 4)],
    "double-verse": [("intro", 4), ("verse", 16), ("verse", 16), ("hook", 8), ("verse", 16), ("hook", 8), ("outro", 4)],
    "16-bar-opener": [("verse", 16), ("hook", 8), ("verse", 16), ("bridge", 8), ("hook", 8), ("outro", 4)],
    "short-form": [("intro", 4), ("verse", 8), ("hook", 8), ("verse", 8), ("hook", 8), ("outro", 4)],
    "long-form": [("intro", 4), ("verse", 12), ("hook", 12), ("verse", 12), ("hook", 12), ("bridge", 8), ("outro", 4)],
}


def arrangement_names() -> List[str]:
    return list(ARRANGEMENTS.keys())


# --------------------------------------------------------------------------- #
# Lyrical metrics (#6)
# --------------------------------------------------------------------------- #
def metrics(result: Any) -> Dict[str, Any]:
    """Compute a small set of lyrical-quality metrics for a result.

    - rhyme_density: fraction of adjacent line pairs sharing a rhyme group
    - avg_syllables / syllable_std: pacing and consistency
    - flow_score: a 0-100 composite heuristic
    """
    import statistics

    all_lines: List[str] = []
    rhyme_pairs = 0
    rhyme_total = 0
    bars = 0
    for s in getattr(result, "sections", []):
        all_lines.extend(s.get("lines", []))
        bars += int(s.get("bars", 0))
        rhymes = s.get("line_rhymes") or []
        for i in range(1, len(rhymes)):
            rhyme_total += 1
            if rhymes[i] == rhymes[i - 1]:
                rhyme_pairs += 1

    sylls = [count_syllables(ln) for ln in all_lines]
    avg = statistics.mean(sylls) if sylls else 0.0
    std = statistics.pstdev(sylls) if len(sylls) > 1 else 0.0
    rhyme_density = (rhyme_pairs / rhyme_total) if rhyme_total else 0.0

    score = 50.0 + rhyme_density * 30.0
    if 8 <= avg <= 14:
        score += 10.0
    score -= min(std, 4.0) * 5.0
    score = max(0.0, min(100.0, score))

    return {
        "bars": bars,
        "lines": len(all_lines),
        "avg_syllables": round(avg, 2),
        "syllable_std": round(std, 2),
        "rhyme_density": round(rhyme_density, 3),
        "flow_score": round(score),
    }


# --------------------------------------------------------------------------- #
# LRC (karaoke) export (#9)
# --------------------------------------------------------------------------- #
def lrc(result: Any) -> str:
    """Timestamp each line as one bar, driven by the beat's BPM (4/4)."""
    bpm = float(getattr(result, "bpm", 120.0) or 120.0)
    bar_sec = (60.0 / bpm) * 4.0
    t = 0.0
    out: List[str] = []
    for s in getattr(result, "sections", []):
        for ln in s.get("lines", []):
            mm = int(t // 60)
            ss = t % 60
            out.append(f"[{mm:02d}:{ss:05.2f}] {ln}")
            t += bar_sec
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Side-by-side diff (#7)
# --------------------------------------------------------------------------- #
def diff_results(a: Any, b: Any) -> List[str]:
    """Unified diff between two results' full text."""
    import difflib

    return list(difflib.unified_diff(
        a.full_text().splitlines(),
        b.full_text().splitlines(),
        fromfile=f"{getattr(a, 'artist', 'a')} seed={getattr(a, 'seed', '?')}",
        tofile=f"{getattr(b, 'artist', 'b')} seed={getattr(b, 'seed', '?')}",
        lineterm="",
    ))
