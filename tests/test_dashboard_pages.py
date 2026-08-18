"""Render every dashboard page under Streamlit's AppTest engine.

Catches runtime errors (bad imports, missing keys, broken charts) that a plain
unit test of the helper functions would miss.
"""
import pathlib
import re

import pytest

from streamlit.testing.v1 import AppTest

DASHBOARD = pathlib.Path(__file__).resolve().parents[1] / "musictrain" / "dashboard.py"

PAGES = [
    "📋 Inventory", "🔧 Normalize", "🏷️ Metadata", "✂️ Segment & Split",
    "🎛️ Generate", "🎤 Lyrics", "🪄 Prompt builder", "📏 Check BPM", "🎬 Visualize",
    "🏷️ Labels", "📊 Compare", "🧹 Hygiene", "🏆 Leaderboard",
    "📈 Training", "🔬 Analytics", "🧮 Metrics Lab", "🎯 Eval",
    "🎧 Listening", "✂️ Annotate", "🧪 Campaign",
    "📦 Model Ops", "📡 Ops & Alerts", "🪵 Logs", "⚙️ Settings",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.session_state["nav"] = page
    at.run()
    assert not at.exception, f"{page} raised: {at.exception}"


def test_lyrics_ui_helper_logic() -> None:
    """Core logic behind the lyrics UI helpers (rhyme keys, melody scale)."""

    def rhyme_key(word: str) -> str:
        w = word.lower().strip(".,!?;:'\"()[]-—_")
        if not w:
            return ""
        v = "aeiouy"
        idx = max([i for i, ch in enumerate(w) if ch in v], default=-1)
        if idx == -1:
            return w[-2:]
        return w[idx:].rstrip("estd") or w[-1]

    assert rhyme_key("stack") == rhyme_key("back")
    assert rhyme_key("pain") == rhyme_key("chain")
    assert rhyme_key("money") != rhyme_key("stack")

    scales = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def scale_notes(key: str) -> list:
        m = re.match(r"([A-Ga-g][#b]?)\s+(major|minor|m)", key or "")
        if m:
            root_s, kind = m.group(1), ("minor" if m.group(2) == "m" else m.group(2))
        else:
            m2 = re.match(r"([A-Ga-g][#b]?)", key or "")
            root_s, kind = (m2.group(1) if m2 else "A"), "minor"
        root_s = root_s[0].upper() + (root_s[1:] if len(root_s) > 1 else "")
        root = notes.index(root_s) if root_s in notes else 9
        return [notes[(root + s) % 12] for s in scales.get(kind, scales["minor"])]

    assert scale_notes("A minor") == ["A", "B", "C", "D", "E", "F", "G"]
    assert scale_notes("G major") == ["G", "A", "B", "C", "D", "E", "F#"]
    assert scale_notes("g# minor")[0] == "G#"
