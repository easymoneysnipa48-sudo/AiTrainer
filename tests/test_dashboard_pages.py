"""Render every dashboard page under Streamlit's AppTest engine.

Catches runtime errors (bad imports, missing keys, broken charts) that a plain
unit test of the helper functions would miss.
"""
import pathlib

import pytest

from streamlit.testing.v1 import AppTest

DASHBOARD = pathlib.Path(__file__).resolve().parents[1] / "musictrain" / "dashboard.py"

PAGES = [
    "📋 Inventory", "🔧 Normalize", "🏷️ Metadata", "✂️ Segment & Split",
    "🎛️ Generate", "🪄 Prompt builder", "📏 Check BPM", "🎬 Visualize",
    "🏷️ Labels", "📊 Compare", "🧹 Hygiene", "🏆 Leaderboard",
    "📈 Training", "🔬 Analytics", "🧮 Metrics Lab", "🎯 Eval",
    "🎧 Listening", "✂️ Annotate", "🧪 Campaign",
    "📦 Model Ops", "📡 Ops & Alerts", "🪵 Logs",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.session_state["nav"] = page
    at.run()
    assert not at.exception, f"{page} raised: {at.exception}"
