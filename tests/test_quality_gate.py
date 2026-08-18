"""Tests for the lyric quality gate: CJK leak detection, short/empty lines,
duplicate detection, and gate attachment on generated results."""
from __future__ import annotations

from musictrain.lyrics import BeatContext, LyricsResult, generate, quality_issues


def _result(sections):
    return LyricsResult(
        artist="Future", bpm=140, key="A minor", swing="straight",
        mood="dark", topic="struggle", seed=1, sections=sections,
    )


def test_clean_result_passes():
    r = _result([
        {"role": "verse", "lines": ["I been movin' through the rain",
                                    "they was doubtin' me",
                                    "now they all remember the name",
                                    "whole city on my back"]},
        {"role": "hook", "lines": ["We made it out the dark", "now we shinin'"]},
    ])
    assert quality_issues(r) == []


def test_cjk_leak_flagged():
    r = _result([
        {"role": "verse", "lines": ["I been movin' through the rain",
                                    "可以理解为这是翻译说明",
                                    "now they all remember the name",
                                    "whole city on my back"]},
        {"role": "hook", "lines": ["We made it out", "now we shine"]},
    ])
    issues = quality_issues(r)
    assert any("multilingual/translation leak" in i for i in issues)


def test_too_short_line_flagged():
    r = _result([
        {"role": "verse", "lines": ["I been movin' through the rain", "ok",
                                    "now they all remember the name",
                                    "whole city on my back"]},
        {"role": "hook", "lines": ["We made it out", "now we shine"]},
    ])
    assert any("too-short line" in i for i in quality_issues(r))


def test_duplicate_lines_flagged():
    dup = "I been movin' through the rain"
    r = _result([
        {"role": "verse", "lines": [dup, dup, dup, dup]},
        {"role": "hook", "lines": ["We made it out", "now we shine"]},
    ])
    issues = quality_issues(r)
    assert any("exact duplicates" in i for i in issues)


def test_single_section_flagged():
    r = _result([
        {"role": "verse", "lines": ["a", "b", "c", "d", "e"]},
    ])
    assert any("only 1 section" in i for i in quality_issues(r))


def test_generate_attaches_gate_report():
    ctx = BeatContext(bpm=140, key="A minor", artist="future",
                      mood="dark", topic="struggle", seed=3)
    r = generate(ctx)
    assert isinstance(r.gate_issues, list)
    assert r.gate_issues == []  # offline engine output is gate-clean by construction


def test_gate_in_as_dict():
    ctx = BeatContext(bpm=120, key="C major", artist="drake", seed=5)
    d = generate(ctx).as_dict()
    assert "gate_issues" in d
