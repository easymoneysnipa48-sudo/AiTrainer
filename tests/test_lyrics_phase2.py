"""Tests for the second lyrics batch: syllable flow, annotations, arrangement
presets, multi-artist features, chord→mood suggestion, studio sheet, autopilot."""
from __future__ import annotations

from musictrain import lyrics
from musictrain import lyrictools as LT
from musictrain import lyricsprefs


# --------------------------------------------------------------------------- #
# lyrictools — syllables, annotations, suggestion, sheet, arrangements
# --------------------------------------------------------------------------- #
def test_count_syllables_basics():
    assert LT.count_syllables("") == 0
    assert LT.count_syllables("pain") == 1
    assert LT.count_syllables("I been movin' through the pain") >= 6
    assert LT.count_syllables("situation") >= 3


def test_syllable_target_tracks_tempo_and_flow():
    fast = LT.syllable_target(150, "fast", 5)
    slow = LT.syllable_target(80, "slow", 2)
    assert fast > slow
    assert 6 <= fast <= 16 and 6 <= slow <= 16


def test_annotate_section_counts_syllables():
    sec = {"role": "verse", "rhyme": "ain", "lines": ["I been through the pain"]}
    ann = LT.annotate_section(sec)
    assert ann[0]["line"] == "I been through the pain"
    assert ann[0]["syllables"] >= 5
    assert ann[0]["rhyme"] == "ain"


def test_suggest_from_chords_minor_vs_major():
    minor = LT.suggest_from_chords([{"chord": "Em"}, {"chord": "Am"}], "E minor")
    assert minor["mood"] == "melancholic" and minor["topic"] == "pain"
    major = LT.suggest_from_chords([{"chord": "C"}, {"chord": "G"}], "C major")
    assert major["mood"] == "confident" and major["topic"] == "success"


def test_arrangement_presets():
    names = LT.arrangement_names()
    assert "hook-first" in names and "double-verse" in names
    specs = lyrics.arrangement_specs("hook-first")
    assert specs and specs[0].role == "hook"
    assert lyrics.arrangement_specs("does-not-exist") == []


def test_sheet_has_headers_and_counts():
    r = lyrics.generate(lyrics.BeatContext(artist="future", mood="dark", topic="pain", seed=1))
    s = r.to_sheet()
    assert "Future" in s
    assert "[INTRO]" in s
    assert "syll" in s
    assert "BPM" in s


# --------------------------------------------------------------------------- #
# lyrics.py — multi-artist feature mode
# --------------------------------------------------------------------------- #
def test_multi_artist_feature_sections():
    ctx = lyrics.BeatContext(
        artist="future", mood="dark", topic="pain", seed=1,
        structure=[
            lyrics.SectionSpec(role="verse", bars=4, artist="drake"),
            lyrics.SectionSpec(role="hook", bars=4),
        ],
    )
    r = lyrics.generate(ctx)
    by_role = {s["role"]: s for s in r.sections}
    assert by_role["verse"]["artist"] == "Drake"
    assert by_role["hook"]["artist"] == "Future"


def test_restyle_preserves_feature_artists():
    base = lyrics.generate(lyrics.BeatContext(
        artist="future", mood="dark", topic="pain", seed=3,
        structure=[lyrics.SectionSpec(role="verse", bars=4, artist="drake")],
    ))
    restyled = lyrics.restyle(base, "chief-keef", seed=3)
    by_role = {s["role"]: s for s in restyled.sections}
    # the verse was pinned to Drake; restyle keeps the pinned feature
    assert by_role["verse"]["artist"] == "Drake"


# --------------------------------------------------------------------------- #
# lyricsprefs — autopilot
# --------------------------------------------------------------------------- #
def test_autopilot_returns_valid_recipe(tmp_path):
    from musictrain import artists
    r = lyricsprefs.autopilot(tmp_path, seed=1)
    assert artists.get_artist(r["artist"]) is not None
    assert r["mood"] and r["topic"]


# --------------------------------------------------------------------------- #
# CLI — new flags
# --------------------------------------------------------------------------- #
def test_cli_lyrics_preset_and_feature(tmp_path, capsys):
    from musictrain.cli import main
    rc = main(["lyrics", "--root", str(tmp_path), "--artist", "future",
               "--preset", "hook-first", "--feature", "verse=drake",
               "--seed", "5", "--no-save"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[hook" in out.lower()
    assert "Drake" in out


def test_cli_lyrics_annotate(tmp_path, capsys):
    from musictrain.cli import main
    rc = main(["lyrics", "--root", str(tmp_path), "--artist", "future",
               "--annotate", "--seed", "5", "--no-save"])
    assert rc == 0
    assert "syll" in capsys.readouterr().out


def test_cli_lyrics_sheet(tmp_path):
    from musictrain.cli import main
    rc = main(["lyrics", "--root", str(tmp_path), "--artist", "future",
               "--sheet", str(tmp_path / "sheet.md"), "--seed", "5", "--no-save"])
    assert rc == 0
    sheet = (tmp_path / "sheet.md").read_text()
    assert "[INTRO]" in sheet and "BPM" in sheet


def test_cli_lyrics_suggest_and_vocals(tmp_path, capsys):
    import json
    from musictrain.cli import main

    rec = {
        "key": {"key": "E minor"}, "beat_grid": {"tempo": 144.0},
        "swing": {"feel": "swung"}, "chords": [{"chord": "Em"}, {"chord": "Am"}],
        "vocal": {"verdict": "instrumental"},
        "structure": {"segments": [{"role": "verse", "energy": 0.5}]},
    }
    a = tmp_path / "analysis.json"
    a.write_text(json.dumps(rec))

    assert main(["lyrics", "--root", str(tmp_path), "--analysis", str(a), "--suggest", "--no-save"]) == 0
    assert "melancholic" in capsys.readouterr().out
    assert main(["lyrics", "--root", str(tmp_path), "--analysis", str(a), "--vocals", "--no-save"]) == 0
    assert "instrumental" in capsys.readouterr().out
