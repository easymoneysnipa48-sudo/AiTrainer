"""Tests for the third lyrics batch: rhyme schemes, similes, duet mode, hook/verse
contrast, metrics, LRC export, diff, and project save/load."""
from __future__ import annotations

from musictrain import lyrics
from musictrain import lyricproject
from musictrain import lyrictools as LT


# --------------------------------------------------------------------------- #
# Rhyme schemes (#1), similes (#2), hook/verse contrast (#4), duet (#5)
# --------------------------------------------------------------------------- #
def test_aabb_rhyme_scheme_pairs():
    r = lyrics.generate(lyrics.BeatContext(
        artist="drake", mood="dark", topic="pain", seed=1,
        structure=[lyrics.SectionSpec(role="verse", bars=6)],
    ))
    rhymes = r.sections[0]["line_rhymes"]
    assert rhymes[0] == rhymes[1]
    assert rhymes[2] == rhymes[3]
    assert rhymes[4] == rhymes[5]


def test_internal_rhyme_scheme_alternates():
    r = lyrics.generate(lyrics.BeatContext(
        artist="meek-mill", mood="dark", topic="pain", seed=1,
        structure=[lyrics.SectionSpec(role="verse", bars=4)],
    ))
    rhymes = r.sections[0]["line_rhymes"]
    assert rhymes[0] == rhymes[2]
    assert rhymes[1] == rhymes[3]


def test_simile_templates_present():
    assert any("like a" in t for t in lyrics._LINE_TEMPLATES)


def test_hook_shorter_than_verse():
    r = lyrics.generate(lyrics.BeatContext(
        artist="future", mood="dark", topic="pain", seed=1,
        structure=[lyrics.SectionSpec(role="hook", bars=4),
                   lyrics.SectionSpec(role="verse", bars=4)],
    ))
    by_role = {s["role"]: s for s in r.sections}
    assert by_role["hook"]["syllable_target"] < by_role["verse"]["syllable_target"]


def test_duet_alternates_artists():
    r = lyrics.generate(lyrics.BeatContext(
        artist="future", mood="dark", topic="pain", seed=1,
        structure=[lyrics.SectionSpec(role="verse", bars=4, artist="drake", artist2="gunna")],
    ))
    s = r.sections[0]
    assert s["duet"] is True
    assert s["artist"] == "Drake & Gunna"
    assert s["line_artists"] == ["Drake", "Gunna", "Drake", "Gunna"]


# --------------------------------------------------------------------------- #
# Metrics (#6), LRC (#9), diff (#7)
# --------------------------------------------------------------------------- #
def test_metrics_bounds():
    r = lyrics.generate(lyrics.BeatContext(artist="future", seed=1))
    m = LT.metrics(r)
    assert 0 <= m["flow_score"] <= 100
    assert 0.0 <= m["rhyme_density"] <= 1.0
    assert m["bars"] > 0 and m["lines"] > 0


def test_lrc_timestamps():
    r = lyrics.generate(lyrics.BeatContext(artist="future", bpm=120.0, seed=1))
    lines = LT.lrc(r).splitlines()
    assert lines
    for ln in lines:
        assert ln.startswith("[") and "]" in ln[:10]


def test_diff_results_identical_is_empty():
    r = lyrics.generate(lyrics.BeatContext(artist="future", seed=1))
    assert LT.diff_results(r, r) == []


def test_diff_results_notes_changes():
    a = lyrics.generate(lyrics.BeatContext(artist="future", seed=1))
    b = lyrics.generate(lyrics.BeatContext(artist="future", seed=2))
    assert LT.diff_results(a, b)  # seeds differ -> some diff lines


# --------------------------------------------------------------------------- #
# Project save/load (#10)
# --------------------------------------------------------------------------- #
def test_project_roundtrip(tmp_path):
    lyricproject.save_project(tmp_path, "demo", {
        "recipe": {"artist": "future", "mood": "dark", "seed": 3},
        "result": {"sections": []},
    })
    assert "demo" in lyricproject.list_projects(tmp_path)
    loaded = lyricproject.load_project(tmp_path, "demo")
    assert loaded["recipe"]["artist"] == "future"
    assert lyricproject.delete_project(tmp_path, "demo") is True
    assert lyricproject.load_project(tmp_path, "demo") is None


# --------------------------------------------------------------------------- #
# CLI — new flags
# --------------------------------------------------------------------------- #
def test_cli_lyrics_metrics(tmp_path, capsys):
    from musictrain.cli import main
    rc = main(["lyrics", "--root", str(tmp_path), "--artist", "future",
               "--metrics", "--no-save"])
    assert rc == 0
    assert "flow_score" in capsys.readouterr().out


def test_cli_lyrics_lrc(tmp_path):
    from musictrain.cli import main
    out = tmp_path / "x.lrc"
    rc = main(["lyrics", "--root", str(tmp_path), "--artist", "future",
               "--lrc", str(out), "--no-save"])
    assert rc == 0
    assert "[00:00.00]" in out.read_text()


def test_cli_lyrics_duet_feature(tmp_path, capsys):
    from musictrain.cli import main
    rc = main(["lyrics", "--root", str(tmp_path), "--artist", "drake",
               "--feature", "verse=future+gunna", "--seed", "5", "--no-save"])
    assert rc == 0
    assert "Future & Gunna" in capsys.readouterr().out


def test_cli_lyrics_project_roundtrip(tmp_path, capsys):
    from musictrain.cli import main
    assert main(["lyrics", "--root", str(tmp_path), "--artist", "future",
                 "--project-save", "p", "--no-save"]) == 0
    assert main(["lyrics", "--root", str(tmp_path), "--project-load", "p",
                 "--no-save"]) == 0
    assert "loaded project 'p'" in capsys.readouterr().out
