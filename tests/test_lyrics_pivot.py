"""Tests for the rap/lyrics pivot: artist profiles, the lyric engine, preferences,
rating/style-profile, and the new CLI subcommands."""
from __future__ import annotations

import json

import pytest

from musictrain import artists
from musictrain import lyrics
from musictrain import lyricrating
from musictrain import lyricsprefs
from musictrain.labels import VOCAB


# --------------------------------------------------------------------------- #
# artists.py
# --------------------------------------------------------------------------- #
def test_all_22_artists_present():
    assert len(artists.ARTISTS) == 22
    ids = {a.id for a in artists.ARTISTS}
    for want in ("drake", "omb-peezy", "lil-durk", "chief-keef", "meek-mill",
                 "kendrick-lamar", "jackboy", "quavo", "gunna", "offset",
                 "takeoff", "juice-wrld", "future", "lil-baby", "jay-z",
                 "kanye-west", "michael-jackson", "nocap", "quando-rondo",
                 "dababy", "young-thug", "lil-gotit"):
        assert want in ids, f"missing artist {want}"


def test_artist_lookup_by_id_name_alias():
    assert artists.get_artist("lil-durk").name == "Lil Durk"
    assert artists.get_artist("future").name == "Future"
    assert artists.get_artist("Pluto").id == "future"
    assert artists.get_artist(" 6 God ").id == "drake"
    assert artists.get_artist("nope") is None


def test_genre_templates():
    assert len(artists.GENRES) >= 6
    g = artists.get_genre("melodic trap")
    assert g is not None and g.autotune is True
    assert artists.get_genre("does not exist") is None


def test_moods_expanded_in_labels_vocab():
    # feature #25 — the rap-specific moods must live in the controlled vocab too
    for mood in ("braggadocious", "menacing", "heartbroken", "euphoric", "wounded"):
        assert mood in VOCAB["mood"], f"mood {mood!r} missing from labels VOCAB"


# --------------------------------------------------------------------------- #
# lyrics.py engine
# --------------------------------------------------------------------------- #
def _ctx(**kw):
    base = dict(artist="future", mood="dark", topic="pain", seed=42)
    base.update(kw)
    return lyrics.BeatContext(**base)


def test_generate_is_deterministic_per_seed():
    a = lyrics.generate(_ctx(seed=1))
    b = lyrics.generate(_ctx(seed=1))
    assert a.full_text() == b.full_text()


def test_generate_differs_across_seeds():
    a = lyrics.generate(_ctx(seed=1))
    b = lyrics.generate(_ctx(seed=2))
    assert a.full_text() != b.full_text()


def test_generate_follows_default_structure():
    r = lyrics.generate(_ctx())
    roles = [s["role"] for s in r.sections]
    assert roles[0] == "intro" and roles[-1] == "outro"
    assert "verse" in roles and "hook" in roles
    assert all(s["lines"] for s in r.sections)


def test_generate_reports_artist_and_backend():
    r = lyrics.generate(_ctx(artist="lil-durk"))
    assert r.artist == "Lil Durk"
    assert r.backend == "offline"


def test_negative_terms_are_filtered():
    r = lyrics.generate(_ctx(topic="success", negative=["success", "money"]))
    joined = r.full_text().lower()
    assert "success" not in joined and "money" not in joined


def test_beat_context_from_analysis():
    rec = {
        "key": {"key": "E minor"},
        "beat_grid": {"tempo": 144.0},
        "swing": {"feel": "swung"},
        "structure": {"segments": [
            {"role": "intro", "energy": 0.2},
            {"role": "verse", "energy": 0.5},
            {"role": "chorus", "energy": 0.9},
            {"role": "outro", "energy": 0.2},
        ]},
    }
    ctx = lyrics.beat_context_from_analysis(rec, artist="gunna", mood="smooth", topic="wealth")
    assert ctx.bpm == 144.0
    assert ctx.key == "E minor"
    assert ctx.swing == "swung"
    assert ctx.artist == "gunna"
    assert [s.role for s in ctx.structure] == ["intro", "verse", "chorus", "outro"]


def test_regenerate_section():
    sec = lyrics.regenerate_section(_ctx(), "hook", seed=9)
    assert sec["role"] == "hook"
    assert sec["lines"]


def test_restyle_changes_artist():
    base = lyrics.generate(_ctx(artist="future", seed=3))
    restyled = lyrics.restyle(base, "chief-keef", seed=3)
    assert restyled.artist == "Chief Keef"
    assert restyled.full_text() != base.full_text()


# --------------------------------------------------------------------------- #
# lyricsprefs.py
# --------------------------------------------------------------------------- #
def test_favorite_roundtrip(tmp_path):
    lyricsprefs.add_favorite(tmp_path, "my-fav", {"artist": "future", "mood": "dark"})
    assert lyricsprefs.get_favorite(tmp_path, "my-fav")["artist"] == "future"
    assert "my-fav" in lyricsprefs.favorite_keys(tmp_path)
    assert lyricsprefs.remove_favorite(tmp_path, "my-fav") is True
    assert lyricsprefs.remove_favorite(tmp_path, "my-fav") is False


def test_random_recipe_is_valid(tmp_path):
    r = lyricsprefs.random_recipe(tmp_path, seed=1)
    assert artists.get_artist(r["artist"]) is not None
    assert r["mood"] in artists.MOODS
    assert r["topic"]


def test_weights_and_negatives(tmp_path):
    lyricsprefs.set_weight(tmp_path, "topic", 2.0)
    assert lyricsprefs.weights(tmp_path)["topic"] == 2.0
    lyricsprefs.add_negative(tmp_path, "violence")
    assert "violence" in lyricsprefs.negatives(tmp_path)
    lyricsprefs.remove_negative(tmp_path, "violence")
    assert "violence" not in lyricsprefs.negatives(tmp_path)


def test_history_and_diff(tmp_path):
    lyricsprefs.record_history(tmp_path, {"artist": "future", "mood": "dark", "seed": 1})
    lyricsprefs.record_history(tmp_path, {"artist": "drake", "mood": "smooth", "seed": 2})
    rows = lyricsprefs.history(tmp_path)
    assert len(rows) == 2
    lines = lyricsprefs.history_diff(rows[0], rows[1])
    assert any("artist" in ln for ln in lines)


def test_normalize_recipe_applies_genre_defaults():
    r = lyricsprefs.normalize_recipe(genre="drill")
    assert r["artist"]
    assert r["topic"]
    assert r["mood"] in ("menacing", "gritty", "aggressive")


# --------------------------------------------------------------------------- #
# lyricrating.py
# --------------------------------------------------------------------------- #
def test_rating_profile(tmp_path):
    lyricrating.record_rating(tmp_path, {"item": "a", "artist": "future", "mood": "dark", "score": 1.0})
    lyricrating.record_rating(tmp_path, {"item": "b", "artist": "future", "mood": "dark", "score": 0.5})
    lyricrating.record_rating(tmp_path, {"item": "c", "artist": "drake", "mood": "smooth", "score": 0.5})
    p = lyricrating.build_profile(tmp_path)
    assert p["n_ratings"] == 3
    assert lyricrating.top_preference(p, "artists") == "future"
    assert lyricrating.top_preference(p, "moods") == "dark"


def test_bias_recipe_overrides_with_strong_profile(tmp_path):
    lyricrating.record_rating(tmp_path, {"item": "a", "artist": "future", "mood": "dark",
                                         "topic": "pain", "genre": "melodic trap", "score": 1.0})
    out = lyricrating.bias_recipe(tmp_path, {"artist": "drake", "mood": "smooth"}, strength=0.8)
    assert out["artist"] == "future"


def test_build_queue_pairs_and_shuffles(tmp_path):
    items = [{"id": str(i), "label": str(i)} for i in range(6)]
    q = lyricrating.build_queue(tmp_path, items, n=6, seed=0)
    assert len(q) == 3
    for pair in q:
        assert "A" in pair and "B" in pair


# --------------------------------------------------------------------------- #
# CLI subcommands
# --------------------------------------------------------------------------- #
def test_cli_artists_lists_profiles(tmp_path, capsys):
    from musictrain.cli import main
    rc = main(["artists", "--root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "22 artist style profiles" in out


def test_cli_artists_show(tmp_path, capsys):
    from musictrain.cli import main
    rc = main(["artists", "--root", str(tmp_path), "--show", "future"])
    assert rc == 0
    assert "Future" in capsys.readouterr().out


def test_cli_lyrics_no_save(tmp_path, capsys):
    from musictrain.cli import main
    rc = main(["lyrics", "--root", str(tmp_path), "--artist", "future",
               "--mood", "dark", "--topic", "pain", "--seed", "5", "--no-save"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[intro" in out and "seed=5" in out
    # --no-save must not write history
    assert not (tmp_path / "metadata" / "lyric_history.jsonl").exists()


def test_cli_lyrics_section_regen(tmp_path, capsys):
    from musictrain.cli import main
    rc = main(["lyrics", "--root", str(tmp_path), "--artist", "future",
               "--section", "hook", "--seed", "5", "--no-save"])
    assert rc == 0
    assert "hook" in capsys.readouterr().out.lower()


def test_cli_lyrics_restyle(tmp_path, capsys):
    from musictrain.cli import main
    rc = main(["lyrics", "--root", str(tmp_path), "--artist", "future",
               "--restyle", "drake", "--seed", "5", "--no-save"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "restyled as Drake" in out


def test_cli_lyrate_profile(tmp_path, capsys):
    from musictrain.cli import main
    main(["lyrate", "--root", str(tmp_path), "--task", "record",
          "--item", "x", "--artist", "future", "--score", "0.8"])
    rc = main(["lyrate", "--root", str(tmp_path), "--task", "profile"])
    assert rc == 0
    assert "style profile" in capsys.readouterr().out
