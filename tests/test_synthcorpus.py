"""Tests for synthcorpus: deterministic generation, idempotent merging, and
the append_records merge helper used by lyricscrape + synthcorpus."""
from __future__ import annotations

import json

from musictrain import lyricdataset as LD
from musictrain import synthcorpus as SC


def test_generate_corpus_writes_songs(tmp_path):
    summary = SC.generate_corpus(tmp_path, artists=["future", "drake"],
                                 per_artist=3, seed=7)
    assert summary["added"] == 6
    for aid in ("future", "drake"):
        rows = [json.loads(line) for line in (tmp_path / "lyrics" / aid / "songs.jsonl").open()]
        assert len(rows) == 3
        for r in rows:
            assert len(r["lines"]) >= 8
            assert r["mood"] and r["topic"]
            assert r["source"] == "synthetic-seed7"


def test_generate_corpus_is_idempotent(tmp_path):
    SC.generate_corpus(tmp_path, artists=["future"], per_artist=4, seed=7)
    again = SC.generate_corpus(tmp_path, artists=["future"], per_artist=4, seed=7)
    assert again["added"] == 0
    rows = [json.loads(line) for line in (tmp_path / "lyrics" / "future" / "songs.jsonl").open()]
    assert len(rows) == 4


def test_generate_corpus_dry_run_writes_nothing(tmp_path):
    summary = SC.generate_corpus(tmp_path, artists=["future"], per_artist=3,
                                 seed=7, dry_run=True)
    assert summary["added"] == 3
    assert not (tmp_path / "lyrics").exists()


def test_append_records_merges_and_preserves_others(tmp_path):
    root = tmp_path / "root"
    (root / "lyrics" / "future").mkdir(parents=True)
    (root / "lyrics" / "future" / "songs.jsonl").write_text(json.dumps({
        "artist": "Future", "artist_id": "future", "title": "Old Song",
        "lines": ["a", "b", "c", "d"], "source": "existing",
    }) + "\n")
    (root / "lyrics" / "index.json").write_text(
        json.dumps({"future": {"display": "Future", "songs": 1, "lines": 4}}))

    records = [
        {"artist": "Future", "title": "New Song", "lines": ["x", "y", "z", "w"],
         "source": "scrape"},
        {"artist": "Future", "title": "Old Song", "lines": ["dup"], "source": "scrape"},
        {"artist": "Drake", "title": "Fresh", "lines": ["1", "2", "3", "4"],
         "source": "scrape"},
    ]
    out = LD.append_records(root, records)
    assert out["added"] == 2  # New Song + Drake's Fresh; Old Song is a dup

    future = [json.loads(line) for line in (root / "lyrics" / "future" / "songs.jsonl").open()]
    assert {r["title"] for r in future} == {"Old Song", "New Song"}

    index = json.loads((root / "lyrics" / "index.json").read_text())
    assert index["future"]["songs"] == 2
    assert index["drake"]["songs"] == 1


def test_import_txt_artist_dash_title(tmp_path):
    p = tmp_path / "Future - March Madness.txt"
    p.write_text("grindin' day and night\nno days off\nI was made for the mission\nit's in my blood\n")
    rows = LD.import_records(p)
    assert len(rows) == 1
    assert rows[0]["artist"] == "Future"
    assert rows[0]["title"] == "March Madness"
    assert len(rows[0]["lines"]) == 4
