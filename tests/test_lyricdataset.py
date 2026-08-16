"""Tests for lyric dataset import/prep: rap CSV grouping, artist normalization,
song-level splits, and instruction-format generation."""
from __future__ import annotations

import json
from pathlib import Path

from musictrain import lyricdataset as LD


def _rap_csv(path: Path) -> Path:
    path.write_text(
        "artist,song,lyric,next lyric\n"
        "Drake,God's Plan,I been movin' through the rain,next\n"
        "Drake,God's Plan,they was doubtin' me,next\n"
        "Drake,God's Plan,now they all remember the name,next\n"
        "Drake,God's Plan,whole city on my back,next\n"
        "Future,March Madness,grindin' day and night,next\n"
        "Future,March Madness,no days off,next\n"
        "Future,March Madness,I was made for the mission,next\n"
        "Future,March Madness,it's in my blood,next\n"
        "Jay Z,Empire State,concrete jungle where dreams are made,next\n"
        "Jay Z,Empire State,ain't no sleep till Brooklyn,next\n"
        "Jay Z,Empire State,from the bottom to the top,next\n"
        "Jay Z,Empire State,that's the come-up,next\n",
        encoding="utf-8",
    )
    return path


def test_rap_csv_groups_rows_into_songs(tmp_path):
    p = _rap_csv(tmp_path / "rap.csv")
    songs = LD.import_rap_csv(p)
    assert len(songs) == 3
    by_title = {s["title"]: s for s in songs}
    assert len(by_title["God's Plan"]["lines"]) == 4
    assert len(by_title["March Madness"]["lines"]) == 4


def test_rap_csv_artist_filter(tmp_path):
    p = _rap_csv(tmp_path / "rap.csv")
    songs = LD.import_rap_csv(p, artists=["future"])
    assert [s["artist"] for s in songs] == ["Future"]


def test_rap_csv_drops_noise_lines(tmp_path):
    p = _rap_csv(tmp_path / "rap.csv")
    p.write_text(
        "artist,song,lyric,next lyric\n"
        "Drake,Test,[Intro],x\n"
        "Drake,Test,produced by metro, x\n"
        "Drake,Test,real line one,x\n"
        "Drake,Test,real line two,x\n"
        "Drake,Test,real line three,x\n"
        "Drake,Test,real line four,x\n",
        encoding="utf-8",
    )
    songs = LD.import_rap_csv(p)
    lines = songs[0]["lines"]
    assert "[Intro]" not in lines and "produced by metro" not in lines
    assert len(lines) == 4


def test_normalize_artist_maps_to_profile_ids():
    assert LD.normalize_artist("Jay Z") == ("jay-z", "Jay-Z")
    assert LD.normalize_artist("Kendrick Lamar")[0] == "kendrick-lamar"
    assert LD.normalize_artist("Fetty Wap") == ("fetty-wap", "Fetty Wap")


def test_build_dataset_writes_songs_and_index(tmp_path):
    src = tmp_path / "rap.csv"
    _rap_csv(src)
    root = tmp_path / "root"
    summary = LD.build_dataset(root, [src])
    assert summary["songs"] == 3
    assert (root / "lyrics" / "drake" / "songs.jsonl").exists()
    idx = json.loads((root / "lyrics" / "index.json").read_text())
    assert idx["drake"]["songs"] == 1
    rec = json.loads((root / "lyrics" / "jay-z" / "songs.jsonl").read_text().splitlines()[0])
    assert rec["artist_id"] == "jay-z" and rec["n_lines"] == 4


def test_split_dataset_is_deterministic(tmp_path):
    src = tmp_path / "rap.csv"
    _rap_csv(src)
    root = tmp_path / "root"
    LD.build_dataset(root, [src])
    # need >=10 songs per artist for a val/test holdout; synthesize many
    lines = ["line one", "line two", "line three", "line four"]
    songs_path = root / "lyrics" / "drake" / "songs.jsonl"
    with songs_path.open("w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({
                "artist": "Drake", "artist_id": "drake", "title": f"s{i}",
                "lines": lines, "n_lines": 4, "source": "x",
            }) + "\n")
    a = LD.split_dataset(root, seed=7)
    b = LD.split_dataset(root, seed=7)
    assert a == b
    # 20 drake songs (split into train/val/test) + 1 each for future/jay-z
    assert a["train"] + a["val"] + a["test"] == 22
    # every record ends with a newline, so the count == number of records
    train_rows = (root / "metadata" / "lyrics_train.jsonl").read_text().count("\n")
    assert train_rows == a["train"]


def test_instruction_format(tmp_path):
    src = tmp_path / "rap.csv"
    _rap_csv(src)
    root = tmp_path / "root"
    LD.build_dataset(root, [src])
    LD.split_dataset(root, seed=1)
    counts = LD.write_training_files(root)
    assert counts["train"] > 0  # val may be 0 when every artist has <10 songs
    ex = json.loads((root / "metadata" / "lyrics_train_instructions.jsonl")
                    .read_text().splitlines()[0])
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "ghostwriter in the style of" in ex["messages"][0]["content"]
    assert ex["messages"][-1]["content"]  # target lyrics present
