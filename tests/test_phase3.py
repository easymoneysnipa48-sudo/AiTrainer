"""Unit tests for Phase 3 — segmentation & splits.

Covers the pure logic: downbeat/overlap/fade window computation, ffmpeg slice
args, stratified + k-fold assignment, and export row joining. No ffmpeg, torch,
or datasets are loaded.
"""
from __future__ import annotations

import json

from musictrain import split as split_mod
from musictrain import export as export_mod
from musictrain.audio.segment import compute_windows, downbeat_windows, _slice_args
from musictrain.config import Config


def _cfg(**split_kwargs) -> Config:
    cfg = Config()
    for k, v in split_kwargs.items():
        setattr(cfg.split, k, v)
    return cfg


# --------------------------------------------------------------------------- #
# segment.py — window computation (#21, #24)
# --------------------------------------------------------------------------- #


def test_compute_windows_no_overlap():
    ws = compute_windows(duration=30.0, length=10.0, overlap=0.0, min_seconds=8.0)
    assert ws == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]


def test_compute_windows_with_overlap():
    # length 10, overlap 2 -> stride 8: [0,10],[8,18],[16,26],[24,30]
    ws = compute_windows(duration=30.0, length=10.0, overlap=2.0, min_seconds=1.0)
    assert ws == [(0.0, 10.0), (8.0, 18.0), (16.0, 26.0), (24.0, 30.0)]


def test_compute_windows_clamps_overlap_to_length():
    ws = compute_windows(duration=20.0, length=10.0, overlap=20.0, min_seconds=8.0)
    # overlap >= length -> clamped (to length - min_seconds) so stride stays positive
    assert ws[0][0] == 0.0
    assert all(ws[i][1] > ws[i][0] for i in range(len(ws)))  # every window positive
    assert ws[-1][1] <= 20.0  # never past duration


def test_compute_windows_drops_short_tail():
    ws = compute_windows(duration=15.0, length=10.0, overlap=0.0, min_seconds=8.0)
    # last window [10, 15] is 5s < 8s min -> dropped
    assert ws == [(0.0, 10.0)]


def test_downbeat_windows_align_to_downbeats():
    dbs = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    ws = downbeat_windows(dbs, bars_per_seg=2, duration=11.0)
    assert ws == [(0.0, 4.0), (4.0, 8.0), (8.0, 11.0)]


def test_downbeat_windows_empty():
    assert downbeat_windows([], bars_per_seg=2, duration=10.0) == []


def test_slice_args_basic():
    cfg = Config()
    args = _slice_args(
        __import__("pathlib").Path("in.wav"),
        __import__("pathlib").Path("out.wav"),
        0.0, 10.0, cfg, fade=0.0,
    )
    assert "-ss" in args and "0.0000" in args
    assert "-t" in args and "10.0000" in args
    assert "afade" not in args


def test_slice_args_includes_fade():
    cfg = Config()
    args = _slice_args(
        __import__("pathlib").Path("in.wav"),
        __import__("pathlib").Path("out.wav"),
        0.0, 10.0, cfg, fade=1.0,
    )
    af_idx = args.index("-af")
    assert "afade=t=in" in args[af_idx + 1]
    assert "afade=t=out" in args[af_idx + 1]


# --------------------------------------------------------------------------- #
# split.py — stratified (#23) + k-fold (#22)
# --------------------------------------------------------------------------- #


def test_assign_random_ratios():
    cfg = _cfg(train=0.6, val=0.2, test=0.2, seed=1)
    songs = [f"s{i}" for i in range(10)]
    a = split_mod._assign(songs, cfg, __import__("pathlib").Path("."))
    assert len(a["train"]) == 6
    assert len(a["val"]) == 2
    assert len(a["test"]) == 2
    # no overlap across splits
    all_s = set(a["train"] + a["val"] + a["test"])
    assert all_s == set(songs)


def test_assign_stratified_balanced(tmp_path):
    # write a manifest with two genres; stratify must put both in each split
    manifest = tmp_path / "metadata" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(8):
        genre = "trap" if i < 4 else "ambient"
        rows.append({"path": f"data/clean/s{i}.wav", "genre": genre, "key": "A minor", "bpm": 90})
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))

    cfg = _cfg(train=0.5, val=0.25, test=0.25, seed=42, stratify="genre")
    songs = [f"s{i}" for i in range(8)]
    a = split_mod._assign(songs, cfg, tmp_path)
    for split in ("train", "val", "test"):
        genres = set()
        for s in a[split]:
            idx = int(s[1:])
            genres.add("trap" if idx < 4 else "ambient")
        assert genres == {"trap", "ambient"}, f"{split} missing a genre"


def test_stratify_key_buckets_bpm():
    cfg = Config()
    # bpm stratified into 10-BPM buckets
    assert split_mod._stratify_key("x", {"x": {"bpm": 92}}, "bpm") == "90s"
    assert split_mod._stratify_key("x", {"x": {"bpm": 140}}, "bpm") == "140s"
    assert split_mod._stratify_key("x", {}, "bpm") == "unknown"


def test_assign_kfold_sizes():
    songs = [f"s{i}" for i in range(10)]
    folds = split_mod._assign_kfold(songs, k=5, seed=42)
    assert len(folds) == 5
    # each song appears exactly once in val, k-1 times in train
    from collections import Counter

    val_count = Counter()
    for f in folds:
        assert len(f["val"]) == 2
        assert len(f["train"]) == 8
        val_count.update(f["val"])
    assert set(val_count.values()) == {1}


# --------------------------------------------------------------------------- #
# export.py — row joining (#26)
# --------------------------------------------------------------------------- #


def test_export_text_from_description():
    assert export_mod._text({"description": "hook, 96 BPM, A minor"}) == "hook, 96 BPM, A minor"


def test_export_text_assembled_from_fields():
    rec = {
        "section": "chorus",
        "genre": "melodic trap",
        "mood": ["dark", "emotional"],
        "instruments": ["808 bass", "trap hi-hats"],
        "bpm": 96,
        "key": "A minor",
    }
    text = export_mod._text(rec)
    assert "chorus" in text and "melodic trap" in text
    assert "dark, emotional" in text
    assert "96 BPM" in text and "A minor" in text


def test_export_load_split_rows_joins_features(tmp_path):
    # feature manifest keyed by song_id
    feat = tmp_path / "metadata" / "manifest.jsonl"
    feat.parent.mkdir(parents=True, exist_ok=True)
    feat.write_text(
        json.dumps({"path": "data/clean/song_a.wav", "source_id": "song_a",
                    "genre": "trap", "mood": ["dark"], "bpm": 96, "key": "A minor"}) + "\n"
    )
    # train split manifest referencing song_a
    train = tmp_path / "metadata" / "train.jsonl"
    train.write_text(
        json.dumps({"path": "data/train/song_a_seg000.wav", "song_id": "song_a", "split": "train"}) + "\n"
    )
    rows = export_mod._load_split_rows(tmp_path, "train")
    assert len(rows) == 1
    row = rows[0]
    assert row["genre"] == "trap"
    assert row["bpm"] == 96
    assert row["split"] == "train"
    assert "trap" in row["text"]
