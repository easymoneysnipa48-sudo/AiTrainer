"""Tests for the lyrics fine-tune helpers (trainlyrics) and the audio label
generator (beatlabels) — pure-logic paths only, no model downloads."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from musictrain import beatlabels as BL
from musictrain import trainlyrics as TL


# --------------------------------------------------------------------------- #
# trainlyrics — pure helpers
# --------------------------------------------------------------------------- #
def test_run_dir_creates_unique_dirs(tmp_path):
    a = TL.run_dir(tmp_path / "out", tag="x")
    b = TL.run_dir(tmp_path / "out", tag="x")
    assert a != b and a.parent == b.parent == tmp_path / "out"
    assert a.exists() and b.exists()
    assert a.name.startswith("x-")


def test_resolve_device_returns_known_value():
    assert TL.resolve_device() in ("cuda", "mps", "cpu")


def test_load_examples_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TL._load_examples(tmp_path / "nope.jsonl")


def test_train_dry_run_returns_plan(tmp_path):
    meta = tmp_path / "root" / "metadata"
    meta.mkdir(parents=True)
    ex = {"messages": [
        {"role": "system", "content": "ghostwriter in the style of Drake"},
        {"role": "user", "content": "Write a verse"},
        {"role": "assistant", "content": "line one\nline two\nline three\nline four"},
    ]}
    (meta / "lyrics_train_instructions.jsonl").write_text(
        json.dumps(ex) + "\n" + json.dumps(ex) + "\n", encoding="utf-8")
    (meta / "lyrics_val_instructions.jsonl").write_text(
        json.dumps(ex) + "\n", encoding="utf-8")
    plan = TL.train(tmp_path / "root", steps=5, dry_run=True)
    assert plan["train_examples"] == 2 and plan["val_examples"] == 1
    assert plan["device"] in ("cuda", "mps", "cpu")
    assert plan["run_dir"]


def test_tokenize_example_masks_prompt(tmp_path):
    """Tokenize needs a real tokenizer; verify masking with a minimal stub."""
    class _Tok:
        eos_token_id = 0
        pad_token_id = 0

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "PROMPT"

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [5, 6, 7]}  # every chunk -> 3 tokens

    ex = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": "body"},
    ]}
    out = TL._tokenize_example(_Tok(), ex, max_len=12)
    assert len(out["input_ids"]) == len(out["labels"])
    assert out["labels"][0] == -100  # prompt masked
    assert out["labels"][-1] == 0    # eos contributes loss


# --------------------------------------------------------------------------- #
# beatlabels
# --------------------------------------------------------------------------- #
def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "data" / "segments").mkdir(parents=True)
    (root / "metadata").mkdir(parents=True)
    (root / "metadata" / "manifest.jsonl").write_text(
        json.dumps({
            "path": "data/clean/Song One.wav",
            "source_id": "song_one",
            "bpm": 156.25, "key": "G major",
        }) + "\n", encoding="utf-8")
    for i in range(3):
        (root / "data" / "segments" / f"song_one_seg00{i}.wav").touch()
    return root


def test_beatlabels_writes_rows_keyed_to_segments(tmp_path):
    root = _make_root(tmp_path)
    n = BL.generate_labels(root)
    assert n == 3
    with (root / "metadata" / "labels.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["source_id"] for r in rows} == {
        "song_one_seg000", "song_one_seg001", "song_one_seg002"}
    first = rows[0]
    assert "156 BPM" in first["description"] and "G major" in first["description"]
    assert first["section"] == "intro"
    assert rows[-1]["section"] == "outro"  # last segment closes


def test_beatlabels_refuses_overwrite_without_force(tmp_path):
    root = _make_root(tmp_path)
    BL.generate_labels(root)
    assert BL.generate_labels(root) == 0  # refuses, keeps existing
    BL.generate_labels(root, force=True)  # overwrites
    assert (root / "metadata" / "labels.csv").exists()


def test_beatlabels_finetune_pairs_resolve(tmp_path):
    root = _make_root(tmp_path)
    BL.generate_labels(root)
    from musictrain.finetune import _pairs
    pairs = _pairs(root)
    assert len(pairs) == 3  # every segment now has a description
