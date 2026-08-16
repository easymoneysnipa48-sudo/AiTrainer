"""Tests for gap batch 4 (#14 dataset versioning, #7 listening campaign)."""
from __future__ import annotations

import json

import pytest

from musictrain import dataversion as dv
from musictrain import listening_campaign as lc


# --------------------------------------------------------------------------- #
# #14 content-addressed dataset versioning
# --------------------------------------------------------------------------- #
def _write(root, rel, content=b"x"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def test_commit_and_list(tmp_path):
    _write(tmp_path, "data/clean/a.wav", b"AAA")
    _write(tmp_path, "data/clean/b.wav", b"BBB")
    out = dv.commit(tmp_path, which="clean", label="baseline")
    assert out["version"]["n_files"] == 2
    assert out["n_stored"] == 2
    assert (tmp_path / "data_versions" / "objects").exists()
    versions = dv.load_versions(tmp_path)
    assert len(versions) == 1 and versions[0]["name"] == "v1"


def test_commit_dedupes_identical_content(tmp_path):
    _write(tmp_path, "data/clean/a.wav", b"SAME")
    _write(tmp_path, "data/clean/b.wav", b"SAME")
    out = dv.commit(tmp_path, which="clean")
    assert out["n_stored"] == 1  # identical bytes -> one blob


def test_diff_added_removed_changed(tmp_path):
    _write(tmp_path, "data/clean/a.wav", b"AAA")
    _write(tmp_path, "data/clean/b.wav", b"BBB")
    dv.commit(tmp_path, which="clean", label="v1")

    _write(tmp_path, "data/clean/b.wav", b"BBB2")  # changed
    _write(tmp_path, "data/clean/c.wav", b"CCC")   # added
    (tmp_path / "data/clean/a.wav").unlink()        # removed
    dv.commit(tmp_path, which="clean", label="v2")

    out = dv.diff(tmp_path, "v1", "v2")
    assert out["n_added"] == 1 and "c.wav" in out["added"]
    assert out["n_removed"] == 1 and "a.wav" in out["removed"]
    assert out["n_changed"] == 1 and "b.wav" in out["changed"]


def test_rollback_restores(tmp_path):
    _write(tmp_path, "data/clean/a.wav", b"ORIGINAL")
    dv.commit(tmp_path, which="clean", label="v1")
    _write(tmp_path, "data/clean/a.wav", b"CHANGED")
    dv.commit(tmp_path, which="clean", label="v2")

    out = dv.rollback(tmp_path, "v1")
    assert out["restored"] == 1
    assert (tmp_path / "data/clean/a.wav").read_bytes() == b"ORIGINAL"


def test_commit_missing_dir(tmp_path):
    out = dv.commit(tmp_path, which="nope")
    assert "error" in out


# --------------------------------------------------------------------------- #
# #7 listening campaign
# --------------------------------------------------------------------------- #
def _eval_rows(root, n_checkpoints=2):
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(3):
        for c in range(n_checkpoints):
            ap = root / "outputs" / f"p{i}_c{c}.wav"
            ap.parent.mkdir(parents=True, exist_ok=True)
            ap.write_bytes(b"RIFF" + b"\x00" * 100)
            rows.append({"prompt": f"p{i}", "checkpoint": f"ck{c}",
                         "audio_path": str(ap), "clap_score": 0.5, "deviation": 0.01,
                         "section": "chorus", "bpm_target": 96})
    (meta / "eval_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_campaign_start_ab_blinds(tmp_path):
    _eval_rows(tmp_path, n_checkpoints=2)
    camp = lc.start(tmp_path, "test", mode="ab", seed=0)
    assert camp["n_items"] == 3
    it = camp["items"][0]
    assert it["mode"] == "ab"
    assert set(it["x"]["checkpoint"], ) and set(it["y"]["checkpoint"])
    assert it["x"]["checkpoint"] != it["y"]["checkpoint"]


def test_campaign_record_and_agreement(tmp_path):
    _eval_rows(tmp_path, n_checkpoints=2)
    lc.start(tmp_path, "test", mode="ab", seed=0)
    camp = lc.load_campaign(tmp_path, "test")
    item = camp["items"][0]["id"]
    lc.record(tmp_path, "test", "r1", item, "X")
    lc.record(tmp_path, "test", "r2", item, "X")
    lc.record(tmp_path, "test", "r3", item, "Y")
    a = lc.agreement(tmp_path, "test")
    assert a["n_ratings"] == 3 and a["n_raters"] == 3
    assert a["agreement"] == pytest.approx(2 / 3, abs=1e-4)


def test_campaign_unblind(tmp_path):
    _eval_rows(tmp_path, n_checkpoints=2)
    lc.start(tmp_path, "test", mode="ab", seed=0)
    mapping = lc.unblind(tmp_path, "test")
    assert len(mapping) == 3
    first = next(iter(mapping.values()))
    assert set(first.values()) == {"ck0", "ck1"}


def test_campaign_record_unknown(tmp_path):
    out = lc.record(tmp_path, "nope", "r1", "x", "X")
    assert "error" in out
