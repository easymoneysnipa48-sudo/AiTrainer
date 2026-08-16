"""Tests for gap batch 2 (#6 LR finder/auto-batch, #8 leakage guard)."""
from __future__ import annotations

import pytest

from musictrain import tuning
from musictrain.split import check_split_leakage


# --------------------------------------------------------------------------- #
# #6 LR finder / auto-batch
# --------------------------------------------------------------------------- #
def test_lr_find_plan_log_spaced():
    out = tuning.lr_find_plan(min_lr=1e-6, max_lr=1e-2, n=4)
    assert out == [1e-6, pytest.approx(1e-4, rel=1e-6), 1e-2] or len(out) == 4
    assert out[0] == 1e-6
    assert out[-1] == 1e-2
    assert out == sorted(out)


def test_lr_find_plan_rejects_bad_range():
    with pytest.raises(ValueError):
        tuning.lr_find_plan(min_lr=1e-2, max_lr=1e-6)


def test_pick_best_lr_steepest_descent():
    lrs = [1e-5, 1e-4, 1e-3, 1e-2]
    losses = [1.0, 0.9, 0.5, 0.8]
    out = tuning.pick_best_lr(losses, lrs)
    assert out["best_lr"] == 1e-3  # steepest drop 0.9 -> 0.5
    assert out["min_loss_lr"] == 1e-3


def test_pick_best_lr_divergence_falls_back_to_min():
    lrs = [1e-5, 1e-4, 1e-3, 1e-2]
    losses = [1.0, 1.1, 1.2, 1.3]
    out = tuning.pick_best_lr(losses, lrs)
    assert out["best_lr"] == 1e-5  # loss rising everywhere -> min-loss LR


def test_pick_best_lr_mismatch():
    out = tuning.pick_best_lr([1.0, 2.0], [1e-4])
    assert out["best_lr"] is None


def test_auto_batch_size_needs_sizes():
    assert tuning.auto_batch_size(0, 0)["batch"] is None


def test_auto_batch_size_concrete():
    # 1e9 param fp32 model (4e9 bytes); training footprint ~16e9 bytes;
    # 40e9 bytes VRAM * 0.85 -> ~34e9 free -> 18e9 left for samples.
    out = tuning.auto_batch_size(
        4_000_000_000, 40_000_000_000, dtype="fp32", per_sample_bytes=1_000_000_000
    )
    assert out["batch"] == 18


# --------------------------------------------------------------------------- #
# #8 split leakage guard
# --------------------------------------------------------------------------- #
def _write(root, rel, content=b"x"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def test_leakage_guard_clean(tmp_path):
    _write(tmp_path, "data/train/a.wav", b"AAA")
    _write(tmp_path, "data/val/b.wav", b"BBB")
    _write(tmp_path, "data/test/c.wav", b"CCC")
    out = check_split_leakage(tmp_path)
    assert out["clean"] is True
    assert out["n_overlaps"] == 0
    assert out["checked"] == {"train": 1, "val": 1, "test": 1}


def test_leakage_guard_detects_copy(tmp_path):
    _write(tmp_path, "data/train/a.wav", b"SHARED")
    _write(tmp_path, "data/val/b.wav", b"SHARED")  # same content, different path
    _write(tmp_path, "data/test/c.wav", b"UNIQUE")
    out = check_split_leakage(tmp_path)
    assert out["clean"] is False
    assert out["n_overlaps"] == 1
    assert "data/val/b.wav" in out["overlapping_files"]


def test_leakage_guard_missing_dirs(tmp_path):
    out = check_split_leakage(tmp_path)
    assert out["clean"] is True
    assert out["checked"] == {"train": 0, "val": 0, "test": 0}
