"""Unit tests for advanced training batch #21-#30.

Covers the pure training helpers (LR schedule, curriculum ordering, streaming
batches, CFG candidates) and checkpoint pruning (#25) — no real model load.
"""
from __future__ import annotations

import numpy as np
import pytest

from musictrain import finetune as ft
from musictrain.config import Config
from musictrain.registry import prune_checkpoints


# --------------------------------------------------------------------------- #
# #23 LR schedule
# --------------------------------------------------------------------------- #
def test_lr_warmup_ramps_up():
    assert ft.lr_schedule(0, lr=1e-3, warmup_steps=5, total_steps=100) == pytest.approx(2e-4)
    assert ft.lr_schedule(4, lr=1e-3, warmup_steps=5, total_steps=100) == pytest.approx(1e-3)


def test_lr_constant_after_warmup():
    for step in (5, 50, 99):
        assert ft.lr_schedule(step, 1e-3, warmup_steps=5, total_steps=100, mode="constant") == 1e-3


def test_lr_cosine_decays_to_zero():
    end = ft.lr_schedule(99, 1e-3, warmup_steps=0, total_steps=100, mode="cosine")
    assert end == pytest.approx(0.0, abs=1e-5)
    mid = ft.lr_schedule(49, 1e-3, warmup_steps=0, total_steps=100, mode="cosine")
    assert mid == pytest.approx(0.5e-3, abs=1e-4)


# --------------------------------------------------------------------------- #
# #27 curriculum ordering
# --------------------------------------------------------------------------- #
def test_sort_curriculum_by_size(tmp_path):
    a = tmp_path / "a.wav"; a.write_bytes(b"0" * 10)
    b = tmp_path / "b.wav"; b.write_bytes(b"0" * 1000)
    c = tmp_path / "c.wav"; c.write_bytes(b"0" * 100)
    pairs = [(c, "c"), (a, "a"), (b, "b")]
    ordered = ft.sort_curriculum(pairs)
    assert [p[0].name for p in ordered] == ["a.wav", "c.wav", "b.wav"]


def test_sort_curriculum_by_explicit_difficulty(tmp_path):
    a = tmp_path / "a.wav"; a.write_bytes(b"0")
    b = tmp_path / "b.wav"; b.write_bytes(b"0")
    pairs = [(b, "b"), (a, "a")]
    ordered = ft.sort_curriculum(pairs, difficulty={"a": 0.1, "b": 0.9})
    assert [p[0].name for p in ordered] == ["a.wav", "b.wav"]


# --------------------------------------------------------------------------- #
# #26 streaming batches
# --------------------------------------------------------------------------- #
def test_iter_batches_splits():
    pairs = [(f"{i}", str(i)) for i in range(7)]
    batches = list(ft.iter_batches(pairs, batch_size=3))
    assert [len(b) for b in batches] == [3, 3, 1]


# --------------------------------------------------------------------------- #
# #28 CFG candidates
# --------------------------------------------------------------------------- #
def test_guidance_candidates():
    out = ft.guidance_candidates(base=3.0, n=5)
    assert len(out) == 5
    assert all(v >= 0.5 for v in out)
    assert min(out) < 3.0 < max(out)


# --------------------------------------------------------------------------- #
# #25 checkpoint pruning
# --------------------------------------------------------------------------- #
def _make_checkpoint(root, name):
    d = root / "checkpoints" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(b"0")
    return d


def test_prune_keeps_top_by_leaderboard(tmp_path):
    cfg = Config()
    cfg.project_root = tmp_path
    for name in ("a", "b", "c", "d", "e"):
        _make_checkpoint(tmp_path, name)
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "leaderboard.json").write_text(
        '{"leaderboard": ['
        '{"checkpoint": "a", "score": 0.9}, {"checkpoint": "b", "score": 0.8},'
        '{"checkpoint": "c", "score": 0.5}, {"checkpoint": "d", "score": 0.4},'
        '{"checkpoint": "e", "score": 0.1}]}'
    )
    out = prune_checkpoints(tmp_path, cfg, keep=2)
    assert set(out["kept"]) == {"a", "b"}
    assert set(out["pruned"]) == {"c", "d", "e"}
    # archived, not deleted
    archives = tmp_path / "checkpoints" / "archives"
    assert archives.exists()
    assert len(list(archives.iterdir())) == 3


def test_prune_nothing_when_at_or_below_keep(tmp_path):
    cfg = Config()
    cfg.project_root = tmp_path
    for name in ("a", "b"):
        _make_checkpoint(tmp_path, name)
    out = prune_checkpoints(tmp_path, cfg, keep=2)
    assert out["pruned"] == []
    assert set(out["kept"]) == {"a", "b"}


def test_prune_delete_mode(tmp_path):
    cfg = Config()
    cfg.project_root = tmp_path
    for name in ("a", "b", "c"):
        _make_checkpoint(tmp_path, name)
    out = prune_checkpoints(tmp_path, cfg, keep=1, delete=True)
    assert len(out["pruned"]) == 2
    assert out["mode"] == "delete"
    # only one checkpoint dir remains (no archives dir created)
    remaining = [d.name for d in (tmp_path / "checkpoints").iterdir() if d.is_dir()]
    assert len(remaining) == 1
    assert "archives" not in remaining
