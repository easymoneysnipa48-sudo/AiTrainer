"""Tests for the gap-closing training/inference features (#1-#5).

* #1/#2 — base-vs-fine-tuned A/B eval win-rate math (no model load needed)
* #3   — optimizer/checkpoint state save + resume round-trip
* #4/#5 — dry-run train() honors full/accum/resume without touching torch
"""
from __future__ import annotations

import json

import pytest

from musictrain import ab_eval, finetune as ft
from musictrain.config import Config


# --------------------------------------------------------------------------- #
# #2 A/B eval win-rate (pure)
# --------------------------------------------------------------------------- #
def _rows(claps, devs):
    return [
        {"prompt": f"p{i}", "bpm_target": 100, "clap_score": c, "deviation": d, "status": "ok"}
        for i, (c, d) in enumerate(zip(claps, devs))
    ]


def test_win_rate_clap_higher_better():
    base = _rows([0.3, 0.4, 0.5], [0.1, 0.1, 0.1])
    adapt = _rows([0.4, 0.4, 0.5], [0.1, 0.1, 0.1])
    out = ab_eval._win_rate(base, adapt, "clap_score", higher_is_better=True)
    assert out["n"] == 3
    assert out["wins"] == 1 and out["ties"] == 2 and out["losses"] == 0
    assert out["win_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_win_rate_deviation_lower_better():
    base = _rows([0.5, 0.5, 0.5], [0.10, 0.10, 0.10])
    adapt = _rows([0.5, 0.5, 0.5], [0.05, 0.20, 0.10])
    out = ab_eval._win_rate(base, adapt, "deviation", higher_is_better=False)
    # adapt wins where its |dev| is lower: 0.05<0.10 (win), 0.20>0.10 (loss), tie
    assert out["wins"] == 1 and out["losses"] == 1 and out["ties"] == 1


def test_win_rate_empty_on_missing_metric():
    base = _rows([None, None], [None, None])
    adapt = _rows([None, None], [None, None])
    out = ab_eval._win_rate(base, adapt, "clap_score", higher_is_better=True)
    assert out["n"] == 0 and out["win_rate"] is None


# --------------------------------------------------------------------------- #
# #3 checkpoint state save/resume
# --------------------------------------------------------------------------- #
def _tiny_optimizer():
    import torch
    import torch.nn as nn

    lin = nn.Linear(4, 4)
    return torch.optim.AdamW(lin.parameters(), lr=1e-3)


def test_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("torch")
    opt = _tiny_optimizer()
    ckpt = tmp_path / "ckpt"
    ft._save_checkpoint(ckpt, opt, opt_step=7, step=3, losses=[1.0, 2.0],
                        val_losses=[0.5], ema=False, ema_params=[], meta={"mode": "lora"})
    assert (ckpt / "trainer_state.pt").exists()
    assert (ckpt / "trainer_state.json").exists()

    opt2 = _tiny_optimizer()
    st = ft._load_checkpoint_state(ckpt, opt2, [], False)
    assert st["opt_step"] == 7
    assert st["step"] == 3
    assert st["losses"] == [1.0, 2.0]
    assert st["val_losses"] == [0.5]


def test_checkpoint_load_missing_dir_returns_zeros(tmp_path):
    # the missing-dir path returns before touching the optimizer, so no torch needed
    st = ft._load_checkpoint_state(tmp_path / "nope", None, [], False)
    assert st == {"opt_step": 0, "step": 0, "losses": [], "val_losses": []}


# --------------------------------------------------------------------------- #
# #4/#5 dry-run train() honors full/accum/resume (no torch)
# --------------------------------------------------------------------------- #
def _make_project(tmp_path):
    seg = tmp_path / "data" / "segments"
    seg.mkdir(parents=True)
    (seg / "song1_seg0.wav").write_bytes(b"0" * 64)
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "manifest.jsonl").write_text(
        json.dumps({"path": "data/clean/song1.wav", "description": "dark melodic trap"}) + "\n"
    )
    return tmp_path


def test_train_dry_run_full(tmp_path):
    cfg = Config()
    cfg.project_root = _make_project(tmp_path)
    out = ft.train(cfg, steps=0, full=True, accum=2, resume="adapters/x")
    assert out["dry_run"] is True
    assert out["n_pairs"] == 1
    assert out["full"] is True


def test_train_dry_run_lora(tmp_path):
    cfg = Config()
    cfg.project_root = _make_project(tmp_path)
    out = ft.train(cfg, steps=0, full=False)
    assert out["dry_run"] is True
    assert out["full"] is False


def test_train_no_pairs(tmp_path):
    cfg = Config()
    cfg.project_root = tmp_path
    out = ft.train(cfg, steps=1)
    assert out == {}
