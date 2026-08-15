"""Unit tests for Phase 6 — eval & metrics (#41-#46).

Exercises the pure logic without heavy model loads: FAD on synthetic
embeddings, spectral KL on synthetic audio, paired significance, leaderboard
ranking/merging, auto-reject thresholds, and per-tag CLAP aggregation.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from musictrain import leaderboard as lb
from musictrain import metrics as mt
from musictrain import significance as sig
from musictrain.config import Config
from musictrain.evalset import _aggregate, tag_phrases


def _cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.project_root = tmp_path
    return cfg


def _row(prompt: str, bpm: int, checkpoint: str = "ckpt-a",
         clap: float = 0.5, dev: float = 0.01, status: str = "ok",
         per_tag: dict | None = None) -> dict:
    return {
        "experiment_id": "test",
        "checkpoint": checkpoint,
        "prompt": prompt,
        "bpm_target": bpm,
        "section": "chorus",
        "clap_score": clap,
        "deviation": dev,
        "status": status,
        "clap_per_tag": per_tag or {"genre": clap, "key": clap},
    }


# --------------------------------------------------------------------------- #
# #41 FAD + spectral KL
# --------------------------------------------------------------------------- #

def test_frechet_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(20, 8))
    assert mt.fad_from_embeddings(embs, embs.copy()) == pytest.approx(0.0, abs=1e-9)


def test_frechet_separated_distributions_positive():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=-3.0, scale=0.5, size=(20, 8))
    b = rng.normal(loc=3.0, scale=0.5, size=(20, 8))
    assert mt.fad_from_embeddings(a, b) > 1.0


def test_frechet_insufficient_samples_returns_none():
    assert mt.fad_from_embeddings(np.zeros((1, 4)), np.zeros((2, 4))) is None


def test_spectral_kl_identical_files_is_zero(tmp_path):
    cfg = _cfg(tmp_path)
    import soundfile as sf

    sr = 32000
    t = np.linspace(0, 1.0, sr)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, tone, sr)
    sf.write(b, tone, sr)
    assert mt.kl_spectral([a], [b], cfg) == pytest.approx(0.0, abs=1e-4)


def test_spectral_kl_different_files_positive(tmp_path):
    cfg = _cfg(tmp_path)
    import soundfile as sf

    sr = 32000
    t = np.linspace(0, 1.0, sr)
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    sf.write(b, (0.5 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32), sr)
    assert mt.kl_spectral([a], [b], cfg) > 0.01


def test_kl_requires_both_sets(tmp_path):
    cfg = _cfg(tmp_path)
    assert mt.kl_spectral([], [tmp_path / "x.wav"], cfg) is None


# --------------------------------------------------------------------------- #
# #44 significance
# --------------------------------------------------------------------------- #

def _eval_rows(n: int, base_clap: float, base_dev: float, prompt_prefix: str):
    return [
        {
            "prompt": f"{prompt_prefix} prompt {i}",
            "bpm_target": 100,
            "clap_score": base_clap + i * 0.001,
            "deviation": base_dev,
        }
        for i in range(n)
    ]


def test_significance_pairs_on_shared_prompts(tmp_path):
    cfg = _cfg(tmp_path)
    a = _eval_rows(12, 0.40, 0.05, "p")
    b = _eval_rows(12, 0.60, 0.02, "p")
    out = sig.compare(cfg, a, b, label_a="A", label_b="B")
    assert out["n_paired"] == 12
    m = out["metrics"]["clap_score"]
    assert m["delta"] > 0  # B better
    assert m["p_value"] is not None and m["p_value"] < 0.05
    assert m["verdict"] == "improved"
    d = out["metrics"]["abs_deviation"]
    assert d["verdict"] == "improved"  # lower is better
    assert (cfg.project_root / "metadata" / "significance.json").exists()


def test_significance_identical_sets_no_difference(tmp_path):
    cfg = _cfg(tmp_path)
    rows = _eval_rows(10, 0.5, 0.03, "q")
    out = sig.compare(cfg, rows, [dict(r) for r in rows], label_a="A", label_b="B")
    assert out["metrics"]["clap_score"]["verdict"] == "no significant difference"


def test_significance_no_shared_prompts(tmp_path):
    cfg = _cfg(tmp_path)
    a = _eval_rows(3, 0.4, 0.05, "left")
    b = _eval_rows(3, 0.6, 0.02, "right")
    assert sig.compare(cfg, a, b) == {}


def test_from_checkpoints_filters(tmp_path):
    cfg = _cfg(tmp_path)
    rows = [_row("p1", 100, checkpoint="m-a", clap=0.4),
            _row("p1", 100, checkpoint="m-b", clap=0.7)]
    p = cfg.project_root / "metadata"
    p.mkdir(parents=True, exist_ok=True)
    (p / "eval_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    out = sig.from_checkpoints(cfg, "m-a", "m-b")
    assert out["n_paired"] == 1
    assert out["label_b"] == "m-b"


# --------------------------------------------------------------------------- #
# #45 leaderboard
# --------------------------------------------------------------------------- #

def test_leaderboard_ranks_and_merges(tmp_path):
    cfg = _cfg(tmp_path)
    p = cfg.project_root / "metadata"
    p.mkdir(parents=True, exist_ok=True)
    rows = [
        _row("good prompt", 100, checkpoint="ckpt-b", clap=0.8, dev=0.01, status="ok"),
        _row("meh prompt", 100, checkpoint="ckpt-a", clap=0.3, dev=0.15, status="rejected"),
    ]
    (p / "eval_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (p / "human_ratings.jsonl").write_text(
        json.dumps({"prompt": "good prompt", "checkpoint": "ckpt-b", "rating": 5}) + "\n"
    )

    out = lb.build(cfg)
    entries = out["leaderboard"]
    assert [e["checkpoint"] for e in entries] == ["ckpt-b", "ckpt-a"]
    assert entries[0]["rank"] == 1
    assert entries[0]["mean_human_rating"] == 5.0
    assert entries[0]["clap_per_tag"]["genre"] == 0.8
    assert (p / "leaderboard.json").exists()


def test_leaderboard_no_results(tmp_path):
    cfg = _cfg(tmp_path)
    assert lb.build(cfg) == {}


# --------------------------------------------------------------------------- #
# #43 auto-reject thresholds + #46 per-tag aggregation in _aggregate
# --------------------------------------------------------------------------- #

def _seed(clap=0.5, status="ok", dev=0.01, per_tag=None) -> dict:
    return {
        "seed": 1, "audio_path": "/tmp/x.wav", "detected_bpm": 100.0,
        "deviation": dev, "clap_score": clap, "status": status, "note": "",
        "clap_per_tag": per_tag or {"genre": clap, "bpm": clap},
    }


def test_aggregate_clap_below_threshold_rejects(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.eval.min_clap_score = 0.5
    agg = _aggregate({"bpm": 100, "description": "x"}, [_seed(clap=0.3)], cfg)
    assert agg["status"] == "rejected"
    assert "auto-reject" in agg["notes"]
    assert "CLAP" in agg["notes"]


def test_aggregate_deviation_above_threshold_rejects(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.eval.max_abs_deviation = 0.10
    seed = _seed(clap=0.8, status="ok", dev=0.25)
    seed["detected_bpm"] = 125.0  # (125-100)/100 = 0.25 deviation
    agg = _aggregate({"bpm": 100, "description": "x"}, [seed], cfg)
    assert agg["status"] == "rejected"
    assert "|dev|" in agg["notes"]


def test_check_folded_octave_reports_folded_deviation(tmp_path, monkeypatch):
    """Regression: octave-folded clips must report the folded (small) deviation
    so downstream auto-reject treats them as on-target, not raw double-time."""
    from musictrain import evaluate as ev

    cfg = _cfg(tmp_path)
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"x")

    monkeypatch.setattr(ev, "load_audio", lambda path, sr: (np.zeros(16000, dtype=np.float32), 32000))
    monkeypatch.setattr(ev, "estimate_bpm", lambda y, sr: 156.0)  # double-time of 78

    report = ev.check(cfg, wav, target_bpm=78.0)
    assert report["status"] == "ok"
    assert report["raw_deviation"] == pytest.approx(1.0, abs=1e-4)
    assert report["deviation"] == pytest.approx(0.0, abs=1e-2)  # folded 156*0.5 = 78
    assert report["folded_bpm"] == 78.0
    assert "double-time" in report["note"]


def test_check_folds_4_over_3_and_2_over_3(tmp_path, monkeypatch):
    """Triplet-grid folds: detected at 4/3x (96 vs 72) or 2/3x (104 vs 155)
    of the target must fold to ok, mirroring the octave folds."""
    from musictrain import evaluate as ev

    cfg = _cfg(tmp_path)
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"x")
    monkeypatch.setattr(ev, "load_audio", lambda path, sr: (np.zeros(16000, dtype=np.float32), 32000))

    # detected 96 vs target 72 -> 96 * 0.75 = 72 (4/3-time)
    monkeypatch.setattr(ev, "estimate_bpm", lambda y, sr: 96.0)
    r = ev.check(cfg, wav, target_bpm=72.0)
    assert r["status"] == "ok"
    assert r["folded_bpm"] == 72.0
    assert r["deviation"] == pytest.approx(0.0, abs=1e-2)
    assert "4/3-time" in r["note"]

    # detected 104 vs target 155 -> 104 * 1.5 = 156 (2/3-time)
    monkeypatch.setattr(ev, "estimate_bpm", lambda y, sr: 104.0)
    r = ev.check(cfg, wav, target_bpm=155.0)
    assert r["status"] == "ok"
    assert r["folded_bpm"] == 156.0
    assert r["deviation"] == pytest.approx(0.0065, abs=1e-3)
    assert "2/3-time" in r["note"]


def test_check_non_foldable_stays_rejected(tmp_path, monkeypatch):
    """A 1.179x mismatch (e.g. 70.7 vs 60) is not a tempo-ratio multiple and
    must stay rejected rather than being force-folded."""
    from musictrain import evaluate as ev

    cfg = _cfg(tmp_path)
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"x")
    monkeypatch.setattr(ev, "load_audio", lambda path, sr: (np.zeros(16000, dtype=np.float32), 32000))
    monkeypatch.setattr(ev, "estimate_bpm", lambda y, sr: 70.7)
    r = ev.check(cfg, wav, target_bpm=60.0)
    assert r["status"] == "rejected"


def test_aggregate_accepts_folded_deviation(tmp_path):
    """A seed whose check folded (status ok, small deviation) must not be
    auto-rejected on the raw double-time deviation."""
    cfg = _cfg(tmp_path)
    cfg.eval.max_abs_deviation = 0.20
    seed = _seed(clap=0.6, status="ok", dev=0.001)
    seed["detected_bpm"] = 156.0
    agg = _aggregate({"bpm": 78, "description": "x"}, [seed], cfg)
    assert agg["status"] == "ok"
    assert agg["notes"] == ""


def test_aggregate_passes_with_thresholds(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.eval.min_clap_score = 0.5
    cfg.eval.max_abs_deviation = 0.10
    agg = _aggregate(
        {"bpm": 100, "description": "x"},
        [_seed(clap=0.7, status="ok", dev=0.02)], cfg,
    )
    assert agg["status"] == "ok"
    assert agg["notes"] == ""


def test_aggregate_per_tag_mean(tmp_path):
    cfg = _cfg(tmp_path)
    agg = _aggregate(
        {"bpm": 100, "description": "x"},
        [
            _seed(clap=0.5, per_tag={"genre": 0.4, "bpm": 0.6}),
            _seed(clap=0.9, per_tag={"genre": 0.8, "bpm": 1.0}),
        ],
        cfg,
    )
    assert agg["clap_per_tag"]["genre"] == 0.6
    assert agg["clap_per_tag"]["bpm"] == 0.8


def test_tag_phrases_shape():
    phrases = tag_phrases(
        {
            "section": "chorus", "genre": "melodic trap", "key": "A minor",
            "mood": ["dark"], "instruments": ["piano", "808 bass"], "bpm": 140,
        }
    )
    assert phrases["section"] == "chorus"
    assert phrases["mood"] == "dark"
    assert phrases["instruments"] == "piano, 808 bass"
    assert phrases["bpm"] == "140 BPM"
