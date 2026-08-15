import math

import pytest

from musictrain import evalx


def test_mos_proxy_needs_clap():
    assert evalx.mos_proxy(None) is None


def test_mos_proxy_bounds_and_weights():
    s = evalx.mos_proxy(0.5, clipping=0.0, silence=0.0, snr_db=0.0)
    assert 0.0 <= s <= 1.0
    # clean audio, no silence, 0 dB SNR -> clap dominates
    assert s > 0.3


def test_mos_proxy_penalizes_artifacts():
    clean = evalx.mos_proxy(0.5, clipping=0.0, silence=0.0, snr_db=20.0)
    dirty = evalx.mos_proxy(0.5, clipping=0.1, silence=0.5, snr_db=-20.0)
    assert clean > dirty


def test_listening_ab_no_data():
    out = evalx.listening_ab(0, 0)
    assert out["verdict"] == "no_data"
    assert out["p_value"] is None


def test_listening_ab_symmetric():
    out = evalx.listening_ab(6, 6)
    assert out["win_rate_a"] == 0.5
    assert out["verdict"] == "no_significant_difference"
    assert out["p_value"] == pytest.approx(1.0, abs=0.001)


def test_listening_ab_significant_win():
    out = evalx.listening_ab(15, 5)
    assert out["verdict"] == "a_better"
    assert out["p_value"] < 0.05
    assert out["win_rate_a"] == 0.75


def test_listening_ab_ties_excluded():
    out = evalx.listening_ab(8, 8, ties=10)
    assert out["n"] == 16
    assert out["ties"] == 10


def test_embedding_leakage_detects_duplicate():
    ref = [[1.0, 0.0], [0.0, 1.0]]
    gen = [[1.0, 0.0], [0.5, 0.5], [0.0, -1.0]]
    out = evalx.embedding_leakage(ref, gen, threshold=0.9)
    assert out["n_ref"] == 2
    assert out["n_gen"] == 3
    assert out["n_leaks"] == 1  # exact match to ref[0]


def test_embedding_leakage_empty():
    out = evalx.embedding_leakage([], [])
    assert out["leak_rate"] == 0.0
    assert out["n_leaks"] == 0


def test_robustness_prompts_seeded_and_reversible():
    prompts = ["dark trap chorus with 808", "melodic synth intro"]
    out = evalx.robustness_prompts(prompts, n=0, seed=1)
    assert len(out) == 2
    for r in out:
        assert r["original"] in prompts
        assert isinstance(r["perturbed"], str)


def test_robustness_prompts_subset():
    prompts = [f"prompt {i} chorus" for i in range(10)]
    out = evalx.robustness_prompts(prompts, n=3, seed=5)
    assert len(out) == 3


def test_per_genre_gate_passes_and_fails():
    rows = [
        {"genre": "melodic trap", "clap_score": 0.5, "deviation": 0.01},
        {"genre": "melodic trap", "clap_score": 0.4, "deviation": 0.02},
        {"genre": "ambient", "clap_score": 0.1, "deviation": 0.5},
    ]
    gates = {
        "melodic trap": {"min_clap": 0.3, "max_abs_deviation": 0.15},
        "ambient": {"min_clap": 0.2, "max_abs_deviation": 0.3},
    }
    out = evalx.per_genre_gate(rows, gates)
    assert out["passed"] is False
    assert out["genres"]["melodic trap"]["passed"] is True
    assert out["genres"]["ambient"]["passed"] is False


def test_per_genre_gate_default_fallback():
    rows = [{"genre": "orchestral", "clap_score": 0.5, "deviation": 0.01}]
    out = evalx.per_genre_gate(rows, {})
    assert out["passed"] is True  # default gate (0.0 clap / 0.2 dev)


def test_fad_gate_below_threshold():
    out = evalx.fad_gate(5.0, threshold=10.0)
    assert out["passed"] is True


def test_fad_gate_above_threshold():
    out = evalx.fad_gate(15.0, threshold=10.0)
    assert out["passed"] is False


def test_fad_gate_missing():
    out = evalx.fad_gate(None)
    assert out["passed"] is None
    assert out["reason"] == "FAD unavailable"
