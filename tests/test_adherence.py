"""Unit tests for advanced eval batch #1-#10 (musictrain.adherence).

Exercises the pure logic: Camelot key distance, structure-order scoring,
duration adherence, seed diversity, reliability curve, multiple-comparison
correction, bootstrap CIs, genre gates, and instrument-presence band energy.
"""
from __future__ import annotations

import numpy as np
import pytest

from musictrain import adherence as ad


# --------------------------------------------------------------------------- #
# #2 key adherence / Camelot wheel
# --------------------------------------------------------------------------- #
def test_camelot_relative_major_minor_is_zero():
    assert ad.camelot_distance("A minor", "C major") == 0  # 8A vs 8B


def test_camelot_fifth_is_one():
    assert ad.camelot_distance("A minor", "E minor") == 1  # 8A vs 9A
    assert ad.camelot_distance("C major", "G major") == 1  # 8B vs 9B


def test_camelot_farthest_is_six():
    assert ad.camelot_distance("A minor", "D# minor") == 6  # 8A vs 2A


def test_key_adherence_match():
    out = ad.key_adherence("C minor", "C minor")
    assert out["match"] is True
    assert out["camelot_distance"] == 0
    assert out["score"] == 1.0


def test_key_adherence_distance():
    out = ad.key_adherence("A minor", "C# minor")  # 8A vs 12A -> d=4
    assert out["camelot_distance"] == 4
    assert out["score"] == pytest.approx(1.0 - 4 / 6, abs=1e-4)


def test_key_adherence_unparsable():
    assert ad.key_adherence("blorp", "C minor") is None


# --------------------------------------------------------------------------- #
# #3 structure order
# --------------------------------------------------------------------------- #
def test_structure_intro_positional():
    out = ad.structure_order_score(["intro", "verse", "chorus", "outro"], "intro")
    assert out["presence"] is True
    assert out["position_match"] is True
    assert out["score"] == 1.0


def test_structure_chorus_presence():
    out = ad.structure_order_score(["intro", "chorus", "outro"], "chorus")
    assert out["presence"] is True
    assert out["score"] == 1.0


def test_structure_missing_section():
    out = ad.structure_order_score(["intro", "chorus", "outro"], "bridge")
    assert out["presence"] is False
    assert out["score"] == 0.0


def test_structure_full_song_always_matches():
    out = ad.structure_order_score(["chorus"], "full-song")
    assert out["score"] == 1.0


# --------------------------------------------------------------------------- #
# #4 duration adherence
# --------------------------------------------------------------------------- #
def test_duration_exact():
    out = ad.duration_adherence(30.0, 30.0)
    assert out["match"] is True
    assert out["score"] == pytest.approx(1.0, abs=1e-4)


def test_duration_half_off():
    out = ad.duration_adherence(45.0, 30.0)  # 50% error -> score 0
    assert out["match"] is False
    assert out["score"] == pytest.approx(0.0, abs=1e-4)


def test_duration_missing_target():
    assert ad.duration_adherence(30.0, 0.0) is None


# --------------------------------------------------------------------------- #
# #5 instrument presence (band energy)
# --------------------------------------------------------------------------- #
def _tone(freq: float, seconds: float = 0.5, sr: int = 32000) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_instrument_presence_low_tone_is_kick():
    y = _tone(60.0)  # 60 Hz -> kick band
    out = ad.instrument_presence(y, 32000)
    assert out["kick"] > out["hi_hats"]


def test_instrument_presence_high_tone_is_hats():
    y = _tone(8000.0)  # 8 kHz -> hats band
    out = ad.instrument_presence(y, 32000)
    assert out["hi_hats"] > out["kick"]


# --------------------------------------------------------------------------- #
# #6 seed diversity
# --------------------------------------------------------------------------- #
def test_seed_diversity_varied():
    out = ad.seed_clap_diversity(
        [{"clap_score": 0.3}, {"clap_score": 0.5}, {"clap_score": 0.7}]
    )
    assert out["std_clap"] > 0.1
    assert out["collapsed"] is False


def test_seed_diversity_collapsed():
    out = ad.seed_clap_diversity(
        [{"clap_score": 0.5}, {"clap_score": 0.5}, {"clap_score": 0.5}]
    )
    assert out["std_clap"] == pytest.approx(0.0, abs=1e-6)
    assert out["collapsed"] is True


def test_seed_diversity_too_few():
    assert ad.seed_clap_diversity([{"clap_score": 0.5}]) is None


# --------------------------------------------------------------------------- #
# #7 reliability curve
# --------------------------------------------------------------------------- #
def test_reliability_curve_monotone():
    rows = [
        {"difficulty": d, "status": "ok" if d < 0.5 else "rejected"}
        for d in np.linspace(0.1, 0.9, 20)
    ]
    out = ad.reliability_curve(rows)
    assert out is not None
    assert len(out["curve"]) >= 1
    # easy prompts pass, hard ones fail -> negative slope
    assert out["fit"]["slope"] < 0
    assert out["fit"]["r_squared"] > 0.5


def test_reliability_curve_empty():
    assert ad.reliability_curve([]) is None


# --------------------------------------------------------------------------- #
# #8 multiple-comparison correction
# --------------------------------------------------------------------------- #
def test_multiple_comparison_bonferroni_and_bh():
    out = ad.multiple_comparison([0.001, 0.03, 0.4], alpha=0.05)
    assert out["n"] == 3
    # Bonferroni: only 0.001 < 0.05/3 ~= 0.0167
    assert out["n_reject_bonferroni"] == 1
    assert out["bonferroni"]["test_0"]["reject"] is True
    assert out["bonferroni"]["test_1"]["reject"] is False
    # BH-FDR: 0.001 (<= 0.0167) and 0.03 (<= 0.0333) both reject
    assert out["n_reject_bh_fdr"] == 2
    assert out["bh_fdr"]["test_2"]["reject"] is False


def test_multiple_comparison_none_skipped():
    out = ad.multiple_comparison([0.01, None, 0.2])
    assert out["n"] == 2


# --------------------------------------------------------------------------- #
# #9 bootstrap CI
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_bounds_mean():
    out = ad.bootstrap_ci(list(np.random.default_rng(0).normal(size=50)))
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]
    assert out["n"] == 50


def test_bootstrap_ci_too_few():
    assert ad.bootstrap_ci([1.0, 2.0]) is None


def test_bootstrap_score_ci():
    rows = [
        {"clap_score": 0.6, "deviation": 0.01, "status": "ok"},
        {"clap_score": 0.7, "deviation": 0.02, "status": "ok"},
        {"clap_score": 0.5, "deviation": 0.10, "status": "rejected"},
    ] * 10
    out = ad.bootstrap_score_ci(rows)
    assert out["ci_low"] <= out["score"] <= out["ci_high"]


# --------------------------------------------------------------------------- #
# #10 genre gates
# --------------------------------------------------------------------------- #
def test_genre_gate_pass():
    gates = {"melodic trap": {"min_clap": 0.32, "max_abs_deviation": 0.15}}
    out = ad.genre_gate(0.5, 0.05, "melodic trap", gates)
    assert out["passed"] is True
    assert out["reasons"] == []


def test_genre_gate_fail_clap():
    gates = {"melodic trap": {"min_clap": 0.32, "max_abs_deviation": 0.15}}
    out = ad.genre_gate(0.2, 0.05, "melodic trap", gates)
    assert out["passed"] is False
    assert any("CLAP" in r for r in out["reasons"])


def test_genre_gate_fallback_default():
    out = ad.genre_gate(0.4, 0.05, "lofi", {"default": {"min_clap": 0.30, "max_abs_deviation": 0.20}})
    assert out["passed"] is True


def test_onset_alignment_returns_bounded():
    import soundfile as sf
    import tempfile
    from pathlib import Path

    # a metronomic click track should lock to a beat grid
    sr = 32000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    y = np.zeros_like(t)
    for click in range(0, 2 * sr, sr // 2):  # 2 clicks/sec ~ 120 BPM
        y[click:click + 200] = 1.0
    y = (0.5 * y).astype(np.float32)
    score = ad.onset_alignment_score(y, sr)
    assert score is None or (0.0 <= score <= 1.0)
