"""Tests for advanced batch 1 (#1-#10): metrics, significance, difficulty, adversarial prompts."""
import json

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# metrics — two-sample tests
# --------------------------------------------------------------------------- #
def test_kld_gaussian_zero_for_identical():
    from musictrain.metrics import kld_gaussian

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (50, 6))
    assert kld_gaussian(x, x) < 1e-6


def test_mmd_rbf_small_for_identical():
    from musictrain.metrics import mmd_rbf

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (30, 6))
    assert mmd_rbf(x, x) < 1e-6


def test_one_nn_duplicate_sets_chance():
    from musictrain.metrics import one_nn_acc

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (30, 6))
    assert one_nn_acc(x, x) == 0.5


def test_one_nn_separates_different():
    from musictrain.metrics import one_nn_acc

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (40, 6))
    b = rng.normal(3, 1, (40, 6))
    assert one_nn_acc(a, b) > 0.85


# --------------------------------------------------------------------------- #
# significance — bootstrap, Bayesian, meta-analysis
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_excludes_zero_for_real_diff():
    from musictrain.significance import bootstrap_ci

    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 200)
    b = rng.normal(1.0, 1, 200)
    lo, hi, _ = bootstrap_ci(a, b, n_boot=500)
    assert lo > 0 and hi > 0


def test_bootstrap_ci_includes_zero_for_no_diff():
    from musictrain.significance import bootstrap_ci

    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 100)
    b = rng.normal(0.05, 1, 100)
    lo, hi, _ = bootstrap_ci(a, b, n_boot=500)
    assert lo <= 0 <= hi


def test_bayesian_ab_strong_evidence():
    from musictrain.significance import bayesian_ab

    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 300)
    b = rng.normal(1.5, 1, 300)
    out = bayesian_ab(a, b)
    assert out["p_b_over_a"] > 0.99


def test_meta_analyze_pools_valid_studies():
    from musictrain.significance import meta_analyze

    studies = [
        {"delta": 0.5, "se": 0.2, "label": "s1"},
        {"delta": 0.3, "se": 0.25, "label": "s2"},
        {"delta": 1.0, "se": None},  # excluded
        {"delta": None, "se": 0.1},  # excluded
    ]
    out = meta_analyze(studies)
    assert out["n_pooled"] == 2
    assert out["pooled_delta"] is not None
    assert 0.3 < out["pooled_delta"] < 0.5


def test_compare_includes_bootstrap_and_bayesian(tmp_path):
    from musictrain.config import Config
    from musictrain.significance import compare

    cfg = Config()
    cfg.project_root = tmp_path
    rows_a = [
        {"prompt": f"p{i}", "bpm_target": 90 + i, "clap_score": 0.3 + i * 0.01,
         "deviation": 0.1 + i * 0.01, "status": "ok"}
        for i in range(10)
    ]
    rows_b = [
        {"prompt": f"p{i}", "bpm_target": 90 + i, "clap_score": 0.4 + i * 0.01,
         "deviation": 0.05 + i * 0.01, "status": "ok"}
        for i in range(10)
    ]
    out = compare(cfg, rows_a, rows_b, label_a="A", label_b="B")
    m = out["metrics"]["clap_score"]
    assert "bootstrap_95ci" in m and "bayesian" in m
    assert m["bayesian"]["p_b_over_a"] is not None
    assert m["bootstrap_95ci"][0] > 0  # B clearly better on CLAP


# --------------------------------------------------------------------------- #
# difficulty — prompt difficulty, section x BPM, z-scores, calibration
# --------------------------------------------------------------------------- #
def _fake_rows():
    rows = []
    for i, sec in enumerate(["chorus", "chorus", "bridge", "bridge", "verse"]):
        rows.append(
            {
                "prompt": f"p{i}", "section": sec, "bpm_target": 90 + i,
                "clap_score": 0.5 - i * 0.05, "deviation": 0.02 + i * 0.1,
                "status": "ok" if i < 3 else "rejected",
            }
        )
    return rows


def test_prompt_difficulty_ranks_hardest_first():
    from musictrain.difficulty import prompt_difficulty

    scored = prompt_difficulty(_fake_rows())
    assert scored[0]["difficulty"] >= scored[-1]["difficulty"]
    assert all("prompt" in s for s in scored)


def test_section_bpm_interaction_shape():
    from musictrain.difficulty import section_bpm_interaction

    rows = []
    for sec in ["chorus", "bridge"]:
        for i in range(6):
            rows.append(
                {"prompt": f"{sec}{i}", "section": sec, "bpm_target": 80 + 10 * i,
                 "clap_score": 0.4, "deviation": 0.05, "status": "ok"}
            )
    table = section_bpm_interaction(rows)
    assert len(table) == 2
    assert all(t["ok_rate"] == 1.0 for t in table)
    assert all("bpm_dev_corr" in t for t in table)


def test_clap_zscores_standardized():
    from musictrain.difficulty import clap_zscores

    rows = [
        {"prompt": f"p{i}", "section": "chorus", "clap_score": 0.2 + i * 0.1}
        for i in range(6)
    ]
    zs = clap_zscores(rows)
    vals = [z["clap_z"] for z in zs]
    assert abs(np.mean(vals)) < 1e-6
    assert np.std(vals) > 0


def test_calibrate_thresholds_rejects_some():
    from musictrain.difficulty import calibrate_thresholds

    rows = []
    for i in range(20):
        rows.append(
            {"prompt": f"p{i}", "deviation": 0.01 + (i % 5) * 0.2,
             "clap_score": 0.5 - (i % 4) * 0.1,
             "status": "ok" if i % 5 < 4 else "rejected"}
        )
    cal = calibrate_thresholds(rows)
    assert cal["n_prompts"] == 20
    assert cal["suggested_max_abs_deviation"] is not None


# --------------------------------------------------------------------------- #
# evalset — adversarial prompts
# --------------------------------------------------------------------------- #
def test_adversarial_prompts_flag(tmp_path):
    from musictrain.evalset import build, load

    prompts = build(tmp_path, force=True, adversarial=3)
    adv = [p for p in prompts if p.get("adversarial")]
    assert len(adv) == 3
    assert all(p["id"].startswith("adv_") for p in adv)
    # persisted to disk
    assert len(load(tmp_path)) == len(prompts)


def test_build_no_overwrite_without_force(tmp_path):
    from musictrain.evalset import build

    build(tmp_path, force=True)
    first = build(tmp_path)  # file exists, force=False -> returns existing
    assert first  # not empty, no crash
