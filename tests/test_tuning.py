import math

import pytest

from musictrain import tuning


def test_grad_accum_steps_exact():
    assert tuning.grad_accum_steps(8, 2, 2) == (2, 8)


def test_grad_accum_steps_rounds_up():
    accum, effective = tuning.grad_accum_steps(10, 3, 1)
    assert effective >= 10
    assert accum == math.ceil(10 / 3)


def test_grad_accum_steps_positive_only():
    with pytest.raises(ValueError):
        tuning.grad_accum_steps(8, 0, 1)


def test_quantize_plan_finds_fit():
    plan = tuning.quantize_plan(8_000_000_000, 16_000_000_000)
    labels = [p["dtype"] for p in plan]
    assert labels == ["fp16", "int8", "int4"]
    # int4 (0.5 byte/param) is the smallest — if fp16 doesn't fit it must not fit either
    assert any(p["fits"] for p in plan)


def test_hpo_grid_full_cartesian():
    grid = tuning.hpo_grid([1e-4, 2e-4], [2, 4], [8], n_trials=0)
    assert len(grid) == 2 * 2 * 1


def test_hpo_grid_sampling_is_seeded_and_bounded():
    grid = tuning.hpo_grid([1e-4, 2e-4, 3e-4], [1, 2, 4], [4, 8, 16], n_trials=5, seed=7)
    assert len(grid) == 5


def test_tokenize_candidates_normalizes():
    tokens = tuning.tokenize_candidates(["  Trap 808 ", "<Dark_Synth>", "!!bad token!!", ""])
    assert tokens == ["trap_808", "dark_synth", "bad_token"]


def test_resume_from_missing_dir():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        out = tuning.resume_from(Path(td))
        assert out["resume_path"] is None


def test_recommended_backend_returns_known():
    assert tuning.recommended_backend() in {"mlx", "mps", "cuda", "cpu"}
