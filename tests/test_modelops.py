"""Unit tests for advanced model-ops batch #41-#46, #49-#50."""
from __future__ import annotations

import json

import numpy as np
import pytest

from musictrain import modelops as mo
from musictrain.config import Config


def _cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.project_root = tmp_path
    return cfg


# --------------------------------------------------------------------------- #
# #41 alias mapping
# --------------------------------------------------------------------------- #
def test_alias_mapping():
    versions = [
        {"version": 1, "stage": "Production"},
        {"version": 2, "stage": "Staging"},
        {"version": 3, "stage": "None"},
    ]
    out = mo.alias_mapping(versions)
    assert out == {1: "champion", 2: "challenger"}


def test_alias_mapping_uses_current_stage_key():
    versions = [{"version": 4, "current_stage": "Production"}]
    assert mo.alias_mapping(versions) == {4: "champion"}


# --------------------------------------------------------------------------- #
# #42 A/B win rate
# --------------------------------------------------------------------------- #
def test_ab_win_rate_challenger_better():
    a = [0.4] * 10
    b = [0.6] * 10
    out = mo.ab_win_rate(a, b, higher_is_better=True)
    assert out["wins"] == 10
    assert out["win_rate"] == 1.0
    assert out["decision"] == "promote"


def test_ab_win_rate_tie_holds():
    a = [0.5] * 10
    b = [0.5] * 10
    out = mo.ab_win_rate(a, b)
    assert out["ties"] == 10
    assert out["decision"] == "hold"


def test_ab_win_rate_lower_is_better_flips():
    a = [0.05] * 10   # low deviation = good
    b = [0.20] * 10
    out = mo.ab_win_rate(a, b, higher_is_better=False)
    assert out["wins"] == 0   # challenger is worse
    assert out["decision"] == "hold"


def test_ab_win_rate_empty():
    out = mo.ab_win_rate([], [])
    assert out["n"] == 0
    assert out["win_rate"] is None


# --------------------------------------------------------------------------- #
# #44 lineage
# --------------------------------------------------------------------------- #
def test_lineage_record_and_graph(tmp_path):
    cfg = _cfg(tmp_path)
    mo.record_lineage(cfg, "facebook/musicgen-small", "my-finetune", note="LoRA")
    graph = mo.lineage_graph(cfg)
    assert graph["nodes"] == ["facebook/musicgen-small", "my-finetune"]
    assert graph["edges"][0]["parent"] == "facebook/musicgen-small"


def test_lineage_graph_empty(tmp_path):
    cfg = _cfg(tmp_path)
    assert mo.lineage_graph(cfg) == {"nodes": [], "edges": []}


# --------------------------------------------------------------------------- #
# #45 checksums
# --------------------------------------------------------------------------- #
def test_checksum_verify_roundtrip(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"x": 1}')
    (model_dir / "ignore.txt").write_text("not hashed")

    manifest = mo.checksum_dir(model_dir)
    assert manifest["n_files"] == 1
    assert mo.verify_checksum(model_dir, manifest) is True


def test_checksum_detects_tamper(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    f = model_dir / "model.safetensors"
    f.write_bytes(b"original")
    manifest = mo.checksum_dir(model_dir)
    f.write_bytes(b"tampered")
    assert mo.verify_checksum(model_dir, manifest) is False


# --------------------------------------------------------------------------- #
# #49 cost breakdown
# --------------------------------------------------------------------------- #
def test_cost_breakdown_attributes():
    out = mo.cost_breakdown("musicgen-small", n_prompts=44, n_seeds=3, tokens_per_clip=256)
    assert out["n_clips"] == 132
    assert out["per_prompt_kwh"] > 0
    assert out["per_seed_kwh"] > 0
    assert out["per_prompt_kwh"] > out["per_seed_kwh"]
    # per-prompt cost is ~ n_seeds x per-seed cost (rounded, so ~, not ==)
    assert out["per_prompt_kwh"] == pytest.approx(out["per_seed_kwh"] * 3, rel=0.2)


# --------------------------------------------------------------------------- #
# #50 config lint
# --------------------------------------------------------------------------- #
def test_lint_default_config_clean(tmp_path):
    cfg = _cfg(tmp_path)
    assert mo.lint_config(cfg) == []


def test_lint_catches_bad_values(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.check.bpm_tolerance = 1.5
    cfg.split.train = 0.9  # train+val+test = 1.1
    cfg.inference.temperature = 0.0
    issues = mo.lint_config(cfg)
    fields = {i["field"] for i in issues}
    assert "check.bpm_tolerance" in fields
    assert "split.train+val+test" in fields
    assert "inference.temperature" in fields


def test_lint_writes_report(tmp_path):
    cfg = _cfg(tmp_path)
    report = mo.lint(cfg)
    assert report["valid"] is True
    assert (tmp_path / "metadata" / "config_lint.json").exists()
