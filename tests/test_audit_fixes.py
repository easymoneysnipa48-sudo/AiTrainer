"""Regression tests for the code-audit fixes.

Covers the backup MLflow-state bug, the merge shard-truncation bug, the
enhanced tuning scaffolds, and the enhanced tokenizer extension.
"""
from __future__ import annotations

import json

import pytest


# --------------------------------------------------------------------------- #
# backup: MLflow state discovery
# --------------------------------------------------------------------------- #
def test_mlflow_files_default_local(tmp_path):
    from musictrain.backup import _mlflow_files
    from musictrain.config import Config

    cfg = Config()
    cfg.project_root = tmp_path
    db = tmp_path / "mlflow.db"
    db.write_bytes(b"x")
    mlruns = tmp_path / "mlruns"
    mlruns.mkdir()
    (mlruns / "meta.yaml").write_text("{}")

    files = _mlflow_files(cfg, tmp_path)
    assert db in files
    assert (mlruns / "meta.yaml") in files


def test_mlflow_files_file_uri(tmp_path):
    from musictrain.backup import _mlflow_files
    from musictrain.config import Config

    cfg = Config()
    cfg.project_root = tmp_path
    cfg.mlflow.tracking_uri = f"file://{tmp_path}/mlart"
    art = tmp_path / "mlart"
    art.mkdir()
    (art / "a.json").write_text("{}")

    files = _mlflow_files(cfg, tmp_path)
    assert (art / "a.json") in files


def test_mlflow_files_sqlite_uri(tmp_path):
    from musictrain.backup import _mlflow_files
    from musictrain.config import Config

    cfg = Config()
    cfg.project_root = tmp_path
    db = tmp_path / "mlflow.db"
    db.write_bytes(b"x")
    cfg.mlflow.tracking_uri = f"sqlite:///{db}"

    assert db in _mlflow_files(cfg, tmp_path)


# --------------------------------------------------------------------------- #
# merge: shard-layout mismatch must be refused, not silently truncated
# --------------------------------------------------------------------------- #
def test_merge_rejects_mismatched_shards(tmp_path):
    from musictrain.merge import merge

    a = tmp_path / "a"
    a.mkdir()
    (a / "model-00001-of-00002.safetensors").write_bytes(b"x")
    (a / "model-00002-of-00002.safetensors").write_bytes(b"x")
    b = tmp_path / "b"
    b.mkdir()
    (b / "model.safetensors").write_bytes(b"x")

    with pytest.raises(ValueError, match="shard"):
        merge([a, b], tmp_path / "out")


# --------------------------------------------------------------------------- #
# tuning scaffolds
# --------------------------------------------------------------------------- #
def test_hpo_search_without_objective_never_runs_trials(tmp_path):
    from musictrain import tuning
    from musictrain.config import Config

    cfg = Config()
    cfg.project_root = tmp_path
    out = tuning.hpo_search(cfg, n_trials=3)
    assert out.get("ran_trials") is not True
    assert "trials" in out or "suggested_grid" in out


def test_extend_tokenizer_persists_json(tmp_path):
    from musictrain import tuning

    out = tuning.extend_tokenizer(
        ["Trap 808", "<Dark_Synth>"], out_path=tmp_path / "tokens.json"
    )
    assert out["added"] == 2
    merged = json.loads((tmp_path / "tokens.json").read_text())
    assert set(merged) == {"trap_808", "dark_synth"}


def test_apply_quantization_rejects_bad_bits():
    from musictrain import tuning

    with pytest.raises(ValueError):
        tuning.apply_quantization("facebook/musicgen-small", bits=3)
