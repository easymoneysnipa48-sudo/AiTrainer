"""Tests for advanced batch 4 (#31-#40): registry, weight diff, archive,
eval gates, drift detector, promotion reports, early stopping, monitor,
experiment matrix, model card."""
import json
from pathlib import Path

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# 31 — checkpoint registry
# --------------------------------------------------------------------------- #
def test_registry_scans_checkpoints(tmp_path):
    from musictrain.config import Config
    from musictrain.registry import scan_registry

    (tmp_path / "checkpoints" / "small").mkdir(parents=True)
    (tmp_path / "checkpoints" / "small" / "config.json").write_text(
        json.dumps({"model_type": "musicgen", "hidden_size": 1024, "num_hidden_layers": 24})
    )
    (tmp_path / "checkpoints" / "small" / "model.safetensors").write_bytes(b"x" * 2_000_000)

    cfg = Config()
    cfg.project_root = tmp_path
    report = scan_registry(tmp_path, cfg)
    assert report["n_checkpoints"] == 1
    entry = report["checkpoints"][0]
    assert entry["name"] == "small"
    assert entry["weight_files"] == 1
    assert entry["size_mb"] > 0
    assert entry["n_params_est"] == 1024 * 1024 * 4 * 24
    assert (tmp_path / "metadata" / "checkpoint_registry.json").exists()


def test_registry_empty(tmp_path):
    from musictrain.config import Config
    from musictrain.registry import scan_registry

    cfg = Config()
    cfg.project_root = tmp_path
    report = scan_registry(tmp_path, cfg)
    assert report["n_checkpoints"] == 0


# --------------------------------------------------------------------------- #
# 34 — weight diff
# --------------------------------------------------------------------------- #
def test_diff_weights_reports_deltas(tmp_path):
    from musictrain.registry import diff_weights

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    import torch

    torch.save({"w": torch.tensor([1.0, 1.0])}, a / "pytorch_model.bin")
    torch.save({"w": torch.tensor([1.0, 5.0])}, b / "pytorch_model.bin")
    report = diff_weights(a, b)
    assert report["tensors_compared"] == 1
    assert report["largest_deltas"][0]["tensor"] == "w"
    assert report["largest_deltas"][0]["max_abs_delta"] == pytest.approx(4.0)


def test_diff_weights_no_shared(tmp_path):
    from musictrain.registry import diff_weights

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "model.safetensors").write_bytes(b"")
    (b / "model.safetensors").write_bytes(b"")
    assert diff_weights(a, b) == {}


# --------------------------------------------------------------------------- #
# 38 — archive
# --------------------------------------------------------------------------- #
def test_archive_bundles_checkpoint(tmp_path):
    from musictrain.config import Config
    from musictrain.registry import archive

    ckpt = tmp_path / "checkpoints" / "v1"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"weights")
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "eval_results.jsonl").write_text("{}\n")

    cfg = Config()
    cfg.project_root = tmp_path
    out = archive(tmp_path, cfg, "v1")
    assert out is not None and out.exists()
    import zipfile

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert any("model.safetensors" in n for n in names)
        assert any("eval_results.jsonl" in n for n in names)


# --------------------------------------------------------------------------- #
# 32 — eval gates
# --------------------------------------------------------------------------- #
def _eval_rows(checkpoint, claps, devs, statuses=None):
    statuses = statuses or ["ok"] * len(claps)
    return [
        {"checkpoint": checkpoint, "prompt": f"p{i} {bpm}",
         "bpm_target": bpm, "clap_score": c, "deviation": d, "status": s,
         "section": "chorus"}
        for i, (c, d, s) in enumerate(zip(claps, devs, statuses))
        for bpm in [96]
    ]


def test_gate_blocks_regression(tmp_path):
    from musictrain.config import Config
    from musictrain.gates import eval_gate

    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    rows = _eval_rows("base", [0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
    rows += _eval_rows("bad", [0.3, 0.3, 0.3], [0.2, 0.2, 0.2])
    (meta / "eval_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    cfg = Config()
    cfg.project_root = tmp_path
    report = eval_gate(tmp_path, cfg, "base", "bad", max_clap_drop=0.02)
    assert report["passed"] is False
    assert any(c["check"] == "clap_drop" and c["blocking"] for c in report["checks"])


def test_gate_passes_within_tolerance(tmp_path):
    from musictrain.config import Config
    from musictrain.gates import eval_gate

    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    rows = _eval_rows("base", [0.5, 0.5], [0.1, 0.1])
    rows += _eval_rows("cand", [0.49, 0.49], [0.11, 0.11])
    (meta / "eval_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    cfg = Config()
    cfg.project_root = tmp_path
    report = eval_gate(tmp_path, cfg, "base", "cand", max_clap_drop=0.02,
                       max_deviation_increase=0.05)
    assert report["passed"] is True


# --------------------------------------------------------------------------- #
# 36 — drift detector
# --------------------------------------------------------------------------- #
def test_drift_detector_detects(tmp_path):
    from musictrain.config import Config
    from musictrain.gates import drift_detector

    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    ref = [{"path": f"data/clean/s{i}.wav", "bpm": 96.0, "loudness": -14.0,
            "duration": 30.0, "key_confidence": 0.8, "key": "A minor"}
           for i in range(25)]
    cur = [{"path": f"data/train/s{i}.wav", "bpm": 170.0, "loudness": -14.0,
            "duration": 30.0, "key_confidence": 0.8, "key": "A minor"}
           for i in range(25)]
    (meta / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ref + cur)
    )

    cfg = Config()
    cfg.project_root = tmp_path
    report = drift_detector(tmp_path, cfg, reference="clean", current="train")
    assert report["passed"] is False
    assert "bpm" in report["drifted_features"]


# --------------------------------------------------------------------------- #
# 37 — promotion report
# --------------------------------------------------------------------------- #
def test_promotion_report_renders(tmp_path, monkeypatch):
    from musictrain.config import Config
    from musictrain.gates import promotion_report

    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    rows = _eval_rows("v1", [0.5, 0.6], [0.1, 0.12])
    (meta / "eval_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    cfg = Config()
    cfg.project_root = tmp_path
    out = promotion_report(tmp_path, cfg, "v1")
    assert out is not None and out.exists()
    text = out.read_text()
    assert "Model card" not in text and "Promotion report" in text
    assert "v1" in text and "Leaderboard rank" in text


# --------------------------------------------------------------------------- #
# 33 — early stopping
# --------------------------------------------------------------------------- #
def test_early_stop_plateau():
    from musictrain.monitor import early_stop

    flat = early_stop([0.30, 0.31, 0.31, 0.31, 0.31], patience=3)
    assert flat["should_stop"] is True

    climbing = early_stop([0.30, 0.31, 0.33, 0.35], patience=3)
    assert climbing["should_stop"] is False

    empty = early_stop([])
    assert empty["should_stop"] is False


# --------------------------------------------------------------------------- #
# 35 — experiment matrix
# --------------------------------------------------------------------------- #
def test_experiment_matrix_empty(tmp_path):
    from musictrain.config import Config
    from musictrain.monitor import experiment_matrix

    cfg = Config()
    cfg.project_root = tmp_path
    cfg.mlflow.enabled = False
    report = experiment_matrix(cfg)
    assert report["runs"] == 0


# --------------------------------------------------------------------------- #
# 40 — model card
# --------------------------------------------------------------------------- #
def test_model_card_renders(tmp_path):
    from musictrain.config import Config
    from musictrain.monitor import model_card

    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    rows = _eval_rows("v1", [0.5, 0.6], [0.1, 0.12])
    (meta / "eval_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    cfg = Config()
    cfg.project_root = tmp_path
    out = model_card(cfg, checkpoint="v1")
    assert out is not None and out.exists()
    text = out.read_text()
    assert "Model card" in text and "v1" in text
    assert "Adherence" in text and "Section coverage" in text
