"""Tests for the combined per-genre + FAD quality gate (CI wiring)."""
import json

import pytest


def _write_results(tmp_path, rows):
    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    (meta / "eval_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return meta


def _rows():
    return [
        {"genre": "melodic trap", "clap_score": 0.50, "deviation": 0.01},
        {"genre": "melodic trap", "clap_score": 0.45, "deviation": 0.02},
        {"genre": "ambient", "clap_score": 0.25, "deviation": 0.10},
    ]


def test_quality_gate_passes_good_rows(tmp_path):
    from musictrain.config import Config
    from musictrain.gates import quality_gate

    _write_results(tmp_path, _rows())
    cfg = Config()
    cfg.project_root = tmp_path

    verdict = quality_gate(tmp_path, cfg, fad=5.0)
    assert verdict["passed"] is True
    assert verdict["fad_gate"]["passed"] is True
    assert verdict["genre_gate"]["passed"] is True
    assert (tmp_path / "metadata" / "quality_gate.json").exists()


def test_quality_gate_blocks_bad_genre(tmp_path):
    from musictrain.config import Config
    from musictrain.gates import quality_gate

    rows = _rows() + [{"genre": "ambient", "clap_score": 0.05, "deviation": 0.60}]
    _write_results(tmp_path, rows)
    cfg = Config()
    cfg.project_root = tmp_path

    verdict = quality_gate(tmp_path, cfg, fad=5.0)
    assert verdict["passed"] is False
    assert verdict["genre_gate"]["passed"] is False


def test_quality_gate_blocks_bad_fad(tmp_path):
    from musictrain.config import Config
    from musictrain.gates import quality_gate

    _write_results(tmp_path, _rows())
    cfg = Config()
    cfg.project_root = tmp_path

    verdict = quality_gate(tmp_path, cfg, fad=25.0)
    assert verdict["passed"] is False
    assert verdict["fad_gate"]["passed"] is False


def test_quality_gate_reads_fad_from_metrics(tmp_path):
    from musictrain.config import Config
    from musictrain.gates import quality_gate

    meta = _write_results(tmp_path, _rows())
    (meta / "metrics.json").write_text(json.dumps({"fad_clap": 4.2}))

    cfg = Config()
    cfg.project_root = tmp_path
    verdict = quality_gate(tmp_path, cfg)
    assert verdict["fad_gate"]["fad"] == pytest.approx(4.2)
    assert verdict["passed"] is True


def test_quality_gate_no_results_fails(tmp_path):
    from musictrain.config import Config
    from musictrain.gates import quality_gate

    cfg = Config()
    cfg.project_root = tmp_path
    verdict = quality_gate(tmp_path, cfg)
    assert verdict["passed"] is False
    assert verdict["reason"] == "no eval results"
