"""Tests for advanced batch 5 (#41-#50): server job queue, export, runlog,
alerts, cost, incremental eval, and the CI gate script."""
import json
import time
from pathlib import Path

import numpy as np
import pytest

import soundfile as sf


# --------------------------------------------------------------------------- #
# 41/42 — FastAPI backend job queue
# --------------------------------------------------------------------------- #
def test_job_queue_runs_and_reports(tmp_path):
    from musictrain.server import JobQueue

    q = JobQueue()

    def job(progress=None, cancel=None):
        for i in range(4):
            progress(i + 1, 4)
            time.sleep(0.01)
        return {"ok": True}

    job_id = q.submit(job)
    for _ in range(100):
        if q.status(job_id) == "done":
            break
        time.sleep(0.02)
    assert q.status(job_id) == "done"
    result = q.get(job_id)
    assert result["result"] == {"ok": True}
    assert result["progress"] == 1.0


def test_job_queue_failure():
    from musictrain.server import JobQueue

    q = JobQueue()

    def boom(progress=None, cancel=None):
        raise RuntimeError("kaput")

    job_id = q.submit(boom)
    for _ in range(100):
        if q.status(job_id) in ("done", "failed"):
            break
        time.sleep(0.02)
    assert q.status(job_id) == "failed"
    assert "kaput" in q.get(job_id)["error"]


def test_job_queue_cancel():
    from musictrain.server import JobQueue

    q = JobQueue()
    state = {"cancelled": False}

    def slow(progress=None, cancel=None):
        for i in range(50):
            if cancel and cancel():
                state["cancelled"] = True
                return {"cancelled": True}
            time.sleep(0.02)
        return {"cancelled": False}

    job_id = q.submit(slow)
    time.sleep(0.05)
    assert q.cancel(job_id) is True
    time.sleep(0.2)
    assert state["cancelled"] is True


# --------------------------------------------------------------------------- #
# 46 — structured JSON runlog
# --------------------------------------------------------------------------- #
def test_json_log_roundtrip(tmp_path):
    from musictrain.telemetry import json_log, read_runlog

    json_log(tmp_path, "eval", checkpoint="v1", ok=3)
    json_log(tmp_path, "eval", checkpoint="v2", ok=5)
    json_log(tmp_path, "generate", prompt="p")

    evals = read_runlog(tmp_path, event="eval")
    assert len(evals) == 2
    assert evals[0]["checkpoint"] == "v1"
    all_rows = read_runlog(tmp_path)
    assert len(all_rows) == 3
    assert all_rows[-1]["event"] == "generate"


# --------------------------------------------------------------------------- #
# 45 — export (CSV fallback)
# --------------------------------------------------------------------------- #
def _eval_rows(checkpoint, claps, devs):
    return [
        {"checkpoint": checkpoint, "prompt": f"p{i}", "bpm_target": 96,
         "clap_score": c, "deviation": d, "status": "ok", "section": "chorus"}
        for i, (c, d) in enumerate(zip(claps, devs))
    ]


def test_export_wandb_writes_csv(tmp_path):
    from musictrain.config import Config
    from musictrain.telemetry import export_wandb

    meta = tmp_path / "metadata"
    meta.mkdir()
    rows = _eval_rows("v1", [0.5, 0.6], [0.1, 0.12])
    rows += _eval_rows("v2", [0.4, 0.5], [0.2, 0.2])
    (meta / "eval_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )

    cfg = Config()
    cfg.project_root = tmp_path
    out = export_wandb(cfg)
    assert out.exists() and out.name == "eval_wandb.csv"
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 checkpoints
    assert "v1" in lines[1]


# --------------------------------------------------------------------------- #
# 47 — alerts
# --------------------------------------------------------------------------- #
def test_check_alerts_finds_violations(tmp_path):
    from musictrain.config import Config
    from musictrain.alerts import check_alerts

    meta = tmp_path / "metadata"
    meta.mkdir()
    rows = _eval_rows("weak", [0.1, 0.2], [0.5, 0.6])
    (meta / "eval_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    cfg = Config()
    cfg.project_root = tmp_path
    violations = check_alerts(cfg, min_clap=0.3, max_abs_deviation=0.2)
    metrics = {v["metric"] for v in violations}
    assert "mean_clap" in metrics
    assert "mean_abs_deviation" in metrics


def test_alert_writes_file(tmp_path):
    from musictrain.config import Config
    from musictrain.alerts import alert

    meta = tmp_path / "metadata"
    meta.mkdir()
    rows = _eval_rows("weak", [0.1, 0.2], [0.5, 0.6])
    (meta / "eval_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    cfg = Config()
    cfg.project_root = tmp_path
    result = alert(cfg, min_clap=0.3)
    assert result["fired"] is True
    assert "file" in result["channels"]
    assert (meta / "alerts.jsonl").exists()


def test_alert_no_violations(tmp_path):
    from musictrain.config import Config
    from musictrain.alerts import alert

    meta = tmp_path / "metadata"
    meta.mkdir()
    rows = _eval_rows("strong", [0.6, 0.7], [0.05, 0.08])
    (meta / "eval_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    cfg = Config()
    cfg.project_root = tmp_path
    assert alert(cfg, min_clap=0.3)["fired"] is False


# --------------------------------------------------------------------------- #
# 48 — cost tracking
# --------------------------------------------------------------------------- #
def test_cost_estimate_scales_with_model():
    from musictrain.cost import estimate

    small = estimate("musicgen-small", 10)
    large = estimate("musicgen-large", 10)
    assert large["total_flops"] > small["total_flops"]
    assert small["n_clips"] == 10
    assert small["estimated_joules"] > 0


def test_cost_log_and_summary(tmp_path):
    from musictrain.config import Config
    from musictrain.cost import log_cost, cost_summary

    cfg = Config()
    cfg.project_root = tmp_path
    log_cost(cfg, "eval", "musicgen-small", 44)
    log_cost(cfg, "finetune", "musicgen-medium", 100, n_epochs=2)
    summary = cost_summary(cfg)
    assert summary["runs"] == 2
    assert summary["total_kwh"] > 0
    assert set(summary["by_task"]) == {"eval", "finetune"}


# --------------------------------------------------------------------------- #
# 44 — MLflow registry: HF model-id resolution from the local cache
# --------------------------------------------------------------------------- #
def test_resolve_model_dir_hf_cache(tmp_path, monkeypatch):
    from musictrain.config import Config
    from musictrain.registry_ml import _resolve_model_dir

    # fake HF cache: models--facebook--musicgen-small/snapshots/<hash>/
    cache = tmp_path / ".cache" / "huggingface" / "hub"
    snap = cache / "models--facebook--musicgen-small" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"w")

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    cfg = Config()
    cfg.project_root = tmp_path
    resolved = _resolve_model_dir(cfg, "facebook/musicgen-small")
    assert resolved is not None and resolved == snap

    # checkpoint dir under checkpoints/ takes priority
    ckpt = tmp_path / "checkpoints" / "my-tuned"
    ckpt.mkdir(parents=True)
    assert _resolve_model_dir(cfg, "my-tuned") == ckpt

    # absolute path works too
    assert _resolve_model_dir(cfg, str(snap)) == snap

    # unknown -> None
    assert _resolve_model_dir(cfg, "nope/does-not-exist") is None


# --------------------------------------------------------------------------- #
# 49 — incremental eval
# --------------------------------------------------------------------------- #
def test_run_eval_incremental_keeps_passed(tmp_path, monkeypatch):
    from musictrain.config import Config
    import musictrain.evalset as ev

    meta = tmp_path / "metadata"
    meta.mkdir()
    prompts = [
        {"id": "p1", "seed": 0, "bpm": 96, "section": "chorus",
         "description": "heavy 808 chorus 96 BPM"},
        {"id": "p2", "seed": 1, "bpm": 96, "section": "chorus",
         "description": "heavy 808 chorus 96 BPM alt"},
    ]
    (meta / "eval_prompts.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in prompts)
    )

    cfg = Config()
    cfg.project_root = tmp_path
    cfg.clap.enabled = False
    cfg.mlflow.enabled = False

    import musictrain.evaluate as evaluate_mod
    import musictrain.inference as inference_mod

    monkeypatch.setattr(inference_mod, "load_model", lambda ics: (None, None, "cpu"))
    calls = {"n": 0}

    def fake_generate(cfg, prompt, out_dir=None, name=None, seed=0, **kw):
        calls["n"] += 1
        wav = Path(out_dir) / f"{name}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav), np.zeros(16000, dtype=np.float32), 32000)
        return {"path": str(wav), "duration": 5.0, "sample_rate": 32000,
                "prompt": prompt, "seed": seed, "device": "cpu"}

    def fake_check(cfg, path, target_bpm=None, **kw):
        return {"detected_bpm": 96.0, "deviation": 0.0, "status": "ok", "note": ""}

    monkeypatch.setattr(inference_mod, "generate", fake_generate)
    monkeypatch.setattr(evaluate_mod, "check", fake_check)

    results = ev.run_eval(cfg, check_bpm=True)
    assert len(results) == 2 and calls["n"] == 2

    # add a third prompt; incremental must keep the 2 passed rows and run only p3
    (meta / "eval_prompts.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in prompts + [
            {"id": "p3", "seed": 2, "bpm": 96, "section": "chorus",
             "description": "new prompt"}
        ])
    )
    calls["n"] = 0
    results2 = ev.run_eval(cfg, check_bpm=True, incremental=True)
    assert calls["n"] == 1  # only the new prompt regenerated
    assert len(results2) == 1
    file_rows = [json.loads(ln) for ln in (meta / "eval_results.jsonl").read_text().splitlines() if ln.strip()]
    assert len(file_rows) == 3  # 2 kept + 1 new


# --------------------------------------------------------------------------- #
# 50 — CI gate script
# --------------------------------------------------------------------------- #
def test_ci_gate_script_behavior(tmp_path):
    """The fixture gate script must exit 0 (blocked regression + passing cand)."""
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "run_gate.py"
    assert script.exists(), "CI gate script missing"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "gate OK" in res.stdout
