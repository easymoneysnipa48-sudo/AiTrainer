"""MLflow experiment tracking for inference, evaluation, and dataset runs.

All tracking is local by default (SQLite at <project_root>/mlflow.db) and can
be disabled via `mlflow.enabled: false` in config, or pointed at a remote
tracking server via `mlflow.tracking_uri`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import console
from .config import Config


def _mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError:
        console.warn("mlflow is not installed; skipping tracking.")
        return None


def _uri(cfg: Config) -> str:
    mcfg = cfg.mlflow
    return mcfg.tracking_uri or f"sqlite:///{cfg.project_root / 'mlflow.db'}"


def _configure(cfg: Config):
    ml = _mlflow()
    if ml is None:
        return None
    ml.set_tracking_uri(_uri(cfg))
    ml.set_experiment(cfg.mlflow.experiment_name)
    return ml


def log_inference(cfg: Config, result: dict, clap_score: Optional[float] = None) -> None:
    if not cfg.mlflow.enabled:
        return
    ml = _configure(cfg)
    if ml is None:
        return
    try:
        with ml.start_run(run_name=f"infer/{result['prompt'][:48]}"):
            ml.set_tags({"task": "inference", "device": result.get("device", "")})
            ml.log_params(
                {
                    "model": cfg.inference.model_name,
                    "guidance_scale": cfg.inference.guidance_scale,
                    "max_new_tokens": cfg.inference.max_new_tokens,
                    "do_sample": cfg.inference.do_sample,
                    "seed": result.get("seed"),
                }
            )
            metrics = {"duration_s": result["duration"], "sample_rate": result["sample_rate"]}
            if clap_score is not None:
                metrics["clap_score"] = clap_score
            ml.log_metrics(metrics)
            ml.log_text(result["prompt"], "prompt.txt")
            ml.log_artifact(result["path"], artifact_path="audio")
        console.info(f"MLflow: logged inference run ({Path(result['path']).name})")
    except Exception as exc:  # noqa: BLE001 - tracking must never break the pipeline
        console.warn(f"MLflow logging failed: {exc}")


def log_eval(cfg: Config, report: dict) -> None:
    if not cfg.mlflow.enabled:
        return
    ml = _configure(cfg)
    if ml is None:
        return
    try:
        with ml.start_run(run_name=f"eval/{Path(report['path']).stem[:40]}"):
            ml.set_tags({"task": "eval", "status": report.get("status", "")})
            metrics = {}
            for key in ("detected_bpm", "target_bpm", "deviation"):
                if report.get(key) is not None:
                    metrics[key] = report[key]
            if metrics:
                ml.log_metrics(metrics)
            ml.log_artifact(report["path"], artifact_path="audio")
            if report.get("fixed_path"):
                ml.log_artifact(report["fixed_path"], artifact_path="audio")
        console.info("MLflow: logged eval run")
    except Exception as exc:  # noqa: BLE001
        console.warn(f"MLflow logging failed: {exc}")


def log_dataset(cfg: Config, records: List[dict]) -> None:
    if not cfg.mlflow.enabled or not records:
        return
    ml = _configure(cfg)
    if ml is None:
        return
    try:
        bpms = [r["bpm"] for r in records if r.get("bpm") is not None]
        durations = [r["duration"] for r in records if r.get("duration")]

        with ml.start_run(run_name=f"dataset/{len(records)}tracks"):
            ml.set_tags({"task": "dataset"})
            metrics = {"n_tracks": len(records)}
            if bpms:
                metrics.update(
                    bpm_mean=round(sum(bpms) / len(bpms), 2),
                    bpm_min=min(bpms),
                    bpm_max=max(bpms),
                )
            if durations:
                metrics.update(
                    duration_mean_s=round(sum(durations) / len(durations), 3),
                    total_duration_s=round(sum(durations), 3),
                )
            ml.log_metrics(metrics)
            manifest = cfg.project_root / "metadata" / "manifest.jsonl"
            if manifest.exists():
                ml.log_artifact(str(manifest), artifact_path="dataset")
        console.info(f"MLflow: logged dataset run ({len(records)} tracks)")
    except Exception as exc:  # noqa: BLE001
        console.warn(f"MLflow logging failed: {exc}")


def search_runs(cfg: Config):
    """Return a normalized DataFrame of MLflow runs for the current experiment.

    Columns: run_id, name, task (inference/eval/dataset), verdict, model,
    device, seed, target_bpm, detected_bpm, deviation, duration_s, n_tracks,
    bpm_mean. Empty DataFrame if there are no runs or MLflow is unavailable.
    """
    import pandas as pd

    ml = _configure(cfg)
    if ml is None:
        return pd.DataFrame()
    exp = ml.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp is None:
        return pd.DataFrame()
    df = ml.search_runs(experiment_ids=[exp.experiment_id])
    if df.empty:
        return pd.DataFrame()

    def col(name):
        return df[name] if name in df.columns else None

    out = pd.DataFrame(
        {
            "run_id": df["run_id"].astype(str).str[:8],
            "name": col("tags.mlflow.runName"),
            "task": col("tags.task"),
            "verdict": col("tags.status"),
            "model": col("params.model"),
            "device": col("tags.device"),
            "seed": col("params.seed"),
            "target_bpm": col("metrics.target_bpm"),
            "detected_bpm": col("metrics.detected_bpm"),
            "deviation": col("metrics.deviation"),
            "duration_s": col("metrics.duration_s"),
            "n_tracks": col("metrics.n_tracks"),
            "bpm_mean": col("metrics.bpm_mean"),
        }
    )
    for c in ("target_bpm", "detected_bpm", "deviation", "duration_s", "n_tracks", "bpm_mean"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def launch_ui(cfg: Config, port: int = 5000) -> int:
    ml = _mlflow()
    if ml is None:
        return 1
    uri = _uri(cfg)
    console.info(f"MLflow UI -> http://localhost:{port}  (store: {uri})")
    return subprocess.run(
        [
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", uri,
            "--port", str(port),
        ]
    ).returncode
