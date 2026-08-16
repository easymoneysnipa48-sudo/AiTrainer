"""MLflow model registry + stage transitions (Advanced #44).

Registers a checkpoint under ``checkpoints/<name>`` as an MLflow model with
proper versioning, then lets you move versions through stages
(Staging -> Production -> Archived), the same promotion flow the eval gates
already enforce on the metrics side.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from . import console
from .config import Config
from .experiments import _configure

STAGES = ("None", "Staging", "Production", "Archived")


def _resolve_model_dir(cfg: Config, name: str) -> Optional[Path]:
    """Resolve a checkpoint name to a local model directory.

    Order: checkpoints/<name> (fine-tuned), an absolute path, or a HuggingFace
    model id resolved from the local cache (offline-safe).
    """
    cand = cfg.project_root / "checkpoints" / name
    if cand.is_dir():
        return cand
    p = Path(name)
    if p.is_dir():
        return p
    if "/" in name:  # HF model id like facebook/musicgen-small
        # prefer the raw cache snapshot — it may be a partial download (only
        # the safetensors variant) but still a fully loadable model dir
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        folder = cache / f"models--{name.replace('/', '--')}" / "snapshots"
        if folder.is_dir():
            snaps = sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snaps:
                return snaps[0]
        try:
            from huggingface_hub import snapshot_download

            resolved = snapshot_download(name, local_files_only=True)
            return Path(resolved)
        except Exception as exc:  # noqa: BLE001
            console.warn(f"Model {name!r} not in the local HF cache: {exc}")
    return None


def _eval_summary(cfg: Config, checkpoint: str) -> Optional[dict]:
    """Aggregate current eval results for a checkpoint (for registry metadata)."""
    from .report import load_results

    rows = [r for r in load_results(cfg.project_root)
            if (r.get("checkpoint") or "").strip() == checkpoint]
    if not rows:
        return None
    claps = [r["clap_score"] for r in rows if r.get("clap_score") is not None]
    devs = [abs(r["deviation"]) for r in rows if r.get("deviation") is not None]
    ok = sum(1 for r in rows if r.get("status") == "ok")
    ok_pct = ok / len(rows)
    mean_clap = (sum(claps) / len(claps)) if claps else None
    mean_abs_dev = (sum(devs) / len(devs)) if devs else None
    fidelity = max(0.0, 1.0 - (mean_abs_dev / 0.20)) if mean_abs_dev is not None else 0.0
    score = round(0.4 * ok_pct + 0.3 * (mean_clap or 0.0) + 0.3 * fidelity, 4)
    return {
        "n_rows": len(rows),
        "ok_pct": round(ok_pct, 4),
        "ok_count": ok,
        "mean_clap": round(mean_clap, 4) if mean_clap is not None else None,
        "mean_abs_deviation": round(mean_abs_dev, 4) if mean_abs_dev is not None else None,
        "leaderboard_score": score,
    }


def _describe(checkpoint: str, summary: Optional[dict]) -> str:
    if not summary:
        return f"checkpoint {checkpoint}"
    return (
        f"checkpoint {checkpoint} · eval: ok {summary['ok_pct']:.0%} "
        f"({summary['ok_count']}/{summary['n_rows']}), "
        f"mean CLAP {summary['mean_clap']}, "
        f"mean |dev| {summary['mean_abs_deviation']}, "
        f"leaderboard score {summary['leaderboard_score']}"
    )


def register_model(cfg: Config, name: str, stage: str = "None",
                   update: bool = False) -> Optional[dict]:
    """Log a checkpoint (dir under checkpoints/, absolute path, or HF model id)
    as an MLflow model and return its version info.

    With ``update=True``, refresh the latest registered version's description
    (and stage) with the current eval summary instead of creating a new
    version — right for when the weights are unchanged but the measurement
    improved (e.g. a BPM-eval fix).
    """
    model_dir = _resolve_model_dir(cfg, name)
    if model_dir is None:
        console.error(f"Checkpoint not found: {cfg.project_root / 'checkpoints' / name}")
        return None
    if stage not in STAGES:
        console.error(f"stage must be one of {STAGES} (got {stage!r})")
        return None

    ml = _configure(cfg)
    if ml is None:
        return None

    try:
        from mlflow.tracking import MlflowClient

        registry_name = cfg.mlflow.experiment_name.replace("/", "_") + "-models"
        client = MlflowClient()
        summary = _eval_summary(cfg, name)
        description = _describe(name, summary)

        try:
            existing = client.get_registered_model(registry_name)
        except Exception:  # noqa: BLE001 - not registered yet
            existing = None

        # -- update path: refresh the latest version, no new artifact copy -----
        if update and existing is not None:
            versions = client.get_latest_versions(registry_name)
            if not versions:
                console.error(f"No versions of {registry_name} to update.")
                return None
            v = versions[0]
            client.update_model_version(
                registry_name, v.version, description=description
            )
            if stage != "None":
                client.transition_model_version_stage(registry_name, v.version, stage)
            if summary and v.run_id:
                for key, val in (
                    ("eval_ok_pct", summary["ok_pct"]),
                    ("eval_mean_clap", summary["mean_clap"]),
                    ("eval_mean_abs_deviation", summary["mean_abs_deviation"]),
                    ("eval_leaderboard_score", summary["leaderboard_score"]),
                ):
                    if val is not None:
                        client.log_metric(v.run_id, key, val)
            out = {
                "registry_name": registry_name,
                "version": v.version,
                "stage": stage if stage != "None" else v.current_stage,
                "run_id": v.run_id,
                "updated": True,
            }
            ok_pct = f"{summary['ok_pct']:.0%}" if summary else "—"
            console.ok(
                f"Updated {registry_name} v{v.version} [{out['stage']}] with "
                f"eval summary (ok {ok_pct})"
            )
            return out

        # -- fresh registration ------------------------------------------------
        with ml.start_run(run_name=f"register/{name}"):
            ml.set_tags({"task": "model_registry", "checkpoint": name})
            ml.log_params({"checkpoint": name, "stage": stage})
            if summary:
                ml.log_metrics(
                    {
                        "eval_ok_pct": summary["ok_pct"],
                        "eval_mean_clap": summary["mean_clap"],
                        "eval_mean_abs_deviation": summary["mean_abs_deviation"],
                        "eval_leaderboard_score": summary["leaderboard_score"],
                    }
                )
            ml.log_artifact(str(model_dir), artifact_path="model")
            run_id = ml.active_run().info.run_id

        if existing is None:
            client.create_registered_model(registry_name)
        version = client.create_model_version(
            name=registry_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
            description=description,
        )
        client.transition_model_version_stage(
            name=registry_name,
            version=version.version,
            stage=stage,
        )
        out = {
            "registry_name": registry_name,
            "version": version.version,
            "stage": stage,
            "run_id": run_id,
        }
        console.ok(
            f"Registered {name} as {registry_name} v{version.version} "
            f"[{stage}] (run {run_id[:8]})"
        )
        return out
    except Exception as exc:  # noqa: BLE001
        console.warn(f"MLflow registry failed (is the server reachable?): {exc}")
        return None


def list_models(cfg: Config) -> List[dict]:
    """List registered models with all versions + stages."""
    from mlflow.tracking import MlflowClient

    ml = _configure(cfg)
    if ml is None:
        return []
    try:
        client = MlflowClient()
        out: List[dict] = []
        for rm in client.search_registered_models():
            versions = []
            for v in rm.latest_versions:
                versions.append(
                    {"version": v.version, "stage": v.current_stage,
                     "run_id": v.run_id, "status": v.status}
                )
            out.append({"name": rm.name, "versions": versions})
        return out
    except Exception as exc:  # noqa: BLE001
        console.warn(f"Could not list models: {exc}")
        return []


def transition(cfg: Config, registry_name: str, version: int, stage: str) -> bool:
    """Move a model version to a stage (Staging/Production/Archived)."""
    if stage not in STAGES:
        console.error(f"stage must be one of {STAGES} (got {stage!r})")
        return False
    ml = _configure(cfg)
    if ml is None:
        return False
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        client.transition_model_version_stage(registry_name, version, stage)
        console.ok(f"{registry_name} v{version} -> {stage}")
        return True
    except Exception as exc:  # noqa: BLE001
        console.warn(f"Transition failed: {exc}")
        return False
