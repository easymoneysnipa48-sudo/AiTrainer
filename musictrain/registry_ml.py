"""MLflow model registry + stage transitions (Advanced #44).

Registers a checkpoint under ``checkpoints/<name>`` as an MLflow model with
proper versioning, then lets you move versions through stages
(Staging -> Production -> Archived), the same promotion flow the eval gates
already enforce on the metrics side.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from . import console
from .config import Config
from .experiments import _configure

STAGES = ("None", "Staging", "Production", "Archived")


def register_model(cfg: Config, name: str, stage: str = "None") -> Optional[dict]:
    """Log checkpoints/<name> as an MLflow model and return its version info."""
    model_dir = cfg.project_root / "checkpoints" / name
    if not model_dir.is_dir():
        console.error(f"Checkpoint not found: {model_dir}")
        return None
    if stage not in STAGES:
        console.error(f"stage must be one of {STAGES} (got {stage!r})")
        return None

    ml = _configure(cfg)
    if ml is None:
        return None

    try:
        # a model name is an artifact path; use a stable registry name
        registry_name = cfg.mlflow.experiment_name.replace("/", "_") + "-models"
        with ml.start_run(run_name=f"register/{name}"):
            ml.set_tags({"task": "model_registry", "checkpoint": name})
            ml.log_params({"checkpoint": name, "stage": stage})
            ml.log_artifact(str(model_dir), artifact_path="model")
            run_id = ml.active_run().info.run_id

        # create a version from the run's logged model
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        registered = client.create_registered_model(registry_name)
        version = client.create_model_version(
            name=registry_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
            description=f"checkpoint {name}",
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
