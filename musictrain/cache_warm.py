"""HF Hub cache warm-up (#13).

Pre-pulls the model + processor weights so the first generation in CI or a
fresh container doesn't pay the download cost on the critical path. Heavy
deps are imported lazily and every failure degrades to a warning, so this is
safe to call unconditionally (e.g. as a Docker entrypoint or CI step).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import console
from .config import Config
from .logging import get_logger

log = get_logger("cache_warm")


def warm(cfg: Config, model_name: Optional[str] = None) -> dict:
    """Download the model + processor to the local HF cache and report sizes."""
    name = model_name or cfg.inference.model_name
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.warn("huggingface_hub not installed — skipping cache warm-up.")
        return {"model": name, "warmed": False, "note": "huggingface_hub missing"}

    try:
        from .retry import is_transient, retry

        retry(
            snapshot_download,
            name,
            ignore_patterns=["*.msgpack", "*.h5", "*.onnx"],
            retries=3,
            retryable=is_transient,
        )
        console.ok(f"Warmed HF cache for {name}")
    except Exception as exc:  # noqa: BLE001 - offline/rate-limit must not hard-fail
        console.warn(f"HF cache warm-up failed for {name}: {exc}")
        return {"model": name, "warmed": False, "note": str(exc)}

    size = _cached_bytes(name)
    return {"model": name, "warmed": True, "cached_bytes": size}


def _cached_bytes(model_name: str) -> Optional[int]:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return None
    try:
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_name and repo.repo_path is not None:
                return int(sum(f.size_on_disk for f in repo.revisions[0].files))
    except Exception as exc:  # noqa: BLE001
        log.debug("could not measure cache size: %s", exc)
    return None


def warm_dir(root: Path, model_name: str) -> dict:
    """Convenience: build a Config rooted at ``root`` and warm its cache."""
    cfg = Config()
    cfg.project_root = Path(root)
    if model_name:
        cfg.inference.model_name = model_name
    return warm(cfg)
