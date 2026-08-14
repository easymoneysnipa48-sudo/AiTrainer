"""Checkpoint registry + weight diff + eval snapshot archive (Advanced #31/#34/#38).

* **``registry``** — indexes every checkpoint under ``checkpoints/`` with its
  config signature, file count/size, and last-modified time, so you know what
  you have before deciding what to train or promote. Writes
  ``metadata/checkpoint_registry.json``.
* **``diff_weights``** — compares two checkpoints shard-by-shard and reports
  per-tensor max-abs delta (how much did fine-tuning actually move the model?).
* **``archive``** — bundles a checkpoint + its config + current eval report into
  a timestamped zip for reproducible promotion.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import console
from .config import Config


def _weight_files(model_dir: Path) -> List[Path]:
    files = sorted(model_dir.rglob("*.safetensors"))
    if not files:
        files = sorted(model_dir.rglob("*.bin"))
    return files


def scan_registry(root: Path, cfg: Config) -> Dict[str, object]:
    ckpts_dir = root / "checkpoints"
    if not ckpts_dir.exists():
        console.warn(f"No checkpoints/ directory at {root} — nothing to register.")
        return {"n_checkpoints": 0, "checkpoints": []}

    entries: List[dict] = []
    for model_dir in sorted(p for p in ckpts_dir.iterdir() if p.is_dir()):
        try:
            st = model_dir.stat()
        except OSError:
            continue
        config_file = model_dir / "config.json"
        cfg_sig = None
        n_params = None
        if config_file.exists():
            try:
                cfg_json = json.loads(config_file.read_text())
                # musicgen configs carry the transformer dims; sum a rough param count
                hidden = cfg_json.get("hidden_size")
                layers = cfg_json.get("num_hidden_layers")
                if hidden and layers:
                    n_params = int(hidden * hidden * 4 * layers)
                cfg_sig = {
                    k: cfg_json.get(k)
                    for k in ("model_type", "hidden_size", "num_hidden_layers",
                              "num_attention_heads", "vocab_size")
                }
            except Exception:  # noqa: BLE001
                cfg_sig = None

        weights = _weight_files(model_dir)
        total_bytes = sum(p.stat().st_size for p in weights)
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()

        entries.append(
            {
                "name": model_dir.name,
                "path": str(model_dir.relative_to(root)),
                "weight_files": len(weights),
                "size_mb": round(total_bytes / 1e6, 1),
                "n_params_est": n_params,
                "config": cfg_sig,
                "modified_at": mtime,
            }
        )

    entries.sort(key=lambda e: e["modified_at"], reverse=True)
    report = {
        "n_checkpoints": len(entries),
        "checkpoints": entries,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    out = root / "metadata" / "checkpoint_registry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    console.ok(f"Registered {len(entries)} checkpoint(s) -> metadata/checkpoint_registry.json")
    for e in entries:
        console.info(
            f"  {e['name']:<36} {e['size_mb']:>7.1f} MB  {e['weight_files']} file(s)  "
            f"{e['modified_at'][:19]}"
        )
    return report


def diff_weights(a: Path, b: Path, top_k: int = 10) -> Dict[str, object]:
    """Compare two checkpoint dirs, reporting per-tensor max-abs delta."""
    a_files = _weight_files(a)
    b_files = _weight_files(b)
    if not a_files or not b_files:
        console.error("Both checkpoints need weight files (.safetensors or .bin).")
        return {}

    from safetensors import safe_open

    def _load_map(files: List[Path]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for f in files:
            if f.suffix == ".safetensors":
                with safe_open(str(f), framework="np") as sf:
                    for k in sf.keys():
                        out[k] = sf.get_tensor(k)
            else:
                import torch

                sd = torch.load(str(f), map_location="cpu", weights_only=False)
                for k, v in sd.items():
                    if isinstance(v, torch.Tensor):
                        out[k] = v.detach().cpu().numpy()
        return out

    try:
        ma, mb = _load_map(a_files), _load_map(b_files)
    except Exception as exc:  # noqa: BLE001
        console.error(f"Could not read weights: {exc}")
        return {}

    keys = sorted(set(ma) & set(mb))
    if not keys:
        console.error("No shared weight keys between the two checkpoints.")
        return {}

    deltas = []
    for k in keys:
        va, vb = ma[k], mb[k]
        if va.shape != vb.shape:
            continue
        d = float(np.max(np.abs(va - vb)))
        deltas.append({"tensor": k, "max_abs_delta": round(d, 6),
                       "shape": list(va.shape)})
    deltas.sort(key=lambda d: d["max_abs_delta"], reverse=True)

    report = {
        "checkpoint_a": str(a),
        "checkpoint_b": str(b),
        "tensors_compared": len(deltas),
        "largest_deltas": deltas[:top_k],
        "mean_max_delta": round(float(np.mean([d["max_abs_delta"] for d in deltas])), 6),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    console.ok(f"Compared {len(deltas)} tensors (a={a.name}, b={b.name})")
    for d in report["largest_deltas"]:
        console.info(f"  {d['max_abs_delta']:.6f}  {d['tensor']}")
    return report


def archive(root: Path, cfg: Config, checkpoint: str) -> Optional[Path]:
    """Zip a checkpoint + config + eval report into checkpoints/archives/."""
    ckpt_dir = root / "checkpoints" / checkpoint
    if not ckpt_dir.is_dir():
        console.error(f"Checkpoint not found: {ckpt_dir}")
        return None

    archives = root / "checkpoints" / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = archives / f"{checkpoint}_{stamp}.zip"

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in ckpt_dir.rglob("*"):
            if p.is_file() and "archives" not in p.parts:
                zf.write(p, p.relative_to(root))
        for extra in ("metadata/eval_results.jsonl", "metadata/leaderboard.json",
                      "metadata/checkpoint_registry.json"):
            p = root / extra
            if p.exists():
                zf.write(p, p.relative_to(root))

    console.ok(f"Archived -> {out.relative_to(root)} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
