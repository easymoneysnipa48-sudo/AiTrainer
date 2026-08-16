"""Model weight merging (advanced #13).

Averages the weights of two or more checkpoints into a merged model
directory, copying config/index files from the first checkpoint so it can be
loaded by ``AutoModelForTextToWaveform``. Supports safetensors shards and the
legacy pytorch_model.bin format.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import console


def _tensor_files(model_dir: Path) -> List[Path]:
    st = sorted(model_dir.glob("*.safetensors"))
    if st:
        return st
    bin = model_dir / "pytorch_model.bin"
    return [bin] if bin.exists() else []


def _load_weights(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    import torch

    # legacy .bin checkpoints are trusted local files; weights_only=False is
    # required for pickles from older torch versions
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _save_weights(weights: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(weights, str(path))
    else:
        import torch

        tensors = {k: torch.as_tensor(v) for k, v in weights.items()}
        torch.save(tensors, str(path))


def merge(model_dirs: List[Path], out_dir: Path, weights: Optional[List[float]] = None) -> Path:
    """Average weights of model_dirs into out_dir. Returns out_dir."""
    model_dirs = [Path(m) for m in model_dirs]
    if len(model_dirs) < 2:
        console.error("Merging needs >= 2 checkpoints.")
        raise ValueError("need >= 2 checkpoints")
    if weights is None:
        weights = [1.0 / len(model_dirs)] * len(model_dirs)
    if len(weights) != len(model_dirs):
        console.error("--weights must match the number of checkpoints.")
        raise ValueError("weights length mismatch")
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    # each model may be one dir with shards, or a single file per model
    groups: List[List[Path]] = []
    for m in model_dirs:
        files = _tensor_files(m)
        if not files:
            console.error(f"No weights found in {m}")
            raise FileNotFoundError(f"no weights in {m}")
        groups.append(files)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = model_dirs[0]
    # copy config + non-weight files (config.json, generation_config.json, …)
    for f in src.iterdir():
        if f.suffix not in (".safetensors", ".bin") and f.name != "model.safetensors.index.json":
            dst = out_dir / f.name
            if f.is_dir():
                shutil.copytree(f, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(f, dst)

    # all checkpoints must share the same shard layout, else a naive
    # element-wise average would silently drop shards (gap: zip truncation).
    n_shards = {len(g) for g in groups}
    if len(n_shards) != 1:
        console.error(
            f"Checkpoints have different shard counts {sorted(n_shards)} — "
            "cannot merge safely. Re-export to a common shard layout first."
        )
        raise ValueError("mismatched shard layouts")

    import torch

    # average shard-by-shard (same layout guaranteed above)
    for files in zip(*groups, strict=True):
        loaded = [_load_weights(f) for f in files]
        keys = set(loaded[0].keys())
        merged: dict = {}
        for key in keys:
            if any(key not in wts for wts in loaded):
                continue  # key missing in one checkpoint — drop it
            vals = [wts[key].to(torch.float32).numpy() for wts in loaded]
            merged[key] = np.average(np.stack(vals, axis=0), axis=0, weights=w)
        out_file = out_dir / files[0].name
        _save_weights({k: np.asarray(v) for k, v in merged.items()}, out_file)
        console.step(f"merged {out_file.name} ({len(merged)} tensors)")

    console.ok(f"Merged {len(model_dirs)} checkpoints -> {out_dir}")
    return out_dir


def run(paths: List[str], out: str, weights: Optional[List[float]] = None) -> Path:
    return merge([Path(p) for p in paths], Path(out), weights=weights)
