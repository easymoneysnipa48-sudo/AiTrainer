"""CLAP audio-embedding index + similarity search (Phase 1 #9).

Embeds every track once (cached to metadata/audio_embeddings.json keyed by
relative path) and exposes nearest-neighbour search so you can answer
"find me tracks like this one". Also reused by autolabel.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import console
from .audio.inventory import AUDIO_GLOB
from .config import Config


def _scan(dir_path: Path) -> List[Path]:
    found: List[Path] = []
    for pattern in AUDIO_GLOB:
        found.extend(sorted(dir_path.glob(pattern)))
    return sorted(set(found))


def embed_audio(cfg: Config, path: Path) -> np.ndarray:
    """Return the L2-normalized CLAP audio embedding for a single file."""
    import librosa
    import soundfile as sf
    import torch

    from .similarity import load_clap, resolve_device

    device = resolve_device(cfg.clap.device)
    fe, _, model, device = load_clap(cfg.clap.model_name, device)

    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        sr = 48000

    inputs = fe(raw_speech=[audio], sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    with torch.inference_mode():
        emb = model.get_audio_features(
            input_features=inputs["input_features"],
            is_longer=inputs.get("is_longer"),
        )
        if isinstance(emb, tuple):
            emb = emb[0].pooler_output
        else:
            emb = emb.pooler_output
    return emb[0].detach().cpu().numpy().astype(np.float32)


def _cache_path(root: Path) -> Path:
    return root / "metadata" / "audio_embeddings.json"


def embed_dir(root: Path, cfg: Config, which: str = "clean", limit: int = 0) -> Dict[str, np.ndarray]:
    target = root / "data" / which
    cache = _cache_path(root)
    stored: Dict[str, list] = json.loads(cache.read_text()) if cache.exists() else {}

    files = _scan(target)
    out: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=np.float32) for k, v in stored.items()}

    console.step(f"Embedding audio in data/{which} (cached: {len(out)})")
    for i, path in enumerate(files, 1):
        if limit and i > limit:
            break
        rel = str(path.relative_to(root))
        if rel in out:
            continue
        try:
            out[rel] = embed_audio(cfg, path)
            console.info(f"[{i}/{len(files)}] {path.name}")
        except Exception as exc:  # noqa: BLE001
            console.error(f"Embedding failed {path.name}: {exc}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({k: v.tolist() for k, v in out.items()}))
    return out


def refresh(root: Path, cfg: Config, which: str = "clean", limit: int = 0) -> Dict[str, int]:
    """Refresh the embedding cache (Advanced #29).

    Drops entries whose audio files no longer exist, then re-embeds any files
    that changed (size/mtime differ from the stored snapshot) or are new.
    """
    target = root / "data" / which
    cache = _cache_path(root)
    stored: Dict[str, dict] = {}
    if cache.exists():
        raw = json.loads(cache.read_text())
        for k, v in raw.items():
            stored[k] = v if isinstance(v, dict) else {"vec": v, "size": None, "mtime": None}

    files = _scan(target)
    live = {str(p.relative_to(root)) for p in files}

    removed = [k for k in stored if k not in live]
    for k in removed:
        del stored[k]

    out: Dict[str, np.ndarray] = {}
    changed = 0
    for i, p in enumerate(files, 1):
        if limit and i > limit:
            break
        rel = str(p.relative_to(root))
        st = p.stat()
        meta = stored.get(rel)
        if meta is not None and meta.get("size") == st.st_size and meta.get("mtime") == st.st_mtime:
            out[rel] = np.asarray(meta["vec"], dtype=np.float32)
            continue
        try:
            out[rel] = embed_audio(cfg, p)
            stored[rel] = {
                "vec": out[rel].tolist(),
                "size": st.st_size,
                "mtime": st.st_mtime,
            }
            changed += 1
            console.info(f"[{i}/{len(files)}] refreshed {p.name}")
        except Exception as exc:  # noqa: BLE001
            console.error(f"Embedding failed {p.name}: {exc}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(stored))
    console.ok(
        f"Cache refreshed: {len(out)} live, {len(removed)} pruned, "
        f"{changed} (re)embedded -> metadata/audio_embeddings.json"
    )
    return {"live": len(out), "pruned": len(removed), "changed": changed}


def nearest(
    root: Path,
    cfg: Config,
    query: Path,
    which: str = "clean",
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Return the top_k most similar tracks (rel path, cosine) to `query`."""
    emb = embed_dir(root, cfg, which=which)
    q = embed_audio(cfg, query)
    q = q / (np.linalg.norm(q) + 1e-12)

    scored: List[Tuple[str, float]] = []
    for rel, e in emb.items():
        e = e / (np.linalg.norm(e) + 1e-12)
        scored.append((rel, float(np.dot(q, e))))
    scored.sort(key=lambda t: t[1], reverse=True)
    # drop the query itself if it lives in the index
    qrel = str(query.resolve())
    scored = [t for t in scored if str((root / t[0]).resolve()) != qrel]
    return scored[:top_k]
