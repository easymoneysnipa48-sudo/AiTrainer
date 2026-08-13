"""CLAP-driven auto pre-labeling (Phase 1 #6).

Embeds each track's audio and every controlled-vocabulary term with CLAP, then
proposes the top genre/mood/instrument tags per track for a human to review.
Writes metadata/autolabels.csv (suggestions only — never overwrites the manual
labels.csv).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import console
from .config import Config
from .embeddings import embed_dir


def _embed_text(tok, model, device, text: str) -> np.ndarray:
    import torch

    inputs = tok([text], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    with torch.inference_mode():
        emb = model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        if isinstance(emb, tuple):
            emb = emb[0].pooler_output
        else:
            emb = emb.pooler_output
    return emb[0].detach().cpu().numpy().astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def autolabel(root: Path, cfg: Config, which: str = "clean", limit: int = 0) -> List[dict]:
    if not cfg.autolabel.enabled:
        console.warn("Auto-labeling disabled (autolabel.enabled=false).")
        return []

    from .labels import VOCAB
    from .similarity import load_clap, resolve_device

    dims = ["genre", "mood", "instruments"]
    device = resolve_device(cfg.autolabel.device)
    fe, tok, model, device = load_clap(cfg.clap.model_name, device)

    console.info("Embedding vocabulary terms…")
    text_emb: Dict[Tuple[str, str], np.ndarray] = {}
    for dim in dims:
        for term in sorted(VOCAB[dim]):
            text_emb[(dim, term)] = _embed_text(tok, model, device, term)

    emb = embed_dir(root, cfg, which=which, limit=limit)
    if not emb:
        console.warn("No embeddings available.")
        return []

    results: List[dict] = []
    console.step(f"Proposing labels for {len(emb)} tracks")
    for rel, aemb in emb.items():
        rec: dict = {"path": rel}
        for dim in dims:
            scored = sorted(
                ((t, _cos(aemb, text_emb[(dim, t)])) for t in VOCAB[dim]),
                key=lambda x: x[1],
                reverse=True,
            )
            top = [
                {"tag": t, "score": round(s, 4)}
                for t, s in scored[: cfg.autolabel.top_k]
                if s >= cfg.autolabel.min_confidence
            ]
            rec[dim] = top
        results.append(rec)

    out = root / "metadata" / "autolabels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "genre", "genre_scores", "mood", "mood_scores", "instruments", "instruments_scores"])
        for r in results:
            def tags(dim):
                return "|".join(t["tag"] for t in r[dim])
            def scores(dim):
                return "|".join(f"{t['score']:.3f}" for t in r[dim])
            w.writerow([
                r["path"],
                tags("genre"), scores("genre"),
                tags("mood"), scores("mood"),
                tags("instruments"), scores("instruments"),
            ])

    console.ok(f"Wrote {len(results)} suggestions -> metadata/autolabels.csv")
    return results
