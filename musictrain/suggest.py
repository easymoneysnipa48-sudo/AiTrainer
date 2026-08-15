"""Auto-suggest labels for a track (Phase 4 #31).

Combines two signals for a human labeler to review:

1. **Vocabulary proposals** — CLAP text similarity of the track against every
   controlled-vocabulary term, per dimension (same engine as ``autolabel``).
2. **Labeled neighbors** — nearest neighbors from the cached audio-embedding
   index whose labels already exist in ``labels.csv``, so you can copy labels
   from a known track instead of guessing.

Writes ``metadata/label_suggestions.json``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import console
from .config import Config


def _load_labeled(root: Path) -> Dict[str, dict]:
    """source_id -> row for every row in metadata/labels.csv (if present)."""
    p = root / "metadata" / "labels.csv"
    if not p.exists():
        return {}
    rows: Dict[str, dict] = {}
    for row in csv.DictReader(p.open(newline="")):
        sid = (row.get("source_id") or "").strip()
        if sid:
            rows[sid] = row
    return rows


def _vocab_proposals(cfg: Config, audio_emb: np.ndarray, top_k: int) -> Dict[str, List[dict]]:
    """Top vocab terms per dimension, scored by CLAP text-embedding cosine."""
    from .autolabel import _cos, _embed_text
    from .labels import VOCAB
    from .similarity import load_clap

    _fe, tok, model, device = load_clap(cfg.clap.model_name, cfg.clap.device)
    proposals: Dict[str, List[dict]] = {}
    for dim in ("genre", "mood", "instruments"):
        scored = []
        for term in VOCAB[dim]:
            t_emb = _embed_text(tok, model, device, term)
            scored.append((term, _cos(audio_emb, t_emb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        proposals[dim] = [
            {"tag": t, "score": round(s, 4)}
            for t, s in scored[:top_k]
            if s >= cfg.autolabel.min_confidence
        ]
    return proposals


def suggest(
    root: Path,
    cfg: Config,
    query_path: Path,
    top_k: int = 5,
    which: str = "clean",
) -> Dict[str, object]:
    if not cfg.clap.enabled:
        console.warn("CLAP is disabled (clap.enabled=false) — nothing to suggest.")
        return {}

    from .embeddings import embed_audio, nearest

    console.step(f"Suggesting labels for {query_path.name}")
    q = embed_audio(cfg, query_path)
    q = q / (np.linalg.norm(q) + 1e-12)

    vocab = _vocab_proposals(cfg, q, top_k=cfg.autolabel.top_k)
    labeled = _load_labeled(root)

    neighbors = []
    for rel, sim in nearest(root, cfg, query_path, which=which, top_k=top_k):
        sid = Path(rel).stem
        row = labeled.get(sid)
        neighbors.append(
            {
                "path": rel,
                "similarity": round(sim, 4),
                "labels": (
                    {
                        k: row.get(k, "")
                        for k in ("genre", "mood", "instruments", "section", "section_type")
                    }
                    if row
                    else None
                ),
            }
        )

    report = {
        "query": str(query_path),
        "vocab_proposals": vocab,
        "labeled_neighbors": neighbors,
        "at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    out = root / "metadata" / "label_suggestions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    console.ok("Wrote suggestions -> metadata/label_suggestions.json")
    for dim, props in vocab.items():
        console.info(
            f"{dim:12s} "
            + " · ".join(f"{p['tag']} ({p['score']:.3f})" for p in props)
        )
    if neighbors:
        console.info(f"Nearest labeled neighbors: {len(neighbors)}")
    return report
