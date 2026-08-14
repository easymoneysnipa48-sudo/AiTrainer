"""Active learning for labeling (Advanced #23).

Ranks unlabeled tracks by expected labeling value, so a human labeler gets the
most informative tracks first:

* **Uncertainty** — low top-1 CLAP vocab score means the model can't confidently
  tag it; these need human eyes.
* **Diversity** — tracks far from the already-labeled embedding cloud add the
  most new coverage per label.

The two signals are z-scored and combined into a single ``priority`` (higher =
label first). Writes ``metadata/active_learning.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import console
from .config import Config


def _labeled_ids(root: Path) -> set:
    import csv

    p = root / "metadata" / "labels.csv"
    if not p.exists():
        return set()
    ids = set()
    for row in csv.DictReader(p.open(newline="")):
        sid = (row.get("source_id") or "").strip()
        if sid:
            ids.add(sid)
    return ids


def _zscore(vals: List[float]) -> np.ndarray:
    a = np.asarray(vals, dtype=float)
    if a.size == 0 or a.std() < 1e-12:
        return np.zeros_like(a)
    return (a - a.mean()) / (a.std() + 1e-12)


def rank_unlabeled(
    root: Path,
    cfg: Config,
    which: str = "clean",
    top_k: int = 20,
) -> List[dict]:
    if not cfg.clap.enabled:
        console.warn("CLAP disabled (clap.enabled=false) — cannot rank by uncertainty.")
        return []

    from .embeddings import embed_dir, _scan
    from .autolabel import _cos, _embed_text
    from .labels import VOCAB
    from .similarity import load_clap

    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return []

    fe, tok, model, device = load_clap(cfg.clap.model_name, cfg.clap.device)
    # vocabulary embeddings once (across genre/mood/instruments dims)
    vocab_terms = [t for dim in VOCAB.values() for t in dim]
    vocab_embs = [_embed_text(tok, model, device, t) for t in vocab_terms]
    vocab_embs = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in vocab_embs])

    index = embed_dir(root, cfg, which=which)
    labeled = _labeled_ids(root)

    unlabeled: List[Path] = []
    for p in _scan(target):
        if Path(p).stem not in labeled:
            unlabeled.append(p)

    console.step(
        f"Ranking {len(unlabeled)} unlabeled of {len(index)} embedded "
        f"(labeled: {len(labeled)})"
    )
    if not unlabeled:
        console.ok("Nothing to rank — all embedded tracks are labeled.")
        return []

    rows: List[dict] = []
    for p in unlabeled:
        rel = str(p.relative_to(root))
        e = index.get(rel)
        if e is None:
            continue
        e = e / (np.linalg.norm(e) + 1e-12)
        sims = vocab_embs @ e
        uncertainty = 1.0 - float(sims.max())  # low top-1 -> high uncertainty

        # diversity: distance to nearest labeled embedding (if any)
        diversity = 0.0
        labeled_embs = [index[r] for r in index if Path(r).stem in labeled]
        if labeled_embs:
            stack = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in labeled_embs])
            sims_l = stack @ e
            diversity = 1.0 - float(sims_l.max())

        rows.append(
            {
                "path": rel,
                "uncertainty": round(uncertainty, 4),
                "diversity": round(diversity, 4),
                "top_vocab_term": vocab_terms[int(np.argmax(vocab_embs @ e))],
            }
        )

    unc = _zscore([r["uncertainty"] for r in rows])
    div = _zscore([r["diversity"] for r in rows])
    for r, u, d in zip(rows, unc, div):
        r["priority"] = round(float(u + d), 4)
    rows.sort(key=lambda r: r["priority"], reverse=True)

    report = {
        "which": f"data/{which}",
        "unlabeled": len(rows),
        "ranked": rows[:top_k],
        "at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    out = root / "metadata" / "active_learning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    console.ok(f"Ranked {len(rows)} tracks -> metadata/active_learning.json (top {top_k})")
    for r in report["ranked"]:
        console.info(
            f"priority={r['priority']:+.2f}  unc={r['uncertainty']:.3f}  "
            f"div={r['diversity']:.3f}  {r['path']}"
        )
    return report["ranked"]
