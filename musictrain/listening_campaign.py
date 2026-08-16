"""Listening-campaign manager (#7).

Structured blind listening with session persistence and per-rater agreement.
A campaign is a directory under ``metadata/campaigns/<name>/`` holding:

* ``campaign.json`` — the (blind) item list: prompts, clip paths, and the
  randomized X/Y key so results can be unblinded later.
* ``ratings.jsonl`` — one row per rater/item judgement.

Modes: ``ab`` (blind A/B between two checkpoints per prompt) and ``mos``
(single-clip 1–5 scoring). Agreement is the mean majority fraction per item.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import console
from .logging import get_logger

log = get_logger("campaign")


def _clip(r: dict) -> dict:
    return {
        "prompt": r.get("prompt"),
        "audio_path": r.get("audio_path"),
        "checkpoint": r.get("checkpoint"),
        "clap_score": r.get("clap_score"),
        "deviation": r.get("deviation"),
        "section": r.get("section"),
        "bpm_target": r.get("bpm_target"),
    }


def start(root: Path, name: str, mode: str = "ab", seed: int = 0,
          limit: int = 0) -> dict:
    from .report import load_results

    rows = [r for r in load_results(root) if r.get("audio_path")]
    if not rows:
        console.error("No eval results with audio to build a campaign from.")
        return {"error": "no eval results"}

    rng = random.Random(seed)
    items: List[dict] = []
    if mode == "ab":
        by_prompt = defaultdict(list)
        for r in rows:
            by_prompt[r["prompt"]].append(r)
        for prompt, rs in by_prompt.items():
            uniq = {}
            for r in rs:
                uniq.setdefault(r.get("checkpoint"), r)
            if len(uniq) < 2:
                continue
            pair = list(uniq.values())[:2]
            order = [0, 1] if rng.random() < 0.5 else [1, 0]
            items.append({
                "id": f"ab_{len(items):03d}",
                "mode": "ab",
                "prompt": prompt,
                "x": _clip(pair[order[0]]),
                "y": _clip(pair[order[1]]),
            })
    else:  # mos
        for r in rows:
            items.append({"id": f"mos_{len(items):03d}", "mode": "mos", "clip": _clip(r)})

    if limit:
        items = items[:limit]

    camp_dir = root / "metadata" / "campaigns" / name
    camp_dir.mkdir(parents=True, exist_ok=True)
    campaign = {
        "name": name, "mode": mode, "seed": seed,
        "n_items": len(items), "items": items,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (camp_dir / "campaign.json").write_text(json.dumps(campaign, indent=2))
    console.ok(f"Campaign {name!r} ({mode}): {len(items)} item(s)")
    return campaign


def load_campaign(root: Path, name: str) -> Optional[dict]:
    p = root / "metadata" / "campaigns" / name / "campaign.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def record(root: Path, name: str, rater: str, item_id: str,
           choice: str, rating: Optional[int] = None, note: str = "") -> dict:
    """choice: 'X' | 'Y' | 'tie' (ab) or a 1–5 MOS score passed via ``rating``."""
    camp = load_campaign(root, name)
    if camp is None:
        return {"error": f"no campaign {name!r}"}
    row = {
        "rater": rater,
        "item_id": item_id,
        "choice": choice,
        "rating": rating,
        "note": note,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    path = root / "metadata" / "campaigns" / name / "ratings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def load_ratings(root: Path, name: str) -> List[dict]:
    p = root / "metadata" / "campaigns" / name / "ratings.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def agreement(root: Path, name: str) -> dict:
    """Mean majority fraction per item = inter-rater agreement proxy."""
    ratings = load_ratings(root, name)
    if not ratings:
        return {"n_ratings": 0, "n_raters": 0, "n_items": 0, "agreement": None}

    by_item: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in ratings:
        by_item[r["item_id"]][r["choice"]] += 1

    per_item = []
    for item_id, counts in by_item.items():
        n = sum(counts.values())
        majority = max(counts.values())
        per_item.append({"item_id": item_id, "n_raters": n, "agreement": majority / n})

    n_raters = len({r["rater"] for r in ratings})
    mean_agreement = round(sum(p["agreement"] for p in per_item) / len(per_item), 4)
    return {
        "n_ratings": len(ratings),
        "n_raters": n_raters,
        "n_items": len(per_item),
        "agreement": mean_agreement,
        "per_item": per_item,
    }


def unblind(root: Path, name: str) -> Dict[str, dict]:
    """Map each item's blind X/Y back to the underlying checkpoints."""
    camp = load_campaign(root, name)
    if not camp or camp.get("mode") != "ab":
        return {}
    out = {}
    for it in camp["items"]:
        out[it["id"]] = {
            "X": it["x"]["checkpoint"],
            "Y": it["y"]["checkpoint"],
        }
    return out
