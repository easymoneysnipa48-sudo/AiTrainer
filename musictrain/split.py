"""Train/validation/test split by song (no leakage)."""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import jsonlines

from . import console
from .config import Config
from .util import sanitize_slug


def _load_segments(root: Path) -> List[dict]:
    manifest = root / "metadata" / "segments.json"
    if manifest.exists():
        return json.loads(manifest.read_text())

    # fall back to scanning data/segments
    seg_dir = root / "data" / "segments"
    out: List[dict] = []
    for p in sorted(seg_dir.glob("*.wav")):
        stem = p.stem
        song_id = stem.rsplit("_seg", 1)[0] if "_seg" in stem else sanitize_slug(stem)
        out.append({"path": str(p.relative_to(root)), "song_id": song_id, "segment_index": 0})
    return out


def _group_by_song(segments: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for seg in segments:
        groups[seg.get("song_id") or sanitize_slug(Path(seg["path"]).stem)].append(seg)
    return dict(groups)


def _assign(songs: List[str], cfg: Config) -> Dict[str, List[str]]:
    ratios = {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}
    songs = sorted(songs)
    rng = random.Random(cfg.split.seed)
    rng.shuffle(songs)

    n = len(songs)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    # remaining goes to test; guarantee at least 1 per split when possible
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    else:
        n_train, n_val = n, 0

    return {
        "train": songs[:n_train],
        "val": songs[n_train : n_train + n_val],
        "test": songs[n_train + n_val :],
    }


def split(root: Path, cfg: Config, dry_run: bool = False) -> Dict[str, List[str]]:
    segments = _load_segments(root)
    if not segments:
        console.warn("No segments found (run `segment` first).")
        return {}

    groups = _group_by_song(segments)
    assignment = _assign(list(groups.keys()), cfg)

    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)

    summary = {
        "seed": cfg.split.seed,
        "ratios": {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test},
        "splits": {k: {"songs": v, "segments": sum(len(groups[s]) for s in v)} for k, v in assignment.items()},
    }
    (meta / "splits.json").write_text(json.dumps(summary, indent=2))

    console.step("Split summary (by song):")
    for split_name in ("train", "val", "test"):
        songs = assignment[split_name]
        segs = sum(len(groups[s]) for s in songs)
        console.ok(f"{split_name:5s}: {len(songs):3d} songs, {segs:4d} segments")

    if dry_run:
        return assignment

    # materialize files into data/{train,val,test}
    for split_name, songs in assignment.items():
        dest_dir = root / "data" / split_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for stale in dest_dir.glob("*.wav"):
            stale.unlink()
        records = []
        for song in songs:
            for seg in groups[song]:
                src = root / seg["path"]
                dst = dest_dir / src.name
                if cfg.split.mode == "link":
                    if not dst.exists():
                        dst.symlink_to(src)
                else:
                    shutil.copy2(src, dst)
                records.append({**seg, "split": split_name})

        with jsonlines.open(meta / f"{split_name}.jsonl", mode="w") as w:
            for r in records:
                w.write(r)
        console.info(f"data/{split_name}: {len(records)} files")

    console.ok("Split complete.")
    return assignment
