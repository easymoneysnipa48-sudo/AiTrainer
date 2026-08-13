"""Train/validation/test split by song (no leakage), with optional
stratification (#23) and k-fold cross-validation (#22).

Default: shuffle songs, split by ratio. With `split.stratify`, songs are
grouped by a metadata attribute (key/bpm/genre/mood) and split per-group so
each fold keeps the same label distribution. With `split.k_folds`, produces N
rotating train/val folds instead of a single train/val/test split.
"""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

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


def _load_song_attrs(root: Path) -> Dict[str, dict]:
    """Map song_id -> {key, bpm, genre, mood} from the feature manifest."""
    manifest = root / "metadata" / "manifest.jsonl"
    attrs: Dict[str, dict] = {}
    if not manifest.exists():
        return attrs
    try:
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            song_id = sanitize_slug(Path(rec.get("path", "")).stem)
            mood = rec.get("mood")
            if isinstance(mood, list):
                mood = "|".join(sorted(map(str, mood)))
            attrs[song_id] = {
                "key": rec.get("key"),
                "bpm": rec.get("bpm"),
                "genre": rec.get("genre"),
                "mood": mood,
            }
    except Exception:  # noqa: BLE001
        return {}
    return attrs


def _stratify_key(song_id: str, attrs: Dict[str, dict], field: str):
    a = attrs.get(song_id) or {}
    if field == "bpm":
        b = a.get("bpm")
        if b is None:
            return "unknown"
        return f"{(int(round(float(b) / 10)) * 10)}s"
    val = a.get(field)
    return val if val not in (None, "") else "unknown"


def _ratios(cfg: Config) -> Dict[str, float]:
    return {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}


def _split_indices(n: int, ratios: Dict[str, float]) -> tuple:
    """Return (n_train, n_val) honoring ratios, guaranteeing 1/split when possible."""
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    else:
        n_train, n_val = n, 0
    return n_train, n_val


def _assign(songs: List[str], cfg: Config, root: Path) -> Dict[str, List[str]]:
    """Random ratio split (optionally stratified by cfg.split.stratify)."""
    ratios = _ratios(cfg)
    rng = random.Random(cfg.split.seed)

    stratify = (cfg.split.stratify or "").strip()
    if not stratify:
        songs = sorted(songs)
        rng.shuffle(songs)
        n_train, n_val = _split_indices(len(songs), ratios)
        return {
            "train": songs[:n_train],
            "val": songs[n_train : n_train + n_val],
            "test": songs[n_train + n_val :],
        }

    # stratified: split each attribute bucket proportionally (#23)
    attrs = _load_song_attrs(root)
    buckets: Dict[str, List[str]] = defaultdict(list)
    for s in sorted(songs):
        buckets[_stratify_key(s, attrs, stratify)].append(s)

    train, val, test = [], [], []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_train, n_val = _split_indices(len(bucket), ratios)
        train += bucket[:n_train]
        val += bucket[n_train : n_train + n_val]
        test += bucket[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def _assign_kfold(songs: List[str], k: int, seed: int) -> List[Dict[str, List[str]]]:
    """Rotating train/val folds for k-fold CV (#22)."""
    rng = random.Random(seed)
    songs = sorted(songs)
    rng.shuffle(songs)
    n = len(songs)
    if n < k:
        console.warn(f"Only {n} songs for {k} folds — folds will be undersized.")
    folds = []
    for i in range(k):
        lo = (i * n) // k
        hi = ((i + 1) * n) // k
        val = songs[lo:hi]
        train = songs[:lo] + songs[hi:]
        folds.append({"fold": i, "train": train, "val": val})
    return folds


def _materialize(root: Path, assignment: Dict[str, List[str]], groups: Dict[str, List[dict]], cfg: Config) -> None:
    meta = root / "metadata"
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


def split(root: Path, cfg: Config, dry_run: bool = False) -> Dict[str, List[str]]:
    segments = _load_segments(root)
    if not segments:
        console.warn("No segments found (run `segment` first).")
        return {}

    groups = _group_by_song(segments)
    songs = list(groups.keys())
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)

    # k-fold CV (#22) — writes folds.json, no materialization
    if cfg.split.k_folds > 0:
        folds = _assign_kfold(songs, cfg.split.k_folds, cfg.split.seed)
        summary = {
            "mode": "kfold",
            "k": cfg.split.k_folds,
            "seed": cfg.split.seed,
            "folds": [
                {
                    "fold": f["fold"],
                    "train_songs": len(f["train"]),
                    "val_songs": len(f["val"]),
                    "train_segments": sum(len(groups[s]) for s in f["train"]),
                    "val_segments": sum(len(groups[s]) for s in f["val"]),
                    "train": f["train"],
                    "val": f["val"],
                }
                for f in folds
            ],
        }
        (meta / "folds.json").write_text(json.dumps(summary, indent=2))
        console.ok(f"Wrote {len(folds)} folds -> metadata/folds.json")
        for f in summary["folds"]:
            console.info(f"fold {f['fold']}: train {f['train_songs']} songs, val {f['val_songs']} songs")
        return {"folds": [f["train"] + f["val"] for f in folds]}

    assignment = _assign(songs, cfg, root)

    summary = {
        "seed": cfg.split.seed,
        "stratify": cfg.split.stratify or None,
        "ratios": {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test},
        "splits": {k: {"songs": v, "segments": sum(len(groups[s]) for s in v)} for k, v in assignment.items()},
    }
    (meta / "splits.json").write_text(json.dumps(summary, indent=2))

    console.step("Split summary (by song):")
    for split_name in ("train", "val", "test"):
        songs_in = assignment[split_name]
        segs = sum(len(groups[s]) for s in songs_in)
        console.ok(f"{split_name:5s}: {len(songs_in):3d} songs, {segs:4d} segments")

    if dry_run:
        return assignment

    _materialize(root, assignment, groups, cfg)
    console.ok("Split complete.")
    return assignment
