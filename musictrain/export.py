"""Export the split corpus into a training-ready Hugging Face `datasets` format
(#26): Arrow DatasetDict (default) with an Audio column, or a flat JSONL/CSV
manifest.

Reads the split manifests (metadata/{train,val,test}.jsonl) written by
`musictrain split` and joins each segment against the feature manifest so every
row carries its conditioning text (description/genre/mood/instruments) plus
BPM/key. `datasets` is imported lazily so the rest of the toolkit never pays
for it.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import jsonlines

from . import console
from .config import Config
from .util import sanitize_slug

_TEXT_FIELDS = ("description", "genre", "mood", "instruments", "section", "bpm", "key")


def _load_feature_map(root: Path) -> Dict[str, dict]:
    """Map source_id -> full feature record from metadata/manifest.jsonl."""
    manifest = root / "metadata" / "manifest.jsonl"
    out: Dict[str, dict] = {}
    if not manifest.exists():
        return out
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = rec.get("source_id") or sanitize_slug(Path(rec.get("path", "")).stem)
        out[key] = rec
    return out


def _text(rec: dict) -> str:
    """Build conditioning text for a record (description if present)."""
    if rec.get("description"):
        return rec["description"]
    parts = []
    for f in ("section", "genre"):
        v = rec.get(f)
        if v:
            parts.append(str(v))
    for f in ("mood", "instruments"):
        v = rec.get(f)
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        if v:
            parts.append(str(v))
    if rec.get("bpm"):
        parts.append(f"{rec['bpm']} BPM")
    if rec.get("key"):
        parts.append(str(rec["key"]))
    return ", ".join(parts)


def _load_split_rows(root: Path, which: str) -> List[dict]:
    """Load + join split records for `which` (train/val/test/all)."""
    names = ["train", "val", "test"] if which == "all" else [which]
    feat = _load_feature_map(root)
    rows: List[dict] = []
    for name in names:
        p = root / "metadata" / f"{name}.jsonl"
        if not p.exists():
            console.warn(f"Missing {name}.jsonl — run `musictrain split` first.")
            continue
        for seg in jsonlines.open(p):
            key = seg.get("song_id") or sanitize_slug(Path(seg.get("path", "")).stem)
            fr = feat.get(key, {})
            row = {k: v for k, v in fr.items() if k != "path"}
            row.update(seg)
            row.setdefault("text", _text({**fr, **seg}))
            rows.append(row)
    return rows


def export(root: Path, cfg: Config, which: str = "", format_: str = "") -> dict:
    which = which or cfg.export.which or "all"
    format_ = format_ or cfg.export.format or "arrow"
    rows = _load_split_rows(root, which)
    if not rows:
        console.warn("Nothing to export — run `musictrain split` first.")
        return {}

    out_dir = root / "data" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    if format_ == "arrow":
        return _export_arrow(rows, out_dir, cfg)
    if format_ == "jsonl":
        out = out_dir / f"manifest_{which}.jsonl"
        with jsonlines.open(out, mode="w") as w:
            for r in rows:
                w.write(r)
        console.ok(f"Wrote {len(rows)} rows -> {out.relative_to(root)}")
        return {"format": "jsonl", "rows": len(rows), "path": str(out.relative_to(root))}
    if format_ == "csv":
        out = out_dir / f"manifest_{which}.csv"
        cols = list(rows[0].keys())
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
        console.ok(f"Wrote {len(rows)} rows -> {out.relative_to(root)}")
        return {"format": "csv", "rows": len(rows), "path": str(out.relative_to(root))}

    console.error(f"Unknown export format {format_!r} (arrow|jsonl|csv)")
    return {}


def _export_arrow(rows: List[dict], out_dir: Path, cfg: Config) -> dict:
    try:
        from datasets import Audio, Dataset, DatasetDict
    except ImportError:
        console.error("`datasets` is not installed — run `uv pip install datasets` (or export jsonl/csv).")
        return {}

    # normalize list/dict cells that Arrow can't store natively
    clean = []
    for r in rows:
        clean.append({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})

    splits: Dict[str, Dataset] = {}
    by_split: Dict[str, List[dict]] = {}
    for r in clean:
        by_split.setdefault(r.get("split", "all"), []).append(r)

    for name, rows_in in by_split.items():
        ds = Dataset.from_list(rows_in)
        if cfg.export.audio_column and "path" in ds.column_names:
            ds = ds.cast_column("path", Audio(sampling_rate=cfg.normalize.sample_rate))
        splits[name] = ds

    dset = DatasetDict(splits) if len(splits) > 1 else next(iter(splits.values()))
    dset.save_to_disk(str(out_dir), max_shard_size=cfg.export.max_shard_size)
    console.ok(f"Saved {len(clean)} rows -> data/dataset/ ({len(splits)} split(s))")
    return {"format": "arrow", "rows": len(clean), "splits": list(splits.keys()), "path": "data/dataset"}
