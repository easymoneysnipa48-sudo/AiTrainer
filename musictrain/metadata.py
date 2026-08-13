"""Metadata schema, extraction/merge, and validation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonlines

from . import console
from .audio.features import extract_file
from .config import Config
from .util import sanitize_slug

# Fields the model learns the text<->audio relationship from
DESCRIPTION_FIELDS = ["description", "genre", "mood", "instruments", "section", "bpm", "key"]

# Manual/curated fields (from a labels CSV/JSON)
MANUAL_FIELDS = [
    "source_id",
    "license",
    "genre",
    "mood",
    "instruments",
    "section",
    "description",
    "song_id",
    "section_index",
    "section_type",
    "start_time",
    "end_time",
    "narrative_role",
    "energy",
]


def _split_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    if not text.strip():
        return []
    sep = "|" if "|" in text else ";" if ";" in text else ","
    return [p.strip() for p in text.split(sep) if p.strip()]


def load_labels(path: Optional[Path]) -> Dict[str, dict]:
    """Load manual labels keyed by source_id (or file stem)."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        console.warn(f"Labels file not found: {path}")
        return {}

    out: Dict[str, dict] = {}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        items = data if isinstance(data, list) else data.get("labels", [])
    else:  # CSV
        with path.open(newline="") as fh:
            items = list(csv.DictReader(fh))

    for item in items:
        rec = {k: item.get(k) for k in MANUAL_FIELDS if item.get(k) not in (None, "")}
        if "mood" in rec:
            rec["mood"] = _split_list(rec["mood"])
        if "instruments" in rec:
            rec["instruments"] = _split_list(rec["instruments"])
        if "energy" in rec:
            try:
                rec["energy"] = float(rec["energy"])
            except (TypeError, ValueError):
                rec.pop("energy")
        key = str(rec.get("source_id") or item.get("source_id") or "")
        if not key:
            # fall back to filename stem if a "path" column exists
            key = sanitize_slug(item.get("path", ""))
        out[key] = rec
    return out


def build_record(path: Path, rel_path: str, feats: dict, manual: Optional[dict] = None) -> dict:
    record: dict = {"path": rel_path}
    record.update(feats)
    if manual:
        for k, v in manual.items():
            if k != "path":
                record[k] = v
    record.setdefault("source_id", sanitize_slug(Path(rel_path).stem))
    return record


def extract(
    root: Path,
    cfg: Config,
    which: str = "clean",
    labels_path: Optional[Path] = None,
    limit: int = 0,
) -> List[dict]:
    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return []

    labels = load_labels(labels_path)
    files = sorted(target.glob("*.wav")) or sorted(target.glob("*.flac"))
    if not files:
        console.warn(f"No audio files under {target}")
        return []

    records: List[dict] = []
    console.step(f"Extracting features for {len(files)} files (data/{which})")

    for i, path in enumerate(files, 1):
        if limit and i > limit:
            break
        rel = str(path.relative_to(root))
        stem = sanitize_slug(path.stem)
        manual = labels.get(stem) or labels.get(rel) or {}
        try:
            feats = extract_file(path, cfg.features)
        except Exception as exc:  # noqa: BLE001
            console.error(f"Failed {path.name}: {exc}")
            continue
        record = build_record(path, rel, feats, manual)
        records.append(record)
        console.info(
            f"[{i}/{len(files)}] {path.name}: "
            f"bpm={record.get('bpm')} key={record.get('key')} "
            f"lufs={record.get('lufs')}"
        )

    _save(root, records)
    return records


def _save(root: Path, records: List[dict]) -> None:
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)

    (meta / "manifest.json").write_text(json.dumps(records, indent=2))
    with jsonlines.open(meta / "manifest.jsonl", mode="w") as w:
        for r in records:
            w.write(r)
    console.ok(f"Wrote {len(records)} records -> metadata/manifest.json(l)")


def validate(records: List[dict]) -> List[str]:
    issues: List[str] = []
    for r in records:
        src = r.get("path", "?")
        if not r.get("duration") or r.get("duration", 0) <= 0:
            issues.append(f"{src}: missing/invalid duration")
        if r.get("sample_rate") and r["sample_rate"] != 32000:
            issues.append(f"{src}: sample_rate {r['sample_rate']} != 32000")
        if r.get("channels") and r["channels"] != 1:
            issues.append(f"{src}: channels {r['channels']} != 1")
        if r.get("bpm") is not None and not (20 <= r["bpm"] <= 400):
            issues.append(f"{src}: implausible bpm {r['bpm']}")
        if not r.get("license"):
            issues.append(f"{src}: missing license")
        if not r.get("description"):
            issues.append(f"{src}: missing description (critical for conditioning)")
    return issues
