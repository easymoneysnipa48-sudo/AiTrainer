"""Auto-section labeling (Advanced #26).

Maps each segment in ``metadata/segments.json`` to a section role by overlaying
its start time on the source track's detected structure
(``metadata/analysis.json`` → intro/verse/chorus/bridge/outro), then writes
``metadata/section_labels.json`` and patches ``section_type`` into
``metadata/labels.csv`` so the eval set can be filtered by section.

Roles come from structure-detection heuristics (energy/position), so treat them
as soft labels — the point is consistent section breakdown, not ground truth.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from . import console
from .config import Config


def _load_jsonl_or_json(root: Path, name: str) -> List[dict]:
    p = root / "metadata" / name
    if not p.exists():
        return []
    text = p.read_text()
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        data = None
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        return [data]
    # line-delimited JSONL fallback
    out: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _load_analysis(root: Path) -> Dict[str, dict]:
    """source stem -> analysis record (with structure segments)."""
    out: Dict[str, dict] = {}
    for rec in _load_jsonl_or_json(root, "analysis.jsonl"):
        path = rec.get("path", "")
        stem = Path(path).stem
        out[stem] = rec
    if not out:
        for rec in _load_jsonl_or_json(root, "analysis.json"):
            path = rec.get("path", "")
            stem = Path(path).stem
            out[stem] = rec
    return out


def _load_segments(root: Path) -> List[dict]:
    return _load_jsonl_or_json(root, "segments.json")


def _role_at(structure: dict, t: float) -> Optional[str]:
    """Find the section role containing timestamp t."""
    for seg in structure.get("segments", []):
        if seg.get("start", 0.0) <= t < seg.get("end", float("inf")):
            return seg.get("role") or seg.get("label") or "section"
    return None


def auto_sections(root: Path, cfg: Config, force: bool = False) -> Dict[str, object]:
    segments = _load_segments(root)
    if not segments:
        console.warn("No metadata/segments.json — run `musictrain segment` first.")
        return {"labeled": 0, "unmapped": 0}

    analysis = _load_analysis(root)
    if not analysis:
        console.warn(
            "No metadata/analysis.json(l) — run `musictrain analyze` first "
            "so structure roles exist to map onto."
        )
        return {"labeled": 0, "unmapped": 0}

    # id -> current labels.csv row (for patching section_type)
    labels_path = root / "metadata" / "labels.csv"
    rows: Dict[str, dict] = {}
    fieldnames: Optional[List[str]] = None
    if labels_path.exists():
        with labels_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            for row in reader:
                sid = (row.get("source_id") or "").strip()
                if sid:
                    rows[sid] = row

    assigned: List[dict] = []
    unmapped = 0
    for seg in segments:
        sid = seg.get("song_id") or Path(seg.get("source", "")).stem
        rec = analysis.get(sid) or analysis.get(Path(seg.get("path", "")).stem)
        if not rec:
            unmapped += 1
            continue
        role = _role_at(rec.get("structure", {}), float(seg.get("start_time", 0.0)))
        if not role:
            role = "unknown"
        assigned.append(
            {
                "segment": seg.get("path"),
                "song_id": sid,
                "start_time": seg.get("start_time"),
                "section_type": role,
            }
        )
        seg_id = Path(seg.get("path", "")).stem
        if seg_id in rows:
            rows[seg_id]["section_type"] = role

    # patch labels.csv (add section_type column if missing)
    if rows and fieldnames is not None:
        if "section_type" not in fieldnames:
            fieldnames.append("section_type")
        with labels_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows.values())

    out = root / "metadata" / "section_labels.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(assigned, indent=2))

    console.ok(
        f"Auto-labeled {len(assigned)} segment(s) (unmapped: {unmapped}) "
        f"-> metadata/section_labels.json"
    )
    if rows:
        console.info(f"Patched section_type into {len(rows)} labels.csv row(s)")
    from collections import Counter

    counts = Counter(a["section_type"] for a in assigned)
    if counts:
        console.info("Sections: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    return {"labeled": len(assigned), "unmapped": unmapped, "distribution": dict(counts)}
