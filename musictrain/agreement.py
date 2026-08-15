"""Inter-annotator agreement on label files (Phase 4 #29).

Two humans (or a human vs. an auto-labeler) label the same tracks; this module
reports how much they agree, per field, via:

* **exact agreement** — fraction of shared source_ids whose tag sets are equal
* **Cohen's kappa** — chance-corrected agreement, macro-averaged over the union
  of tags per field (each tag treated as present/absent)

Writes ``metadata/agreement.json`` so the dashboard can surface it.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from . import console
from .labels import _split

TAG_FIELDS = ("genre", "mood", "instruments", "section", "section_type")


def load_rows(path: Path) -> Dict[str, dict]:
    """Rows keyed by source_id (duplicate source_ids collapse to the last)."""
    rows: Dict[str, dict] = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("source_id") or "").strip()
            if sid:
                rows[sid] = row
    return rows


def _tag_set(row: dict, field: str) -> Set[str]:
    return set(_split(row.get(field) or ""))


def _cohen_kappa(a: int, b: int, c: int, d: int) -> float:
    """Binary Cohen's kappa from contingency [a b; c d] (a = both present)."""
    n = a + b + c + d
    if n == 0:
        return 0.0
    p_o = (a + d) / n
    p_e = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def field_stats(rows_a: Dict[str, dict], rows_b: Dict[str, dict], field: str, sids: List[str]) -> Dict[str, float]:
    exact_agree = 0
    contingency: Dict[str, List[int]] = {}
    for sid in sids:
        set_a = _tag_set(rows_a[sid], field)
        set_b = _tag_set(rows_b[sid], field)
        if set_a == set_b:
            exact_agree += 1
        for tag in set_a | set_b:
            in_a = tag in set_a
            in_b = tag in set_b
            cell = contingency.setdefault(tag, [0, 0, 0, 0])
            if in_a and in_b:
                cell[0] += 1
            elif in_a:
                cell[1] += 1
            elif in_b:
                cell[2] += 1
            else:  # pragma: no cover — tag only added when in a or b
                cell[3] += 1

    kappas = [_cohen_kappa(*cells) for cells in contingency.values() if sum(cells)]
    return {
        "rows": len(sids),
        "exact_agreement": round(exact_agree / len(sids), 4) if sids else 0.0,
        "kappa": round(sum(kappas) / len(kappas), 4) if kappas else 0.0,
        "tags": len(contingency),
    }


def disagreements(rows_a: Dict[str, dict], rows_b: Dict[str, dict], sids: List[str], limit: int = 10) -> List[dict]:
    out: List[dict] = []
    for sid in sids:
        for field in TAG_FIELDS:
            set_a = _tag_set(rows_a[sid], field)
            set_b = _tag_set(rows_b[sid], field)
            if set_a != set_b:
                out.append(
                    {
                        "source_id": sid,
                        "field": field,
                        "annotator_a": sorted(set_a),
                        "annotator_b": sorted(set_b),
                    }
                )
        if len(out) >= limit:
            break
    return out[:limit]


def agreement(a_path: Path, b_path: Path, root: Path, out_rel: str = "metadata/agreement.json") -> Dict[str, object]:
    rows_a = load_rows(a_path)
    rows_b = load_rows(b_path)
    sids = [s for s in rows_a if s in rows_b]
    if not sids:
        console.error("No shared source_ids between the two label files.")
        return {}

    fields = {f: field_stats(rows_a, rows_b, f, sids) for f in TAG_FIELDS}
    overall_exact = sum(v["exact_agreement"] for v in fields.values()) / len(fields)
    overall_kappa = sum(v["kappa"] for v in fields.values()) / len(fields)

    report = {
        "annotators": [str(a_path), str(b_path)],
        "shared_tracks": len(sids),
        "fields": fields,
        "overall": {
            "exact_agreement": round(overall_exact, 4),
            "kappa": round(overall_kappa, 4),
        },
        "disagreements": disagreements(rows_a, rows_b, sids),
        "at": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(root) / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    console.ok(f"Wrote agreement report -> {out_rel} ({len(sids)} shared tracks)")
    for f, s in fields.items():
        console.info(
            f"{f:12s} exact {s['exact_agreement']:.0%} · kappa {s['kappa']:+.2f} ({s['rows']} rows)"
        )
    console.info(
        f"overall  exact {overall_exact:.0%} · kappa {overall_kappa:+.2f}"
    )
    return report
