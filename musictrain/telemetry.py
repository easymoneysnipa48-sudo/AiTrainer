"""Export + structured logging (Advanced #45/#46).

* **``export_wandb``** — pushes the eval aggregate (and per-prompt rows) to a
  Weights & Biases run; falls back to a TensorBoard-style scalar CSV if wandb
  isn't configured. Comparison views land in ``metadata/export/``.
* **``json_log``** — appends structured JSON lines for every run/command to
  ``metadata/runlog.jsonl`` — the dashboard's log viewer can tail it, and CI
  can query it.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import console
from .config import Config
from .report import load_results


def _runlog_path(root: Path) -> Path:
    return root / "metadata" / "runlog.jsonl"


def json_log(root: Path, event: str, **fields) -> None:
    """Append a structured JSON line to metadata/runlog.jsonl (#46)."""
    record = {
        "at": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    path = _runlog_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def read_runlog(root: Path, event: Optional[str] = None, limit: int = 200) -> List[dict]:
    path = _runlog_path(root)
    if not path.exists():
        return []
    rows: List[dict] = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if event is None or r.get("event") == event:
            rows.append(r)
    return rows[-limit:]


def export_wandb(cfg: Config, project: Optional[str] = None) -> Path:
    """Push eval aggregates to W&B; fall back to a local CSV export."""
    rows = load_results(cfg.project_root)
    if not rows:
        console.error("No eval results to export.")
        return cfg.project_root / "metadata" / "export" / "EMPTY.csv"

    by_ckpt: Dict[str, List[dict]] = {}
    for r in rows:
        by_ckpt.setdefault(r.get("checkpoint") or "?", []).append(r)

    export_dir = cfg.project_root / "metadata" / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_path = export_dir / "eval_wandb.csv"

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["checkpoint", "mean_clap", "mean_abs_deviation", "ok_pct", "n"])
        for ckpt, cres in sorted(by_ckpt.items()):
            claps = [r["clap_score"] for r in cres if r.get("clap_score") is not None]
            devs = [abs(r["deviation"]) for r in cres if r.get("deviation") is not None]
            ok = sum(1 for r in cres if r.get("status") == "ok")
            writer.writerow(
                [
                    ckpt,
                    round(sum(claps) / len(claps), 4) if claps else "",
                    round(sum(devs) / len(devs), 4) if devs else "",
                    round(ok / len(cres), 4),
                    len(cres),
                ]
            )

    try:
        import wandb  # noqa: F401
    except ImportError:
        console.ok(
            f"wandb not installed — wrote CSV comparison -> {csv_path.relative_to(cfg.project_root)}"
        )
        return csv_path

    try:
        run = wandb.init(project=project or "musictrain", reinit=True)
        for ckpt, cres in sorted(by_ckpt.items()):
            claps = [r["clap_score"] for r in cres if r.get("clap_score") is not None]
            devs = [abs(r["deviation"]) for r in cres if r.get("deviation") is not None]
            ok = sum(1 for r in cres if r.get("status") == "ok")
            run.log(
                {
                    f"{ckpt}/mean_clap": (sum(claps) / len(claps)) if claps else None,
                    f"{ckpt}/mean_abs_deviation": (sum(devs) / len(devs)) if devs else None,
                    f"{ckpt}/ok_pct": ok / len(cres),
                }
            )
        run.finish()
        console.ok(
            f"Logged {len(by_ckpt)} checkpoint(s) to W&B + CSV -> "
            f"{csv_path.relative_to(cfg.project_root)}"
        )
    except Exception as exc:  # noqa: BLE001 - W&B must never break the export
        console.warn(
            f"W&B logging failed ({exc}) — CSV comparison kept at "
            f"{csv_path.relative_to(cfg.project_root)}"
        )
    return csv_path
