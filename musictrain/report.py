"""Export eval results to CSV and a self-contained HTML report."""
from __future__ import annotations

import csv
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from . import console
from .config import Config

FIELDS = [
    "experiment_id", "checkpoint", "section", "genre", "key", "bpm_target",
    "detected_bpm", "deviation", "clap_score", "status", "seed", "prompt",
    "audio_path", "human_rating", "notes",
]


def load_results(root: Path) -> List[dict]:
    path = root / "metadata" / "eval_results.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def export(cfg: Config) -> Dict[str, str]:
    root = cfg.project_root
    rows = load_results(root)
    if not rows:
        console.warn("No eval results (metadata/eval_results.jsonl). Run `musictrain eval` first.")
        return {}

    meta = root / "metadata"
    csv_path = meta / "eval_results.csv"
    html_path = meta / "eval_report.html"

    _write_csv(csv_path, rows)
    _write_html(html_path, rows, cfg)

    console.ok(f"CSV  -> {csv_path.relative_to(root)}")
    console.ok(f"HTML -> {html_path.relative_to(root)}")
    return {"csv": str(csv_path), "html": str(html_path)}


def _write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _fmt_dev(d) -> str:
    if d is None:
        return "—"
    return f"{d * 100:+.1f}%"


def _fmt_num(v, nd: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def _write_html(path: Path, rows: List[dict], cfg: Config) -> None:
    n = len(rows)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    devs = [abs(r["deviation"]) for r in rows if r.get("deviation") is not None]
    claps = [r["clap_score"] for r in rows if r.get("clap_score") is not None]
    mean_dev = sum(devs) / len(devs) if devs else 0.0
    mean_clap = sum(claps) / len(claps) if claps else 0.0

    by_section: Dict[str, List[int]] = {}
    for r in rows:
        s = r.get("section") or "?"
        by_section.setdefault(s, [0, 0])
        by_section[s][1] += 1
        if r.get("status") == "ok":
            by_section[s][0] += 1

    meta_dir = path.parent
    rows_html = []
    for r in rows:
        audio = r.get("audio_path", "")
        if audio and os.path.exists(audio):
            rel = os.path.relpath(audio, meta_dir)
            audio_cell = f'<a href="{html.escape(rel)}">▶ listen</a>'
        else:
            audio_cell = "—"
        status = r.get("status") or "—"
        badge = "ok" if status == "ok" else "bad"
        rows_html.append(
            "<tr>"
            f"<td class='prompt'>{html.escape((r.get('prompt') or '')[:70])}</td>"
            f"<td>{html.escape(str(r.get('section') or ''))}</td>"
            f"<td>{html.escape(str(r.get('key') or ''))}</td>"
            f"<td>{_fmt_num(r.get('bpm_target'), 0)}</td>"
            f"<td>{_fmt_num(r.get('detected_bpm'))}</td>"
            f"<td>{_fmt_dev(r.get('deviation'))}</td>"
            f"<td>{_fmt_num(r.get('clap_score'), 3)}</td>"
            f"<td><span class='badge {badge}'>{html.escape(str(status))}</span></td>"
            f"<td>{_fmt_num(r.get('human_rating'), 0)}</td>"
            f"<td>{audio_cell}</td>"
            "</tr>"
        )

    section_rows = "".join(
        f"<tr><td>{html.escape(s)}</td><td>{v[0]}/{v[1]}</td></tr>"
        for s, v in sorted(by_section.items())
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MusicTrain eval report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 40px; color: #1a1a1a; background: #fafafa; }}
  h1 {{ margin-bottom: 4px; }}
  .sub {{ color: #666; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
  .card {{ background: #fff; border: 1px solid #e5e5e5; border-radius: 8px; padding: 16px 20px; min-width: 140px; }}
  .card .v {{ font-size: 26px; font-weight: 700; }}
  .card .k {{ color: #666; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; margin-bottom: 28px; font-size: 14px; }}
  th, td {{ border: 1px solid #e5e5e5; padding: 8px 10px; text-align: left; }}
  th {{ background: #f2f2f2; }}
  td.prompt {{ font-family: ui-monospace, monospace; font-size: 13px; }}
  .badge {{ padding: 2px 8px; border-radius: 10px; font-size: 12px; }}
  .badge.ok {{ background: #e6f4ea; color: #1e7e34; }}
  .badge.bad {{ background: #fdecea; color: #c0392b; }}
  a {{ color: #1a73e8; text-decoration: none; }}
  h2 {{ margin-top: 28px; }}
</style>
</head>
<body>
<h1>🎵 MusicTrain eval report</h1>
<div class="sub">checkpoint: {html.escape(str(cfg.inference.model_name))} · generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · {n} runs</div>

<div class="cards">
  <div class="card"><div class="v">{n}</div><div class="k">runs</div></div>
  <div class="card"><div class="v">{ok}/{n}</div><div class="k">BPM in-tolerance</div></div>
  <div class="card"><div class="v">{mean_dev * 100:.1f}%</div><div class="k">mean |deviation|</div></div>
  <div class="card"><div class="v">{mean_clap:.3f}</div><div class="k">mean CLAP score</div></div>
</div>

<h2>By section</h2>
<table style="max-width:360px"><tr><th>section</th><th>in-tolerance</th></tr>{section_rows}</table>

<h2>Results</h2>
<table>
<tr><th>prompt</th><th>section</th><th>key</th><th>target</th><th>detected</th><th>deviation</th><th>clap</th><th>status</th><th>rating</th><th>audio</th></tr>
{''.join(rows_html)}
</table>
</body>
</html>
"""
    path.write_text(doc)
