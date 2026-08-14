"""Eval alerting (Advanced #47).

Watches the eval aggregate against thresholds (CLAP floor, deviation ceiling)
and fires when a checkpoint regresses: Slack webhook (zero deps, urllib),
email (SMTP), or a local alert file that CI/dashboards can poll.
"""
from __future__ import annotations

import json
import smtplib
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional

from . import console
from .config import Config
from .report import load_results


def _aggregate(rows: List[dict]) -> Dict[str, float]:
    claps = [r["clap_score"] for r in rows if r.get("clap_score") is not None]
    devs = [abs(r["deviation"]) for r in rows if r.get("deviation") is not None]
    ok = sum(1 for r in rows if r.get("status") == "ok")
    return {
        "mean_clap": (sum(claps) / len(claps)) if claps else None,
        "mean_abs_deviation": (sum(devs) / len(devs)) if devs else None,
        "ok_pct": (ok / len(rows)) if rows else None,
    }


def check_alerts(
    cfg: Config,
    min_clap: float = 0.30,
    max_abs_deviation: float = 0.20,
    min_ok_pct: float = 0.5,
) -> List[dict]:
    """Evaluate current eval results against thresholds; returns violations."""
    rows = load_results(cfg.project_root)
    if not rows:
        return []
    by_ckpt: Dict[str, List[dict]] = {}
    for r in rows:
        by_ckpt.setdefault(r.get("checkpoint") or "?", []).append(r)

    violations: List[dict] = []
    for ckpt, cres in sorted(by_ckpt.items()):
        agg = _aggregate(cres)
        if agg["mean_clap"] is not None and agg["mean_clap"] < min_clap:
            violations.append(
                {"checkpoint": ckpt, "metric": "mean_clap",
                 "value": round(agg["mean_clap"], 4), "limit": min_clap}
            )
        if agg["mean_abs_deviation"] is not None and agg["mean_abs_deviation"] > max_abs_deviation:
            violations.append(
                {"checkpoint": ckpt, "metric": "mean_abs_deviation",
                 "value": round(agg["mean_abs_deviation"], 4), "limit": max_abs_deviation}
            )
        if agg["ok_pct"] is not None and agg["ok_pct"] < min_ok_pct:
            violations.append(
                {"checkpoint": ckpt, "metric": "ok_pct",
                 "value": round(agg["ok_pct"], 4), "limit": min_ok_pct}
            )
    return violations


def _slack(webhook: str, text: str) -> bool:
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        webhook, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:  # noqa: BLE001
        console.warn(f"Slack webhook failed: {exc}")
        return False


def _email(host: str, port: int, user: str, password: str, to: str, subject: str, body: str) -> bool:
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to
        msg.set_content(body)
        with smtplib.SMTP(host, port, timeout=10) as s:
            s.starttls()
            s.login(user, password)
            s.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001
        console.warn(f"Email alert failed: {exc}")
        return False


def alert(
    cfg: Config,
    min_clap: float = 0.30,
    max_abs_deviation: float = 0.20,
    min_ok_pct: float = 0.5,
    slack_webhook: str = "",
    smtp_host: str = "",
    smtp_port: int = 587,
    smtp_user: str = "",
    smtp_password: str = "",
    smtp_to: str = "",
) -> dict:
    violations = check_alerts(cfg, min_clap, max_abs_deviation, min_ok_pct)
    result = {
        "violations": violations,
        "fired": False,
        "channels": [],
        "at": datetime.now(timezone.utc).isoformat(),
    }
    if not violations:
        console.ok("No threshold violations — nothing to alert.")
        return result

    lines = ["musictrain alert: eval threshold violation(s)"]
    lines += [
        f"- {v['checkpoint']}: {v['metric']} = {v['value']} (limit {v['limit']})"
        for v in violations
    ]
    text = "\n".join(lines)

    fired = False
    if slack_webhook:
        fired = _slack(slack_webhook, text) or fired
        result["channels"].append("slack")
    if smtp_host and smtp_user and smtp_to:
        fired = (
            _email(smtp_host, smtp_port, smtp_user, smtp_password, smtp_to,
                   "musictrain alert", text)
            or fired
        )
        result["channels"].append("email")

    # always write a local alert file (CI-pollable)
    alert_path = cfg.project_root / "metadata" / "alerts.jsonl"
    with alert_path.open("a") as fh:
        fh.write(json.dumps(result) + "\n")
    result["channels"].append("file")

    result["fired"] = True
    for v in violations:
        console.warn(
            f"ALERT {v['checkpoint']}: {v['metric']} = {v['value']} "
            f"(limit {v['limit']}) -> {result['channels']}"
        )
    return result
