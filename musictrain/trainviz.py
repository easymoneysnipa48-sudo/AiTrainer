"""Training-focused visualization helpers for the dashboard (batch 4, #31-40).

Every helper degrades to a caption when its data source is missing, so the
Training page renders even on a fresh checkout with no MLflow runs, no cost
log, and no drift report yet.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:  # noqa: BLE001
        return None


def mlflow_runs(cfg) -> pd.DataFrame:
    from musictrain.experiments import search_runs

    return search_runs(cfg)


# --------------------------------------------------------------------------- #
# 33 — training HUD (run / eval / CLAP / |dev|)
# --------------------------------------------------------------------------- #
def training_hud(cfg) -> None:
    df = mlflow_runs(cfg)
    if df.empty:
        c = st.columns(4)
        c[0].metric("MLflow runs", 0)
        c[1].metric("eval rows", 0)
        c[2].metric("mean CLAP", "—")
        c[3].metric("mean |dev|", "—")
        return
    ev = df[df["task"] == "eval"] if "task" in df else df
    mean_clap = float(ev["clap_score"].mean()) if "clap_score" in ev and ev["clap_score"].notna().any() else None
    mean_dev = float(ev["deviation"].abs().mean()) if "deviation" in ev and ev["deviation"].notna().any() else None
    c = st.columns(4)
    c[0].metric("MLflow runs", len(df))
    c[1].metric("eval rows", int((df["task"] == "eval").sum()))
    c[2].metric("mean CLAP", f"{mean_clap:.3f}" if mean_clap is not None else "—")
    c[3].metric("mean |dev|", f"{mean_dev:.3f}" if mean_dev is not None else "—")


# --------------------------------------------------------------------------- #
# 31 — CLAP trend over eval runs
# --------------------------------------------------------------------------- #
def clap_trend(cfg) -> None:
    import altair as alt

    df = mlflow_runs(cfg)
    if df.empty or "clap_score" not in df:
        st.caption("no MLflow runs with CLAP scores yet")
        return
    d = df[df["task"] == "eval"].dropna(subset=["clap_score"]).copy()
    if d.empty:
        st.caption("no eval runs with CLAP scores yet")
        return
    d = d.sort_values("name").reset_index(drop=True)
    d["i"] = range(len(d))
    chart = (
        alt.Chart(d)
        .mark_line(point=True, color="#7c5cff")
        .encode(x=alt.X("i:Q", title="eval run (chronological)"),
                y=alt.Y("clap_score:Q", scale=alt.Scale(zero=False)),
                tooltip=["run_id", "name", "clap_score", "verdict"])
        .properties(height=180)
    )
    st.altair_chart(chart, width="stretch")


# --------------------------------------------------------------------------- #
# 32 — recent-run metrics panel
# --------------------------------------------------------------------------- #
def metrics_panel(cfg, n: int = 12) -> None:
    df = mlflow_runs(cfg)
    if df.empty:
        st.caption("no MLflow runs to show")
        return
    cols = ["run_id", "task", "model", "device", "clap_score", "deviation", "duration_s", "verdict"]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df[cols].head(n), width="stretch")


# --------------------------------------------------------------------------- #
# 37 — experiment matrix heatmap (runs x numeric metrics)
# --------------------------------------------------------------------------- #
def matrix_heatmap(cfg, n: int = 30) -> None:
    import altair as alt

    df = mlflow_runs(cfg)
    if df.empty:
        st.caption("no runs for the matrix")
        return
    metrics = [m for m in ("clap_score", "deviation", "duration_s", "bpm_mean") if m in df.columns]
    d = df[df["task"] == "eval"].head(n)
    if d.empty or not metrics:
        st.caption("no eval runs / numeric metrics for the matrix")
        return
    rows = []
    for _, r in d.iterrows():
        for m in metrics:
            v = r[m]
            if pd.notna(v):
                rows.append({"run": r["run_id"], "metric": m, "value": float(v)})
    if not rows:
        st.caption("no numeric metric values")
        return
    pdf = pd.DataFrame(rows)
    chart = (
        alt.Chart(pdf)
        .mark_rect()
        .encode(x=alt.X("metric:N"), y=alt.Y("run:N", sort=None),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis")),
                tooltip=["run", "metric", "value"])
        .properties(height=max(200, 12 * pdf["run"].nunique()))
    )
    st.altair_chart(chart, width="stretch")


# --------------------------------------------------------------------------- #
# 36 — cost chart from cost_log.jsonl
# --------------------------------------------------------------------------- #
def cost_chart(cfg) -> None:
    import altair as alt

    rows = _read_jsonl(cfg.project_root / "metadata" / "cost_log.jsonl")
    if not rows:
        st.caption("no cost records yet — run `musictrain cost` to log estimates")
        return
    pdf = pd.DataFrame(rows)
    if "estimated_kwh" not in pdf:
        st.caption("cost records missing estimated_kwh")
        return
    agg = pdf.groupby("task")["estimated_kwh"].sum().reset_index()
    chart = (
        alt.Chart(agg)
        .mark_bar(color="#5b8cff")
        .encode(x=alt.X("estimated_kwh:Q", title="estimated kWh"),
                y=alt.Y("task:N", sort="-x"))
        .properties(height=max(120, 24 * agg["task"].nunique()))
    )
    st.altair_chart(chart, width="stretch")
    st.caption(f"total estimated ~{agg['estimated_kwh'].sum():.4f} kWh across {len(pdf)} logged runs")


# --------------------------------------------------------------------------- #
# 39 — coverage heatmap (genre x mood from labels.csv)
# --------------------------------------------------------------------------- #
def coverage_heatmap(cfg) -> None:
    import altair as alt

    csv = cfg.project_root / "metadata" / "labels.csv"
    if not csv.exists():
        st.caption("no labels.csv — run `musictrain labels` first")
        return
    df = pd.read_csv(csv)
    if df.empty or "genre" not in df or "mood" not in df:
        st.caption("labels.csv missing genre/mood columns")
        return
    df["mood"] = df["mood"].fillna("unknown").astype(str)
    exploded = df.assign(mood=df["mood"].str.split("|")).explode("mood")
    counts = exploded.groupby(["genre", "mood"]).size().reset_index(name="n")
    chart = (
        alt.Chart(counts)
        .mark_rect()
        .encode(x=alt.X("mood:N", sort=None), y=alt.Y("genre:N", sort=None),
                color=alt.Color("n:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["genre", "mood", "n"])
        .properties(height=max(140, 24 * counts["genre"].nunique()))
    )
    st.altair_chart(chart, width="stretch")


# --------------------------------------------------------------------------- #
# 40 — train/val/test split donut
# --------------------------------------------------------------------------- #
def split_donut(cfg) -> None:
    import matplotlib.pyplot as plt

    root = cfg.project_root / "data"
    counts = {}
    for split in ("train", "val", "test"):
        d = root / split
        counts[split] = len(list(d.glob("*.wav"))) + len(list(d.glob("*.mp3"))) if d.exists() else 0
    total = sum(counts.values())
    if total == 0:
        st.caption("no split dirs yet — run `musictrain split` first")
        return
    labels = [k for k, v in counts.items() if v]
    vals = [v for k, v in counts.items() if v]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(vals, labels=labels, autopct="%1.0f%%", startangle=90,
           colors=["#5b8cff", "#7c5cff", "#ff5c8a"], textprops={"color": "#e6e9f2"})
    ax.axis("equal")
    st.pyplot(fig)


# --------------------------------------------------------------------------- #
# 34 — LR schedule reference curve
# --------------------------------------------------------------------------- #
def lr_schedule() -> None:
    import altair as alt
    import numpy as np

    steps = np.arange(0, 5001)
    lr = 1e-4 * (0.5 + 0.5 * np.cos(np.pi * steps / steps[-1]))
    pdf = pd.DataFrame({"step": steps, "lr": lr})
    chart = (
        alt.Chart(pdf)
        .mark_line(color="#ffb020")
        .encode(x=alt.X("step:Q"), y=alt.Y("lr:Q", scale=alt.Scale(zero=False)))
        .properties(height=160)
    )
    st.altair_chart(chart, width="stretch")
    st.caption("reference cosine-decay schedule used by fine-tune")


# --------------------------------------------------------------------------- #
# 38 — drift timeline (metadata/drift.json)
# --------------------------------------------------------------------------- #
def drift_timeline(cfg) -> None:
    rep = _read_json(cfg.project_root / "metadata" / "drift.json")
    if not rep:
        st.caption("no drift report — run `musictrain drift` first")
        return
    cont = rep.get("continuous") or {}
    rows = [
        {"feature": feat, "reference": d.get("reference_mean"), "current": d.get("current_mean"),
         "delta": d.get("delta"), "drifted": bool(d.get("drifted"))}
        for feat, d in cont.items()
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch")
    drifted = rep.get("drifted_features") or []
    if drifted:
        st.warning(f"drifted features: {', '.join(drifted)}")
    else:
        st.success(f"no features drifted ({rep.get('reference')} vs {rep.get('current')})")


# --------------------------------------------------------------------------- #
# 35 — weight-delta heatmap (metadata/weight_diff.json)
# --------------------------------------------------------------------------- #
def weight_diff_heatmap(cfg) -> None:
    import altair as alt

    rep = _read_json(cfg.project_root / "metadata" / "weight_diff.json")
    if not rep:
        st.caption("no weight-diff report — run `musictrain diff` first")
        return
    deltas = rep.get("largest_deltas") or rep.get("deltas") or []
    if not deltas:
        st.caption("weight-diff report has no tensor deltas")
        return
    pdf = pd.DataFrame(deltas)
    name_col = "tensor" if "tensor" in pdf else "name"
    if name_col in pdf and "max_abs_delta" in pdf:
        chart = (
            alt.Chart(pdf.head(20))
            .mark_bar(color="#ff5c8a")
            .encode(x=alt.X("max_abs_delta:Q"),
                    y=alt.Y(f"{name_col}:N", sort="-x"))
            .properties(height=max(120, 20 * len(pdf.head(20))))
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.dataframe(pdf, width="stretch")
