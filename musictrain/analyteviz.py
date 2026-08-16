"""Analytics visualization helpers for the dashboard (batch 5, #41-50)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:  # noqa: BLE001
        return None


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


def _pca2d(matrix):
    import numpy as np

    X = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:2].T


# --------------------------------------------------------------------------- #
# 41 — embedding scatter (PCA 2-D of the CLAP cache)
# --------------------------------------------------------------------------- #
def umap_scatter(cfg) -> None:
    import altair as alt
    import numpy as np

    rep = _read_json(cfg.project_root / "metadata" / "audio_embeddings.json")
    if not rep or len(rep) < 3:
        st.caption("not enough embedded audio yet — run `musictrain embed` first")
        return
    keys = list(rep)
    X = np.stack([np.asarray(rep[k], dtype=float) for k in keys])
    coords = _pca2d(X)
    pdf = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "path": keys})
    chart = (
        alt.Chart(pdf)
        .mark_point(size=90, color="#5b8cff")
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
                tooltip=["path"])
        .properties(height=240)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(f"{len(keys)} tracks projected to 2-D (PCA of CLAP embeddings)")


# --------------------------------------------------------------------------- #
# 42 — active-learning scatter (uncertainty vs diversity)
# --------------------------------------------------------------------------- #
def active_scatter(cfg) -> None:
    import altair as alt

    rows = _read_json(cfg.project_root / "metadata" / "active_learning.json")
    if isinstance(rows, dict):
        rows = rows.get("ranking") or rows.get("rows") or []
    if not rows:
        st.caption("no active-learning ranking yet — run `musictrain active` first")
        return
    pdf = pd.DataFrame(rows)
    if "uncertainty" not in pdf or "diversity" not in pdf:
        st.caption("active-learning rows missing uncertainty/diversity")
        return
    chart = (
        alt.Chart(pdf)
        .mark_point(size=90, color="#ffb020")
        .encode(x=alt.X("uncertainty:Q"), y=alt.Y("diversity:Q"),
                tooltip=["path", "uncertainty", "diversity"])
        .properties(height=240)
    )
    st.altair_chart(chart, width="stretch")
    st.caption("top-right = highest labeling priority (uncertain AND diverse)")


# --------------------------------------------------------------------------- #
# 43 — augmentation coverage panel
# --------------------------------------------------------------------------- #
_AUG_OPS = [
    ("pitch", "semitone transposition"),
    ("stretch", "tempo time-stretch"),
    ("noise", "additive noise floor"),
    ("eq", "EQ band shaping"),
    ("quiet", "gain reduction"),
]


def augmentation_panel() -> None:
    c = st.columns(len(_AUG_OPS))
    for col, (op, desc) in zip(c, _AUG_OPS):
        with col:
            st.markdown(
                f"<div style='background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);"
                f"border-radius:12px;padding:12px;text-align:center'>"
                f"<div style='font-weight:600;color:#eef1fb'>{op}</div>"
                f"<div style='color:#9aa3c0;font-size:.78rem'>{desc}</div></div>",
                unsafe_allow_html=True,
            )
    st.caption("augmentation operators available via `musictrain augment --ops pitch,stretch,noise,eq,quiet`")


# --------------------------------------------------------------------------- #
# 44 — leaderboard bar (sorted by score)
# --------------------------------------------------------------------------- #
def leaderboard_bar(cfg) -> None:
    import altair as alt

    rep = _read_json(cfg.project_root / "metadata" / "leaderboard.json")
    rows = (rep or {}).get("leaderboard") or []
    if not rows:
        st.caption("no leaderboard yet — run `musictrain leaderboard` first")
        return
    pdf = pd.DataFrame(rows)
    pdf["short"] = pdf["checkpoint"].str.split("/").str[-1]
    chart = (
        alt.Chart(pdf)
        .mark_bar(color="#7c5cff")
        .encode(x=alt.X("score:Q", title="leaderboard score"),
                y=alt.Y("short:N", sort="-x"),
                tooltip=["checkpoint", "score", "ok_pct", "mean_clap"])
        .properties(height=max(120, 32 * len(pdf)))
    )
    st.altair_chart(chart, width="stretch")


# --------------------------------------------------------------------------- #
# 45 — curation histogram + top gallery
# --------------------------------------------------------------------------- #
def curation_histogram(cfg) -> None:
    import altair as alt

    rows = _read_json(cfg.project_root / "metadata" / "curation_scores.json")
    if isinstance(rows, dict):
        rows = rows.get("scores") or rows.get("tracks") or []
    if not rows:
        st.caption("no curation scores yet — run `musictrain curation` first")
        return
    pdf = pd.DataFrame(rows)
    score_col = "score" if "score" in pdf else "curation_score"
    if score_col not in pdf:
        st.caption("curation rows missing a score column")
        return
    if pdf[score_col].apply(lambda v: pd.notna(v) and pd.api.types.is_number(v)).sum() < 2:
        st.caption("curation scores have no numeric values to histogram")
        return
    hist = (
        alt.Chart(pdf)
        .mark_bar(color="#2ad4c4")
        .encode(x=alt.X(f"{score_col}:Q", bin=alt.Bin(maxbins=20)),
                y=alt.Y("count()"))
        .properties(height=180)
    )
    st.altair_chart(hist, width="stretch")
    top = pdf.sort_values(score_col, ascending=False).head(10)
    st.dataframe(top, width="stretch")


# --------------------------------------------------------------------------- #
# 46 — checkpoint registry timeline
# --------------------------------------------------------------------------- #
def checkpoint_timeline(cfg) -> None:
    rep = _read_json(cfg.project_root / "metadata" / "checkpoint_registry.json")
    entries = (rep or {}).get("checkpoints") or []
    if not entries:
        st.caption("no registered checkpoints — run `musictrain registry` first")
        return
    for e in entries:
        name = e.get("name", "?")
        size = e.get("size_mb")
        params = e.get("n_params_est")
        modified = (e.get("modified_at") or "")[:19]
        left = f"**{name}**"
        right = f"{size} MB · params ~{params}" if params else f"{size} MB"
        c1, c2 = st.columns([3, 1])
        c1.markdown(left)
        c2.caption(f"{right} · {modified}")


# --------------------------------------------------------------------------- #
# 47 — two-run overlay (mean CLAP vs mean |dev| per checkpoint)
# --------------------------------------------------------------------------- #
def two_run_overlay(cfg) -> None:
    import altair as alt

    rows = _read_jsonl(cfg.project_root / "metadata" / "eval_results.jsonl")
    if not rows:
        st.caption("no eval results yet")
        return
    by_ckpt: dict = {}
    for r in rows:
        ck = (r.get("checkpoint") or "?").split("/")[-1]
        agg = by_ckpt.setdefault(ck, {"clap": [], "dev": []})
        if r.get("clap_score") is not None:
            agg["clap"].append(float(r["clap_score"]))
        if r.get("deviation") is not None:
            agg["dev"].append(abs(float(r["deviation"])))
    pts = []
    for ck, agg in by_ckpt.items():
        if agg["clap"] and agg["dev"]:
            pts.append({"checkpoint": ck,
                        "mean_clap": sum(agg["clap"]) / len(agg["clap"]),
                        "mean_abs_dev": sum(agg["dev"]) / len(agg["dev"])})
    if len(pts) < 2:
        st.caption("need >=2 checkpoints in eval results for an overlay")
        return
    pdf = pd.DataFrame(pts)
    chart = (
        alt.Chart(pdf)
        .mark_point(size=140, filled=True)
        .encode(x=alt.X("mean_clap:Q", title="mean CLAP"),
                y=alt.Y("mean_abs_dev:Q", title="mean |deviation|"),
                color=alt.Color("checkpoint:N"),
                tooltip=["checkpoint", "mean_clap", "mean_abs_dev"])
        .properties(height=240)
    )
    st.altair_chart(chart, width="stretch")


# --------------------------------------------------------------------------- #
# 48 — early-stop curve (patience window shaded)
# --------------------------------------------------------------------------- #
def early_stop_curve() -> None:
    import altair as alt
    import numpy as np

    # illustrative: metric improves then plateaus; the patience window is shaded
    steps = np.arange(0, 40)
    metric = 1.0 - 0.65 * np.exp(-steps / 8.0) + 0.02 * np.sin(steps / 1.7)
    pdf = pd.DataFrame({"step": steps, "metric": metric})
    line = alt.Chart(pdf).mark_line(color="#5b8cff", point=True).encode(
        x=alt.X("step:Q"), y=alt.Y("metric:Q", scale=alt.Scale(zero=False)))
    # patience-window boundaries as dashed rules (a rect with no y encoding
    # produced an infinite y extent and Vega "scale bindings" warnings).
    band = alt.Chart(pd.DataFrame({"x": [24.0, 39.0]})).mark_rule(
        strokeWidth=1.5, color="#ffb020", opacity=0.6, strokeDash=[5, 4]).encode(
        x=alt.X("x:Q"))
    chart = (line + band).properties(height=180)
    st.altair_chart(chart, width="stretch")
    st.caption("illustration — early-stop patience window marked by dashed rules "
               "(live curve streams from MLflow during fine-tune)")


# --------------------------------------------------------------------------- #
# 49 — token usage gauge
# --------------------------------------------------------------------------- #
def token_gauge(cfg) -> None:
    rows = _read_jsonl(cfg.project_root / "metadata" / "cost_log.jsonl")
    tokens = int(sum(r.get("tokens", 0) for r in rows)) if rows else 0
    if not tokens:
        st.caption("no token counts logged yet — run `musictrain cost` to record usage")
        return
    c1, c2 = st.columns([1, 2])
    c1.metric("tokens generated", f"{tokens:,}")
    c2.caption("token usage is logged to metadata/cost_log.jsonl alongside kWh estimates")


# --------------------------------------------------------------------------- #
# 50 — model size vs compute cost
# --------------------------------------------------------------------------- #
def model_size_cost(cfg) -> None:
    import altair as alt

    rows = _read_jsonl(cfg.project_root / "metadata" / "cost_log.jsonl")
    if not rows:
        st.caption("no cost records yet — run `musictrain cost` to compare model cost")
        return
    pdf = pd.DataFrame(rows)
    if "model_name" not in pdf or "estimated_kwh" not in pdf:
        st.caption("cost records missing model_name/estimated_kwh")
        return
    agg = pdf.groupby("model_name")["estimated_kwh"].sum().reset_index()
    chart = (
        alt.Chart(agg)
        .mark_bar(color="#ff5c8a")
        .encode(x=alt.X("estimated_kwh:Q", title="estimated kWh"),
                y=alt.Y("model_name:N", sort="-x"))
        .properties(height=max(120, 28 * agg["model_name"].nunique()))
    )
    st.altair_chart(chart, width="stretch")
