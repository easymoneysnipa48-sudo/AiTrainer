"""Streamlit dashboard for the musictrain toolkit.

A live, tidy control surface: every page gets loading skeletons, widget keys,
and animated progress bars for long-running jobs, plus auto-refreshing
"live" fragments on the data pages.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import streamlit as st

from musictrain.config import Config

st.set_page_config(page_title="MusicTrain", page_icon="🎵", layout="wide")

ROOT = Path.cwd()

_THEME = """
<style>
  /* ---------- global ---------- */
  .stApp { background: linear-gradient(160deg, #0f1220 0%, #141a2e 55%, #0e1120 100%); }
  .stApp, [data-testid="stHeader"] { background: transparent; }
  html, body, [class*="css"], .stMarkdown, .stText, p, span, label {
    color: #e6e9f2;
  }
  [data-testid="stSidebar"] {
    background: rgba(20, 24, 40, 0.92);
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  [data-testid="stSidebar"] * { color: #dfe3ee; }
  h1, h2, h3 { color: #f2f4fb !important; letter-spacing: -0.01em; }

  /* ---------- metric cards ---------- */
  [data-testid="stMetric"] {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 16px;
    backdrop-filter: blur(6px);
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
  }
  [data-testid="stMetricLabel"] { color: #9aa3c0; }
  [data-testid="stMetricValue"] { color: #eef1fb; }
  [data-testid="stMetricDelta"] { color: #7ee2a8; }

  /* ---------- buttons ---------- */
  .stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.06);
    color: #eef1fb;
    transition: all .15s ease;
  }
  .stButton > button:hover { border-color: #6ea8ff; background: rgba(110,168,255,0.12); }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #5b8cff 0%, #7c5cff 100%);
    border: none; color: white; font-weight: 600;
  }
  .stButton > button[kind="primary"]:hover { filter: brightness(1.12); }

  /* ---------- inputs / frames ---------- */
  [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
  [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06); color: #eef1fb;
    border-radius: 10px;
  }
  [data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; overflow: hidden;
  }
  [data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; background: rgba(255,255,255,0.03);
  }
  [data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.025) !important;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: 8px 18px;
  }

  /* ---------- page header ---------- */
  .mt-header { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
  .mt-header .mt-emoji { font-size: 2.1rem; }
  .mt-header .mt-title { font-size: 1.65rem; font-weight: 700; color: #f2f4fb; letter-spacing: -0.02em; }
  .mt-caption { color: #9aa3c0; margin-bottom: 18px; }
</style>
"""


def _page_header(emoji: str, title: str, caption: str = "") -> None:
    st.markdown(
        f'<div class="mt-header"><span class="mt-emoji">{emoji}</span>'
        f'<span class="mt-title">{title}</span></div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(f'<div class="mt-caption">{caption}</div>', unsafe_allow_html=True)


def _skeleton(n: int = 3, height: int = 60) -> None:
    """Render n loading skeletons (real st.skeleton in the live app)."""
    for _ in range(n):
        st.skeleton(height=height)


def _run_job(label: str, fn: Callable, *args, **kwargs):
    """Run `fn` on a worker thread while animating a progress bar.

    Returns the callable's result; re-raises any exception. The progress bar
    advances to ~90% while the thread runs and snaps to 100% on completion —
    a live "work in progress" feel even though the jobs don't report stages.
    """
    out: dict = {}

    def _worker() -> None:
        try:
            out["result"] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            out["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    bar = st.progress(0.0, text=label)
    pct = 0.0
    while thread.is_alive():
        pct = min(pct + 0.03, 0.92)
        bar.progress(pct, text=label)
        time.sleep(0.12)
    thread.join(timeout=5)

    if "error" in out:
        bar.progress(1.0, text=f"{label} — failed")
        st.error(str(out["error"]))
        raise out["error"]
    bar.progress(1.0, text=f"{label} — done")
    return out.get("result")


def _read_json(rel: str):
    p = ROOT / "metadata" / rel
    return json.loads(p.read_text()) if p.exists() else None


def load_cfg() -> Config:
    p = ROOT / "configs" / "default.yaml"
    cfg = Config.load(p) if p.exists() else Config()
    cfg.project_root = ROOT
    return cfg


def _df_or_skeleton(rel: str, n: int = 3):
    """Read a metadata JSON as a DataFrame, showing skeletons while loading."""
    with st.spinner("Loading…"):
        data = _read_json(rel)
    if data is None:
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return pd.DataFrame(data)
    return None


# --------------------------------------------------------------------------- #
# 📋 Inventory
# --------------------------------------------------------------------------- #
def page_inventory() -> None:
    _page_header("📋", "Audio inventory", "Corpus files, validity, sample rates and durations.")
    inv = ROOT / "metadata" / "audio_inventory.json"
    if not inv.exists():
        _skeleton(3)
        st.warning("No inventory yet — run `musictrain inventory`.")
        return

    with st.spinner("Loading inventory…"):
        df = pd.read_json(inv)

    c1, c2, c3 = st.columns(3)
    c1.metric("Files", len(df))
    c2.metric("Valid", int(df["valid"].sum()) if "valid" in df else len(df))
    c3.metric(
        "Total duration (s)",
        f"{df['duration'].sum():,.0f}" if "duration" in df else "n/a",
    )

    t1, t2, t3 = st.tabs(["Sample rates", "Duration distribution", "Records"])
    with t1:
        if "sample_rate" in df:
            st.bar_chart(df["sample_rate"].value_counts().sort_index())
    with t2:
        if "duration" in df:
            import numpy as np

            hist, edges = np.histogram(df["duration"].dropna(), bins=20)
            hdf = pd.DataFrame({"bin_low_s": edges[:-1], "count": hist})
            st.bar_chart(hdf.set_index("bin_low_s")["count"])
    with t3:
        st.dataframe(df, width="stretch")


# --------------------------------------------------------------------------- #
# 🔧 Normalize
# --------------------------------------------------------------------------- #
def page_normalize() -> None:
    _page_header("🔧", "Normalize audio", "Converts data/raw/* to data/clean/* (mono, 32 kHz, PCM).")
    cfg = load_cfg()
    force = st.checkbox("Force overwrite", value=False, key="norm_force")

    if st.button("Run normalization", type="primary", key="norm_run"):
        from musictrain.audio.normalize import normalize
        from musictrain.paths import ensure_layout

        ensure_layout(ROOT)

        def _go():
            return normalize(ROOT, cfg, force=force)

        converted, skipped, failed = _run_job("Normalizing audio", _go)
        st.success(f"{converted} converted · {skipped} skipped · {failed} failed")


# --------------------------------------------------------------------------- #
# 🏷️ Metadata
# --------------------------------------------------------------------------- #
def page_features() -> None:
    _page_header("🏷️", "Extract metadata", "Detects BPM, key, loudness, silence, clipping; merges manual labels.")
    labels = st.text_input("Labels file (CSV/JSON, optional)", value="", key="feat_labels")
    limit = st.number_input("Limit (0 = all)", min_value=0, value=0, key="feat_limit")

    if st.button("Extract features", type="primary", key="feat_run"):
        from musictrain.metadata import extract

        def _go():
            return extract(
                ROOT, load_cfg(), which="clean",
                labels_path=Path(labels) if labels else None,
                limit=int(limit),
            )

        records = _run_job("Extracting features", _go)
        st.success(f"Processed {len(records)} files")
        if records:
            st.dataframe(pd.DataFrame(records), width="stretch")


# --------------------------------------------------------------------------- #
# ✂️ Segment & split
# --------------------------------------------------------------------------- #
def page_split() -> None:
    _page_header("✂️", "Segment & split", "Bar-aligned segmentation and train/val/test splitting.")
    cfg = load_cfg()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Segment")
        if st.button("Segment audio", type="primary", key="seg_run"):
            from musictrain.audio.segment import segment

            segs = _run_job("Segmenting audio", segment, ROOT, cfg)
            st.success(f"{len(segs)} segments written")
    with col2:
        st.subheader("Train/val/test split")
        st.write(
            f"Ratios: {cfg.split.train:.0%}/{cfg.split.val:.0%}/{cfg.split.test:.0%} "
            f"· seed {cfg.split.seed}"
        )
        if st.button("Run split", type="primary", key="split_run"):
            from musictrain.split import split

            _run_job("Splitting corpus", split, ROOT, cfg)
            st.success("Split complete")

    splits = ROOT / "metadata" / "splits.json"
    if splits.exists():
        data = json.loads(splits.read_text())
        rows = [
            {"split": k, "songs": v["songs"], "segments": v["segments"]}
            for k, v in data["splits"].items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")


# --------------------------------------------------------------------------- #
# 🎛️ Generate
# --------------------------------------------------------------------------- #
def page_generate() -> None:
    _page_header("🎛️", "Generate audio", "MusicGen on MPS — prompt, guidance, and sampling control.")
    cfg = load_cfg()
    prompt = st.text_area(
        "Prompt",
        value="cinematic hip hop chorus, 96 BPM, A minor, dark piano, "
        "deep 808 bass, wide strings, powerful drums",
        key="gen_prompt",
    )
    c1, c2, c3, c4 = st.columns(4)
    model = c1.selectbox(
        "Model",
        ["facebook/musicgen-small", "facebook/musicgen-medium", "facebook/musicgen-melody"],
        index=0, key="gen_model",
    )
    guidance = c2.slider("Guidance", 1.0, 10.0, 3.0, 0.5, key="gen_guidance")
    tokens = c3.slider("Max new tokens", 64, 1500, 256, 64, key="gen_tokens")
    seed = c4.number_input("Seed (0 = random)", min_value=0, value=0, key="gen_seed")

    if st.button("Generate", type="primary", key="gen_run"):
        from musictrain.inference import generate

        cfg.inference.model_name = model
        cfg.inference.guidance_scale = guidance
        cfg.inference.max_new_tokens = int(tokens)

        def _go():
            return generate(cfg, prompt, out_dir=ROOT / "outputs", seed=int(seed) or None)

        result = _run_job("Generating audio (MPS)", _go)
        st.success(f"Saved {result['path']} ({result['duration']}s, {result['device']})")
        st.audio(str(result["path"]))
        st.json(result)


# --------------------------------------------------------------------------- #
# 📏 Check BPM
# --------------------------------------------------------------------------- #
def page_check() -> None:
    _page_header("📏", "Check BPM", "Detect BPM drift of generated audio; optional time-stretch fix.")
    outputs = sorted((ROOT / "outputs").glob("*.wav")) if (ROOT / "outputs").exists() else []
    if not outputs:
        _skeleton(2, height=40)
        st.warning("No generated audio yet.")
        return

    names = [str(p.relative_to(ROOT)) for p in outputs]
    pick = st.selectbox("Audio file", names, key="chk_file")
    target = st.number_input("Target BPM (blank = just measure)", value=0.0, key="chk_target")
    fix = st.checkbox("Time-stretch to fix drift", value=False, key="chk_fix")

    if st.button("Check", type="primary", key="chk_run"):
        from musictrain.evaluate import check

        report = _run_job(
            "Checking BPM",
            check, load_cfg(), ROOT / pick,
            target_bpm=float(target) if target > 0 else None,
            fix=fix,
        )
        st.json(report)
        st.audio(str(ROOT / pick))
        if report.get("fixed_path"):
            st.audio(report["fixed_path"])


# --------------------------------------------------------------------------- #
# 🧪 stable verdicts (shared by Compare)
# --------------------------------------------------------------------------- #
@st.fragment(run_every=30)
def _stable_verdicts(cfg: Config) -> None:
    """Majority-vote verdicts from metadata/eval_results.jsonl (seeded runs)."""
    from musictrain.report import load_results

    rows = load_results(ROOT)
    if not rows:
        st.info("No eval results yet — run `musictrain eval` (ideally `--seeds 3`).")
        return

    df = pd.DataFrame(rows)
    n = len(df)
    ok = int((df["status"] == "ok").sum()) if "status" in df else 0
    pct = ok / n if n else 0.0

    n_seeds = (
        df["n_seeds"].fillna(1).astype(int)
        if "n_seeds" in df
        else pd.Series([1] * n, index=df.index)
    )
    seeded = int((n_seeds > 1).sum())

    devs = df["deviation"].dropna().abs() if "deviation" in df else pd.Series(dtype=float)
    mean_dev = float(devs.mean()) if len(devs) else 0.0
    claps = df["clap_score"].dropna() if "clap_score" in df else pd.Series(dtype=float)
    mean_clap = float(claps.mean()) if len(claps) else 0.0

    st.subheader("🧪 Stable verdicts (majority vote)")
    st.caption(
        "Aggregated over repeated seeds — the reliable baseline. The MLflow "
        "table/scatter below shows raw per-seed runs. Auto-refreshes every 30s."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Prompts", n)
    c2.metric("In-tolerance", f"{ok} ({pct:.0%})")
    c3.metric("Mean |deviation|", f"{mean_dev:.2%}")
    c4.metric("Mean CLAP", f"{mean_clap:.3f}")
    c5.metric("Multi-seed prompts", f"{seeded}/{n}")

    if "section" in df:
        by = (
            df.groupby("section")["status"]
            .agg(total="count", ok=lambda s: int((s == "ok").sum()))
            .reset_index()
        )
        by["in-tolerance"] = by.apply(lambda r: f"{r['ok']}/{r['total']}", axis=1)
        st.dataframe(by[["section", "in-tolerance"]], width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# 📊 Compare
# --------------------------------------------------------------------------- #
def page_compare() -> None:
    _page_header("📊", "MLflow run comparison", "Live per-seed runs vs stable majority verdicts.")
    cfg = load_cfg()

    with st.spinner("Loading stable verdicts…"):
        _stable_verdicts(cfg)
    st.markdown("---")

    from musictrain.experiments import search_runs

    with st.spinner("Querying MLflow…"):
        df = search_runs(cfg)
    if df is None or df.empty:
        st.warning("No MLflow runs yet — run `musictrain infer`, `eval`, or `features` first.")
        return

    tasks = sorted(x for x in df["task"].dropna().unique() if x)
    sel_task = st.multiselect("Task", tasks, default=tasks, key="cmp_tasks")
    view = df[df["task"].isin(sel_task)] if sel_task else df

    models = sorted(x for x in view["model"].dropna().unique() if x)
    if len(models) > 1:
        sel_model = st.selectbox("Checkpoint / model", ["(all)"] + models, key="cmp_model")
        if sel_model != "(all)":
            view = view[view["model"] == sel_model]

    st.subheader("Runs")
    st.dataframe(view, width="stretch")

    ev = view.dropna(subset=["target_bpm", "detected_bpm"])
    if not ev.empty:
        st.subheader("Detected vs target BPM")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(ev["target_bpm"], ev["detected_bpm"], alpha=0.7)
        lo = min(ev["target_bpm"].min(), ev["detected_bpm"].min())
        hi = max(ev["target_bpm"].max(), ev["detected_bpm"].max())
        ax.plot([lo, hi], [lo, hi], "k--", label="perfect adherence")
        ax.set_xlabel("target BPM")
        ax.set_ylabel("detected BPM")
        ax.legend()
        st.pyplot(fig)

        st.subheader("BPM adherence summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Runs", len(ev))
        c2.metric("In-tolerance", int((ev["verdict"] == "ok").sum()))
        mean_dev = ev["deviation"].abs().mean() if ev["deviation"].notna().any() else 0.0
        c3.metric("Mean |deviation|", f"{mean_dev:.2%}")


# --------------------------------------------------------------------------- #
# 🧹 Hygiene
# --------------------------------------------------------------------------- #
def page_hygiene() -> None:
    _page_header(
        "🧹", "Dataset hygiene",
        "Surfaces quality, dedup, corpus, and OOD results. Run the sweeps below "
        "or via the CLI (`musictrain quality|dedup|corpus|ood`).",
    )

    with st.spinner("Loading hygiene reports…"):
        quality = _read_json("quality_report.json")
        dedup = _read_json("duplicates.json")
        corpus = _read_json("corpus_stats.json")
        ood = _read_json("ood_tracks.json")

    q_flagged = (
        sum(1 for r in quality if r.get("grade") in ("C", "F")) if quality else 0
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Quality flagged (C/F)",
        f"{q_flagged}" if quality else "—",
        delta=None if not quality else f"{len(quality) - q_flagged} A/B",
    )
    c2.metric(
        "Duplicate groups",
        f"{dedup['duplicate_groups']}" if dedup else "—",
        f"{dedup['duplicate_files']} files" if dedup else None,
    )
    c3.metric("Corpus tracks", f"{corpus['n_tracks']}" if corpus else "—")
    c4.metric("OOD flagged", f"{len(ood)}" if ood is not None else "—")

    cfg = load_cfg()
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Run quality", use_container_width=True, key="hyg_quality"):
        from musictrain.audio.quality import quality as run_q

        _run_job("Scoring quality", run_q, ROOT, cfg)
        st.rerun()
    if b2.button("Run dedup", use_container_width=True, key="hyg_dedup"):
        from musictrain.dedup import find_duplicates

        _run_job("Finding duplicates", find_duplicates, ROOT, cfg)
        st.rerun()
    if b3.button("Run corpus", use_container_width=True, key="hyg_corpus"):
        from musictrain.corpus import corpus as run_c

        _run_job("Computing corpus stats", run_c, ROOT, cfg)
        st.rerun()
    if b4.button("Run OOD", use_container_width=True, key="hyg_ood"):
        from musictrain.ood import curate_ood

        _run_job("Flagging OOD tracks", curate_ood, ROOT, cfg)
        st.rerun()

    st.markdown("---")

    t_qual, t_dedup, t_corp, t_ood = st.tabs(["🔊 Quality", "👯 Duplicates", "📈 Corpus", "🚫 OOD"])

    with t_qual:
        st.subheader("Audio quality")
        if not quality:
            st.info("No quality report yet — run `musictrain quality`.")
        else:
            qdf = pd.DataFrame(quality)
            if "grade" in qdf:
                st.bar_chart(qdf["grade"].value_counts().reindex(["A", "B", "C", "F"], fill_value=0))
            show = qdf[~qdf["flags"].apply(lambda f: not f)] if "flags" in qdf else qdf
            st.dataframe(show, width="stretch", hide_index=True)

    with t_dedup:
        st.subheader("Duplicates")
        if not dedup:
            st.info("No duplicate report yet — run `musictrain dedup`.")
        else:
            st.caption(
                f"Scanned {dedup.get('scanned')} files · "
                f"{dedup.get('duplicate_files')} dupes in {dedup.get('duplicate_groups')} groups"
            )
            groups = dedup.get("groups", [])
            if groups:
                rows = [
                    {
                        "kind": g["kind"],
                        "canonical": g["canonical"],
                        "dupes": len(g["members"]) - 1,
                        "members": " · ".join(g["members"]),
                    }
                    for g in groups
                ]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.success("No duplicates found.")

    with t_corp:
        st.subheader("Corpus coverage")
        if not corpus:
            st.info("No corpus stats yet — run `musictrain features` then `musictrain corpus`.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Tracks", corpus.get("n_tracks", 0))
            c2.metric("Total duration", f"{corpus.get('total_duration_s', 0):,.0f}s")
            bpm = corpus.get("bpm") or {}
            c3.metric("BPM", bpm.get("mean") if bpm.get("mean") is not None else "—")

            hist = bpm.get("histogram") or {}
            if hist:
                st.caption("BPM histogram")
                st.bar_chart(pd.Series(hist))

            for dim in ("key", "genre", "mood", "instruments", "sections"):
                counts = corpus.get(dim) or {}
                if counts:
                    top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:12])
                    st.caption(f"{dim.capitalize()} (top 12)")
                    st.bar_chart(pd.Series(top))

    with t_ood:
        st.subheader("Off-distribution")
        if ood is None:
            st.info("No OOD report yet — run `musictrain ood`.")
        elif not ood:
            st.success("No OOD tracks flagged.")
        else:
            odf = pd.DataFrame(ood)
            cols = [c for c in ("path", "bpm", "key", "genre", "mood", "ood_reasons") if c in odf]
            st.dataframe(odf[cols], width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# 🏷️ Labels
# --------------------------------------------------------------------------- #
def page_labels() -> None:
    _page_header("🏷️", "Labels & vocabulary", "Vocabulary tree, label coverage, suggestions, agreement.")
    from musictrain.labels import VOCAB, VOCAB_VERSION
    from musictrain.vocab import render_tree

    labels_csv = ROOT / "metadata" / "labels.csv"
    with st.spinner("Loading labels…"):
        df = pd.read_csv(labels_csv) if labels_csv.exists() else None

    c1, c2 = st.columns(2)
    c1.metric("Vocab version", f"v{VOCAB_VERSION}")
    c2.metric("Labeled tracks", len(df) if df is not None else 0)

    with st.expander("🌳 Vocabulary tree", expanded=False):
        st.code(render_tree(), language=None)

    st.subheader("📈 Label coverage")
    if df is None:
        st.info("No labels.csv yet — run `musictrain labels` to scaffold one.")
    else:
        import csv as _csv

        for dim in ("genre", "mood", "instruments", "section"):
            used: dict = {}
            for cell in df[dim].dropna().astype(str):
                for token in _csv.reader([cell.replace("|", ",").replace(";", ",")]).__next__():
                    t = token.strip()
                    if t:
                        used[t] = used.get(t, 0) + 1
            vocab = VOCAB.get(dim, set())
            missing = sorted(vocab - set(used))
            st.caption(f"{dim} — {len(used)}/{len(vocab)} terms used")
            if used:
                st.bar_chart(pd.Series(dict(sorted(used.items(), key=lambda kv: -kv[1])[:12])))
            if missing:
                with st.expander(f"Unused {dim} terms ({len(missing)})"):
                    st.write(", ".join(missing))

    st.subheader("💡 Label suggestions")
    sug = _read_json("label_suggestions.json")
    if not sug:
        st.info("No suggestions yet — run `musictrain suggest --query <file>`.")
    else:
        st.caption(f"Query: {sug.get('query')}")
        for dim, props in (sug.get("vocab_proposals") or {}).items():
            if props:
                st.caption(dim)
                st.dataframe(pd.DataFrame(props), hide_index=True, width="stretch")
        neigh = sug.get("labeled_neighbors") or []
        if neigh:
            st.caption("Nearest labeled neighbors")
            rows = []
            for n in neigh:
                lab = n.get("labels") or {}
                rows.append(
                    {
                        "path": n["path"],
                        "similarity": n["similarity"],
                        "genre": lab.get("genre", ""),
                        "mood": lab.get("mood", ""),
                        "instruments": lab.get("instruments", ""),
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("🤝 Annotator agreement")
    agr = _read_json("agreement.json")
    if not agr:
        st.info("No agreement report yet — run `musictrain agree --a A.csv --b B.csv`.")
    else:
        ov = agr.get("overall", {})
        a1, a2, a3 = st.columns(3)
        a1.metric("Shared tracks", agr.get("shared_tracks", 0))
        a2.metric("Exact agreement", f"{ov.get('exact_agreement', 0):.0%}")
        a3.metric("Cohen's kappa", f"{ov.get('kappa', 0):+.2f}")
        fields = agr.get("fields", {})
        if fields:
            fdf = pd.DataFrame(fields).T.reset_index().rename(columns={"index": "field"})
            fdf["exact_agreement"] = (
                (fdf["exact_agreement"] * 100).round(0).astype(int).astype(str) + "%"
            )
            st.dataframe(fdf, hide_index=True, width="stretch")
        dis = agr.get("disagreements") or []
        if dis:
            with st.expander(f"Sample disagreements ({len(dis)})"):
                st.dataframe(pd.DataFrame(dis), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# 🏆 Leaderboard
# --------------------------------------------------------------------------- #
@st.fragment(run_every=30)
def _leaderboard_view(cfg: Config) -> None:
    """Auto-refreshing leaderboard read from metadata/leaderboard.json."""
    lb = _read_json("leaderboard.json")
    if not lb:
        st.info("No leaderboard yet — run `musictrain eval` then `musictrain leaderboard`.")
        return

    entries = lb.get("leaderboard", [])
    st.caption(f"{lb.get('n_checkpoints')} checkpoints compared · auto-refreshes every 30s")

    rows = [
        {
            "rank": e["rank"],
            "checkpoint": e["checkpoint"],
            "score": e["score"],
            "ok %": e["ok_pct"],
            "mean CLAP": e["mean_clap"],
            "mean |dev|": e["mean_abs_deviation"],
            "mean human rating": e["mean_human_rating"],
            "runs": e["runs"],
        }
        for e in entries
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Per-tag CLAP adherence (#46)")
    tag_rows = []
    for e in entries:
        for tag, val in (e.get("clap_per_tag") or {}).items():
            if val is not None:
                tag_rows.append({"checkpoint": e["checkpoint"], "tag": tag, "CLAP": val})
    if tag_rows:
        st.bar_chart(pd.DataFrame(tag_rows).pivot(index="tag", columns="checkpoint", values="CLAP"))
    else:
        st.info("No per-tag scores — run eval with `eval.per_tag_clap: true`.")


def page_leaderboard() -> None:
    _page_header("🏆", "Leaderboard", "Ranks checkpoints by adherence, BPM fidelity, verdict share, human rating.")
    cfg = load_cfg()

    if st.button("Rebuild leaderboard", type="primary", key="lb_rebuild"):
        from musictrain.leaderboard import build

        _run_job("Ranking checkpoints", build, cfg)
        st.rerun()

    with st.spinner("Loading leaderboard…"):
        _leaderboard_view(cfg)


# --------------------------------------------------------------------------- #
# 🎧 Listening
# --------------------------------------------------------------------------- #
def page_listening() -> None:
    _page_header("🎧", "Human listening", "Rate generated clips 1–5. Ratings merge into the leaderboard and report.")
    from musictrain.report import load_results

    with st.spinner("Loading eval results…"):
        rows = load_results(ROOT)
    if not rows:
        _skeleton(3)
        st.info("No eval results yet — run `musictrain eval` first.")
        return

    ratings_path = ROOT / "metadata" / "human_ratings.jsonl"
    ratings_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if ratings_path.exists():
        for ln in ratings_path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                existing[(r["prompt"], r["checkpoint"])] = r

    max_clips = min(len(rows), 50)
    if max_clips <= 1:
        limit, idx = 1, 0  # slider needs min < max, so skip it for a single clip
    else:
        limit = st.slider("Clips to review", 1, max_clips, min(max_clips, 10), key="lst_limit")
        idx = st.number_input("Start at #", 0, max(len(rows) - 1, 0), 0, step=1, key="lst_start")
    window = rows[int(idx): int(idx) + int(limit)]

    ratings = {}
    for i, r in enumerate(window):
        key = (r["prompt"], r["checkpoint"])
        prev = existing.get(key, {})
        with st.container(border=True):
            st.caption(
                f"**{r.get('section') or '?'}** · {r.get('bpm_target')} BPM · "
                f"CLAP {r.get('clap_score')} · dev {r.get('deviation')}"
            )
            st.write(r["prompt"])
            ap = r.get("audio_path")
            if ap and Path(ap).exists():
                st.audio(str(ap))
            else:
                st.warning("Audio file missing")
            c1, c2 = st.columns([1, 3])
            rating = c1.slider(
                f"Rating {i + 1}", 1, 5, int(prev.get("rating") or 3),
                key=f"rating_{i}", label_visibility="collapsed",
            )
            note = c2.text_input(
                "Note (optional)", value=prev.get("note", ""),
                key=f"note_{i}", label_visibility="collapsed",
            )
            ratings[key] = {"rating": rating, "note": note}

    if st.button("Save ratings", type="primary", key="lst_save"):
        saved = 0
        with ratings_path.open("a") as fh:
            for (prompt, checkpoint), rr in ratings.items():
                if rr["note"] or rr["rating"] != existing.get((prompt, checkpoint), {}).get("rating", 3):
                    fh.write(
                        json.dumps(
                            {
                                "prompt": prompt,
                                "checkpoint": checkpoint,
                                "rating": rr["rating"],
                                "note": rr["note"],
                            }
                        )
                        + "\n"
                    )
                    saved += 1
        st.success(f"Saved {saved} rating(s) -> metadata/human_ratings.jsonl")
        st.rerun()

    st.caption(f"Already rated: {len(existing)} prompt/checkpoint pairs")


# --------------------------------------------------------------------------- #
# 🪄 Prompt builder
# --------------------------------------------------------------------------- #
def page_promptbuilder() -> None:
    _page_header(
        "🪄", "Prompt builder",
        "Pick controlled-vocabulary tags; the prompt is assembled in the same "
        "shape as the training labels and eval set.",
    )
    from musictrain.labels import VOCAB
    from musictrain.promptbuilder import build_prompt

    cfg = load_cfg()
    c1, c2 = st.columns(2)
    with c1:
        section = st.selectbox("Section", sorted(VOCAB["section"]), index=3, key="pb_section")
        genre = st.selectbox("Genre", sorted(VOCAB["genre"]), index=0, key="pb_genre")
        bpm = st.number_input("BPM", min_value=40.0, max_value=220.0, value=140.0, step=1.0, key="pb_bpm")
        key = st.text_input("Key", value="A minor", key="pb_key")
    with c2:
        mood = st.multiselect("Mood", sorted(VOCAB["mood"]), default=["dark", "emotional"], key="pb_mood")
        instruments = st.multiselect(
            "Instruments",
            sorted(VOCAB["instruments"]),
            default=["piano", "808 bass", "trap hi-hats"],
            key="pb_instruments",
        )
        energy = st.slider("Energy", 0.0, 1.0, 0.7, 0.05, key="pb_energy")
        role = st.text_input("Narrative role", value="", key="pb_role")

    assembled = build_prompt(
        section=section, genre=genre, mood=mood, instruments=instruments,
        bpm=bpm, key=key, energy=energy, role=role or None,
    )
    prompt = st.text_area("Prompt", value=assembled, height=90, key="pb_prompt")
    st.code(prompt, language=None)

    if st.button("Generate with MusicGen", type="primary", key="pb_gen"):
        from musictrain.inference import generate

        result = _run_job("Generating audio (MPS)", generate, cfg, prompt, out_dir=ROOT / "outputs")
        st.success(f"Saved {result['path']} ({result['duration']}s)")
        st.audio(str(result["path"]))


PAGES = {
    "📋 Inventory": page_inventory,
    "🔧 Normalize": page_normalize,
    "🏷️ Metadata": page_features,
    "✂️ Segment & Split": page_split,
    "🎛️ Generate": page_generate,
    "🪄 Prompt builder": page_promptbuilder,
    "📏 Check BPM": page_check,
    "🏷️ Labels": page_labels,
    "📊 Compare": page_compare,
    "🧹 Hygiene": page_hygiene,
    "🏆 Leaderboard": page_leaderboard,
    "🎧 Listening": page_listening,
}


def main() -> None:
    st.markdown(_THEME, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 🎵 MusicTrain")
        st.caption(f"Project: `{ROOT.name}`")
        st.caption(f"Checkpoints: {len(list((ROOT / 'outputs').glob('*.wav')))} clips" if (ROOT / "outputs").exists() else "")
        st.markdown("---")
        choice = st.radio("Go to", list(PAGES.keys()), key="nav")
    PAGES[choice]()


main()
