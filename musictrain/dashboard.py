"""Streamlit dashboard for the musictrain toolkit."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from musictrain.config import Config

st.set_page_config(page_title="MusicTrain", page_icon="🎵", layout="wide")

ROOT = Path.cwd()


def load_cfg() -> Config:
    p = ROOT / "configs" / "default.yaml"
    cfg = Config.load(p) if p.exists() else Config()
    cfg.project_root = ROOT
    return cfg


# --------------------------------------------------------------------------- #
def page_inventory() -> None:
    st.header("📋 Audio inventory")
    inv = ROOT / "metadata" / "audio_inventory.json"
    if not inv.exists():
        st.warning("No inventory yet — run `musictrain inventory`.")
        return

    df = pd.read_json(inv)
    valid = df[df.get("valid", True)] if "valid" in df else df
    c1, c2, c3 = st.columns(3)
    c1.metric("Files", len(df))
    c2.metric("Valid", int(df["valid"].sum()) if "valid" in df else len(df))
    c3.metric(
        "Total duration (s)",
        f"{df['duration'].sum():,.0f}" if "duration" in df else "n/a",
    )

    st.subheader("Sample rates")
    if "sample_rate" in df:
        st.bar_chart(df["sample_rate"].value_counts().sort_index())

    st.subheader("Duration distribution")
    if "duration" in df:
        hist, edges = pd.cut(df["duration"], bins=20, retbins=True)
        st.bar_chart(hist.value_counts().sort_index())

    st.subheader("Records")
    st.dataframe(df, use_container_width=True)


def page_normalize() -> None:
    st.header("🔧 Normalize audio")
    st.caption("Converts data/raw/* to data/clean/* (mono, 32 kHz, PCM).")
    cfg = load_cfg()
    force = st.checkbox("Force overwrite", value=False)

    if st.button("Run normalization", type="primary"):
        from musictrain.audio.normalize import normalize
        from musictrain.paths import ensure_layout

        ensure_layout(ROOT)
        with st.spinner("Normalizing…"):
            converted, skipped, failed = normalize(ROOT, cfg, force=force)
        st.success(f"{converted} converted · {skipped} skipped · {failed} failed")


def page_features() -> None:
    st.header("🏷️ Extract metadata")
    st.caption("Detects BPM, key, loudness, silence, and clipping; merges manual labels.")
    labels = st.text_input("Labels file (CSV/JSON, optional)", value="")
    limit = st.number_input("Limit (0 = all)", min_value=0, value=0)

    if st.button("Extract features", type="primary"):
        from musictrain.metadata import extract

        with st.spinner("Extracting features…"):
            records = extract(
                ROOT,
                load_cfg(),
                which="clean",
                labels_path=Path(labels) if labels else None,
                limit=int(limit),
            )
        st.success(f"Processed {len(records)} files")
        if records:
            st.dataframe(pd.DataFrame(records), use_container_width=True)


def page_split() -> None:
    st.header("✂️ Segment & split")
    cfg = load_cfg()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Segment")
        if st.button("Segment audio", type="primary"):
            from musictrain.audio.segment import segment

            with st.spinner("Segmenting…"):
                segs = segment(ROOT, cfg)
            st.success(f"{len(segs)} segments written")
    with col2:
        st.subheader("Train/val/test split")
        st.write(
            f"Ratios: {cfg.split.train:.0%}/{cfg.split.val:.0%}/{cfg.split.test:.0%} "
            f"· seed {cfg.split.seed}"
        )
        if st.button("Run split", type="primary"):
            from musictrain.split import split

            with st.spinner("Splitting…"):
                split(ROOT, cfg)
            st.success("Split complete")

    splits = ROOT / "metadata" / "splits.json"
    if splits.exists():
        data = json.loads(splits.read_text())
        rows = [
            {"split": k, "songs": v["songs"], "segments": v["segments"]}
            for k, v in data["splits"].items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def page_generate() -> None:
    st.header("🎛️ Generate audio (MusicGen on MPS)")
    cfg = load_cfg()
    prompt = st.text_area(
        "Prompt",
        value="cinematic hip hop chorus, 96 BPM, A minor, dark piano, "
        "deep 808 bass, wide strings, powerful drums",
    )
    c1, c2, c3, c4 = st.columns(4)
    model = c1.selectbox(
        "Model",
        ["facebook/musicgen-small", "facebook/musicgen-medium", "facebook/musicgen-melody"],
        index=0,
    )
    guidance = c2.slider("Guidance", 1.0, 10.0, 3.0, 0.5)
    tokens = c3.slider("Max new tokens", 64, 1500, 256, 64)
    seed = c4.number_input("Seed (0 = random)", min_value=0, value=0)

    if st.button("Generate", type="primary"):
        from musictrain.inference import generate

        cfg.inference.model_name = model
        cfg.inference.guidance_scale = guidance
        cfg.inference.max_new_tokens = int(tokens)
        with st.spinner("Generating (this can take a minute on MPS)…"):
            result = generate(
                cfg, prompt, out_dir=ROOT / "outputs", seed=int(seed) or None
            )
        st.success(f"Saved {result['path']} ({result['duration']}s, {result['device']})")
        st.audio(str(result["path"]))
        st.json(result)


def page_check() -> None:
    st.header("📏 Check BPM")
    outputs = sorted((ROOT / "outputs").glob("*.wav")) if (ROOT / "outputs").exists() else []
    if not outputs:
        st.warning("No generated audio yet.")
        return

    names = [str(p.relative_to(ROOT)) for p in outputs]
    pick = st.selectbox("Audio file", names)
    target = st.number_input("Target BPM (blank = just measure)", value=0.0)
    fix = st.checkbox("Time-stretch to fix drift", value=False)

    if st.button("Check", type="primary"):
        from musictrain.evaluate import check

        report = check(
            load_cfg(),
            ROOT / pick,
            target_bpm=float(target) if target > 0 else None,
            fix=fix,
        )
        st.json(report)
        st.audio(str(ROOT / pick))
        if report.get("fixed_path"):
            st.audio(report["fixed_path"])


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
        "table/scatter below shows raw per-seed runs."
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
        st.dataframe(
            by[["section", "in-tolerance"]], use_container_width=True, hide_index=True
        )


def page_compare() -> None:
    st.header("📊 MLflow run comparison")
    cfg = load_cfg()
    from musictrain.experiments import search_runs

    _stable_verdicts(cfg)
    st.markdown("---")

    df = search_runs(cfg)
    if df is None or df.empty:
        st.warning("No MLflow runs yet — run `musictrain infer`, `eval`, or `features` first.")
        return

    tasks = sorted(x for x in df["task"].dropna().unique() if x)
    sel_task = st.multiselect("Task", tasks, default=tasks)
    view = df[df["task"].isin(sel_task)] if sel_task else df

    models = sorted(x for x in view["model"].dropna().unique() if x)
    if len(models) > 1:
        sel_model = st.selectbox("Checkpoint / model", ["(all)"] + models)
        if sel_model != "(all)":
            view = view[view["model"] == sel_model]

    st.subheader("Runs")
    st.dataframe(view, use_container_width=True)

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


def _read_json(rel: str):
    p = ROOT / "metadata" / rel
    return json.loads(p.read_text()) if p.exists() else None


def page_hygiene() -> None:
    st.header("🧹 Dataset hygiene")
    st.caption(
        "Surfaces quality, dedup, corpus, and OOD results. Run the sweeps below "
        "or via the CLI (`musictrain quality|dedup|corpus|ood`)."
    )

    quality = _read_json("quality_report.json")
    dedup = _read_json("duplicates.json")
    corpus = _read_json("corpus_stats.json")
    ood = _read_json("ood_tracks.json")

    # -- summary cards -------------------------------------------------------
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

    # -- run buttons ---------------------------------------------------------
    cfg = load_cfg()
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Run quality", use_container_width=True):
        from musictrain.audio.quality import quality as run_q

        with st.spinner("Scoring quality…"):
            run_q(ROOT, cfg)
        st.rerun()
    if b2.button("Run dedup", use_container_width=True):
        from musictrain.dedup import find_duplicates

        with st.spinner("Finding duplicates…"):
            find_duplicates(ROOT, cfg)
        st.rerun()
    if b3.button("Run corpus", use_container_width=True):
        from musictrain.corpus import corpus as run_c

        with st.spinner("Computing corpus stats…"):
            run_c(ROOT, cfg)
        st.rerun()
    if b4.button("Run OOD", use_container_width=True):
        from musictrain.ood import curate_ood

        with st.spinner("Flagging OOD tracks…"):
            curate_ood(ROOT, cfg)
        st.rerun()

    st.markdown("---")

    # -- quality -------------------------------------------------------------
    st.subheader("🔊 Audio quality")
    if not quality:
        st.info("No quality report yet — run `musictrain quality`.")
    else:
        qdf = pd.DataFrame(quality)
        if "grade" in qdf:
            st.bar_chart(qdf["grade"].value_counts().reindex(["A", "B", "C", "F"], fill_value=0))
        show = qdf[~qdf["flags"].apply(lambda f: not f)] if "flags" in qdf else qdf
        st.dataframe(show, use_container_width=True, hide_index=True)

    # -- dedup ---------------------------------------------------------------
    st.subheader("👯 Duplicates")
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.success("No duplicates found.")

    # -- corpus ---------------------------------------------------------------
    st.subheader("📈 Corpus coverage")
    if not corpus:
        st.info("No corpus stats yet — run `musictrain features` then `musictrain corpus`.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Tracks", corpus.get("n_tracks", 0))
        c2.metric(
            "Total duration",
            f"{corpus.get('total_duration_s', 0):,.0f}s",
        )
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

    # -- OOD -----------------------------------------------------------------
    st.subheader("🚫 Off-distribution")
    if ood is None:
        st.info("No OOD report yet — run `musictrain ood`.")
    elif not ood:
        st.success("No OOD tracks flagged.")
    else:
        odf = pd.DataFrame(ood)
        cols = [c for c in ("path", "bpm", "key", "genre", "mood", "ood_reasons") if c in odf]
        st.dataframe(odf[cols], use_container_width=True, hide_index=True)


PAGES = {
    "📋 Inventory": page_inventory,
    "🔧 Normalize": page_normalize,
    "🏷️ Metadata": page_features,
    "✂️ Segment & Split": page_split,
    "🎛️ Generate": page_generate,
    "📏 Check BPM": page_check,
    "📊 Compare": page_compare,
    "🧹 Hygiene": page_hygiene,
}


def main() -> None:
    st.sidebar.title("🎵 MusicTrain")
    st.sidebar.caption(f"Project: {ROOT.name}")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[choice]()


main()
