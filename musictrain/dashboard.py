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


def page_compare() -> None:
    st.header("📊 MLflow run comparison")
    cfg = load_cfg()
    from musictrain.experiments import search_runs

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


PAGES = {
    "📋 Inventory": page_inventory,
    "🔧 Normalize": page_normalize,
    "🏷️ Metadata": page_features,
    "✂️ Segment & Split": page_split,
    "🎛️ Generate": page_generate,
    "📏 Check BPM": page_check,
    "📊 Compare": page_compare,
}


def main() -> None:
    st.sidebar.title("🎵 MusicTrain")
    st.sidebar.caption(f"Project: {ROOT.name}")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[choice]()


main()
