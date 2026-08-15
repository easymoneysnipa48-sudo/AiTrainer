"""Visualization helpers for the dashboard — batch 1 (previews & audio).

Feature map (70-feature list):
  1  waveform thumbnails
  2  mel + chroma spectrogram
  4  before/after waveform overlay
  5  stem-preview mixer
  6  section timeline scrubber
  7  live generation view (auto-refresh fragment)
  8  chord / beat-grid strip
  9  piano-roll chromagram
 10  onset + downbeat marker overlay

Helpers are defensive: every function degrades to a caption instead of raising,
so the dashboard can render with an empty / partial corpus.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

AUDIO_GLOB = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def scan_audio(root: Path, dirs: Iterable[str]) -> list[Path]:
    """Collect audio files under a set of project-relative directories."""
    out: list[Path] = []
    seen: set[str] = set()
    for rel in dirs:
        base = root / rel
        if not base.exists():
            continue
        for pattern in AUDIO_GLOB:
            for p in sorted(base.glob(pattern)):
                key = str(p)
                if key not in seen:
                    seen.add(key)
                    out.append(p)
    return out


def _load_env(path: str, sr: int = 8000) -> Optional[pd.DataFrame]:
    """Load a mono envelope for fast altair charting (t seconds -> amplitude)."""
    import librosa
    import numpy as np

    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        if len(y) == 0:
            return None
        hop = max(1, len(y) // 240)
        env = np.abs(y[::hop])
        n = len(env)
        xs = np.linspace(0.0, len(y) / sr, n)
        return pd.DataFrame({"t": xs, "amp": env})
    except Exception:  # noqa: BLE001
        return None


def _load_wave(path: str, sr: int = 16000):
    """Full-rate mono waveform + sample rate for matplotlib rendering."""
    import librosa

    return librosa.load(str(path), sr=sr, mono=True)


# --------------------------------------------------------------------------- #
# 1 — waveform thumbnails
# --------------------------------------------------------------------------- #
def waveform(path: str, key: str, height: int = 60, color: str = "#5b8cff") -> None:
    import altair as alt

    pdf = _load_env(path)
    if pdf is None:
        st.caption("waveform unavailable")
        return
    chart = (
        alt.Chart(pdf)
        .mark_area(opacity=0.5, color=color)
        .encode(x=alt.X("t:Q", axis=None, title="seconds"),
                y=alt.Y("amp:Q", axis=None))
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch", key=f"wav_{key}")


# --------------------------------------------------------------------------- #
# 2 — mel + chroma spectrogram
# --------------------------------------------------------------------------- #
def spectrogram(path: str, key: str, kind: str = "mel") -> None:
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        y, sr = _load_wave(path)
        if kind == "chroma":
            S = librosa.feature.chroma_cqt(y=y, sr=sr)
            ylabel = "pitch class"
        else:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            ylabel = "mel"
        fig, ax = plt.subplots(figsize=(9, 2.6))
        img = librosa.display.specshow(
            librosa.power_to_db(S, ref=np.max),
            sr=sr, x_axis="time", y_axis=ylabel, ax=ax,
        )
        fig.colorbar(img, ax=ax, fraction=0.025)
        ax.set_title(Path(path).name)
        st.pyplot(fig)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"spectrogram unavailable: {exc}")


# --------------------------------------------------------------------------- #
# 4 — before / after waveform overlay
# --------------------------------------------------------------------------- #
def before_after(path_a: str, path_b: str, key: str,
                 labels: tuple = ("raw", "normalized")) -> None:
    import altair as alt

    a = _load_env(path_a)
    b = _load_env(path_b)
    if a is None or b is None:
        st.caption("one or both clips failed to load for the overlay")
        return
    a = a.assign(series=labels[0])
    b = b.assign(series=labels[1])
    pdf = pd.concat([a, b], ignore_index=True)
    chart = (
        alt.Chart(pdf)
        .mark_line(opacity=0.8, strokeWidth=1.2)
        .encode(x=alt.X("t:Q", axis=None, title="seconds"),
                y=alt.Y("amp:Q", axis=None),
                color=alt.Color("series:N", legend=alt.Legend(orient="top")))
        .properties(height=110)
    )
    st.altair_chart(chart, width="stretch", key=f"ba_{key}")


# --------------------------------------------------------------------------- #
# 5 — stem-preview mixer
# --------------------------------------------------------------------------- #
def stem_mixer(stem_files: list, key: str) -> None:
    import librosa
    import numpy as np

    if not stem_files:
        st.caption("no stems available (run demucs separation first)")
        return

    sr = 32000
    gains = []
    for f in stem_files:
        gains.append(st.slider(f"{Path(f).stem}", 0, 200, 100, step=5,
                               key=f"gain_{key}_{Path(f).stem}"))

    waves = []
    for f, g in zip(stem_files, gains):
        if g <= 0:
            continue
        try:
            y, _ = librosa.load(str(f), sr=sr, mono=True)
        except Exception:  # noqa: BLE001
            continue
        waves.append(y * (g / 100.0))

    if waves:
        mix = np.zeros(max(len(w) for w in waves), dtype=np.float32)
        for w in waves:
            mix[: len(w)] += w
        peak = float(np.max(np.abs(mix)) or 1.0)
        mix = mix / peak * 0.9
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, mix, sr)
            st.audio(tmp.name, format="audio/wav")

    for f in stem_files:
        st.caption(Path(f).name)
        st.audio(str(f))


# --------------------------------------------------------------------------- #
# 6 — section timeline scrubber
# --------------------------------------------------------------------------- #
def structure_timeline(audio_path: str, segments: list, key: str) -> None:
    import altair as alt

    if not segments:
        st.caption("no structure detected")
        return
    rows = [{"role": s.get("role", s.get("label", "?")),
             "start": float(s["start"]), "end": float(s["end"])} for s in segments]
    pdf = pd.DataFrame(rows)
    palette = {
        "intro": "#5b8cff", "verse": "#7c5cff", "chorus": "#ff5c8a",
        "bridge": "#2ad4c4", "pre-chorus": "#ffb020", "outro": "#8a5cff",
        "full-song": "#6f7f9f",
    }
    chart = (
        alt.Chart(pdf)
        .mark_bar(size=26)
        .encode(
            x=alt.X("start:Q", title="seconds"),
            x2=alt.X2("end:Q"),
            y=alt.Y("role:N", axis=None),
            color=alt.Color("role:N", scale=alt.Scale(domain=list(palette),
                                                      range=list(palette.values())),
                            legend=alt.Legend(orient="top")),
            tooltip=["role", "start", "end"],
        )
        .properties(height=90)
    )
    st.altair_chart(chart, width="stretch", key=f"struct_{key}")

    for i, s in enumerate(segments):
        role = s.get("role", "?")
        with st.expander(f"{i + 1}. {role}  ({s['start']:.1f}s – {s['end']:.1f}s)"):
            st.audio(audio_path, start_time=float(s["start"]),
                     end_time=float(s["end"]))


# --------------------------------------------------------------------------- #
# 7 — live generation view (auto-refresh fragment)
# --------------------------------------------------------------------------- #
@st.fragment(run_every=2.0)
def live_generation_view(outputs_dir: str, key: str) -> None:
    base = Path(outputs_dir)
    files = []
    if base.exists():
        for pattern in AUDIO_GLOB:
            files.extend(base.glob(pattern))
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    if not files:
        st.caption("watching for generated clips…")
        return
    newest = files[0]
    import time as _t

    c = st.columns([1, 3])
    with c[0]:
        st.metric("latest", newest.name[:24], f"{len(files)} clips")
        st.caption(_t.strftime("%H:%M:%S", _t.localtime(newest.stat().st_mtime)))
    with c[1]:
        waveform(str(newest), f"live_{key}", height=50)
        spectrogram(str(newest), f"live_{key}", kind="mel")
    st.audio(str(newest))


# --------------------------------------------------------------------------- #
# 8 — chord / beat-grid strip
# --------------------------------------------------------------------------- #
def chord_beat_strip(chords: list, beat_grid: dict, key: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not chords and not beat_grid:
        st.caption("no chord / beat data")
        return
    fig, ax = plt.subplots(figsize=(9, 1.8))
    ax.axhline(0, color="#333")

    # beat grid: ticks
    beats = beat_grid.get("beat_times") or []
    downs = beat_grid.get("downbeat_times") or []
    for t in beats:
        ax.plot([t, t], [-0.4, 0.4], color="#5b8cff", lw=0.6, alpha=0.5)
    for t in downs:
        ax.plot([t, t], [-1, 1], color="#ff5c8a", lw=1.4)

    # chord labels
    for ch in chords:
        t = float(ch.get("t", 0.0))
        name = ch.get("chord", "?")
        ax.text(t, 0.55, name, rotation=90, fontsize=7, ha="center",
                color="#e6e9f2", alpha=0.85)

    ax.set_yticks([])
    ax.set_xlabel("seconds")
    ax.set_title("beat grid (blue = beat, red = downbeat) + chords")
    ax.set_ylim(-1.2, 1.2)
    if beats:
        ax.set_xlim(0, max(beats) + 1)
    st.pyplot(fig)


# --------------------------------------------------------------------------- #
# 9 — piano-roll chromagram
# --------------------------------------------------------------------------- #
def chromagram(path: str, key: str) -> None:
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        y, sr = _load_wave(path)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        fig, ax = plt.subplots(figsize=(9, 2.8))
        img = librosa.display.specshow(chroma, sr=sr, x_axis="time",
                                       y_axis="chroma", ax=ax, cmap="magma")
        fig.colorbar(img, ax=ax, fraction=0.025)
        ax.set_title(f"{Path(path).name} — chromagram")
        st.pyplot(fig)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"chromagram unavailable: {exc}")


# --------------------------------------------------------------------------- #
# 10 — onset + downbeat marker overlay
# --------------------------------------------------------------------------- #
def onset_overlay(path: str, beat_grid: dict, onsets: dict, key: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        y, sr = _load_wave(path, sr=8000)
        t = np.linspace(0, len(y) / sr, len(y))
        fig, ax = plt.subplots(figsize=(9, 2.4))
        ax.plot(t, y, color="#9aa3c0", lw=0.5, alpha=0.9)
        for d in (beat_grid.get("downbeat_times") or []):
            ax.axvline(d, color="#ff5c8a", lw=1.4, alpha=0.8)
        # onset density overlays as translucent vertical ticks
        if onsets.get("n_onsets"):
            ax.text(0.99, 0.92, f"onsets/s: {onsets.get('onset_density', '?')}",
                    transform=ax.transAxes, ha="right", color="#e6e9f2", fontsize=9)
        ax.set_title(f"{Path(path).name} — waveform + downbeats (red)")
        ax.set_xlabel("seconds")
        st.pyplot(fig)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"onset overlay unavailable: {exc}")


# --------------------------------------------------------------------------- #
# 11 — per-tag CLAP heat strip
# --------------------------------------------------------------------------- #
def clap_heat_strip(rows: list, key: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in rows if r.get("clap_per_tag")]
    if not rows:
        st.caption("no per-tag CLAP data in the eval results")
        return
    tags = sorted({t for r in rows for t in r["clap_per_tag"]})
    if not tags:
        st.caption("empty clap_per_tag maps")
        return
    means = [float(np.mean([r["clap_per_tag"].get(t, 0.0) for r in rows])) for t in tags]
    fig, ax = plt.subplots(figsize=(9, 1.6))
    X = np.array(means).reshape(1, -1)
    im = ax.imshow(X, cmap="RdYlGn", vmin=-0.1, vmax=0.6, aspect="auto")
    for i, (t, m) in enumerate(zip(tags, means)):
        ax.text(i, 0, f"{t}\n{m:.2f}", ha="center", va="center",
                fontsize=8, color="#141414", weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("mean per-tag CLAP similarity (green = on-prompt)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    st.pyplot(fig)


# --------------------------------------------------------------------------- #
# 12 — zoomable waveform with segment boundaries + role labels
# --------------------------------------------------------------------------- #
_SEG_COLORS = {
    "intro": "#5b8cff", "verse": "#7c5cff", "chorus": "#ff5c8a",
    "bridge": "#2ad4c4", "pre-chorus": "#ffb020", "outro": "#8a5cff",
    "full-song": "#6f7f9f",
}


def segmented_waveform(path: str, segments: list, key: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        y, sr = _load_wave(path, sr=8000)
        t = np.linspace(0, len(y) / sr, len(y))
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.plot(t, y, color="#9aa3c0", lw=0.5, alpha=0.9)
        for s in segments:
            role = s.get("role", "?")
            a, b = float(s["start"]), float(s["end"])
            ax.axvspan(a, b, color=_SEG_COLORS.get(role, "#6f7f9f"), alpha=0.16)
            ax.text((a + b) / 2, float(np.max(y)) * 0.92, role, ha="center",
                    fontsize=8, color="#e6e9f2")
        ax.set_title(f"{Path(path).name} — detected sections")
        ax.set_xlabel("seconds")
        st.pyplot(fig)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"segmented waveform unavailable: {exc}")


# --------------------------------------------------------------------------- #
# 13 — side-by-side augmented-variant previews
# --------------------------------------------------------------------------- #
def variant_grid(files: list, key: str, cols: int = 3) -> None:
    if not files:
        st.caption("no variant clips to preview")
        return
    for i in range(0, len(files), cols):
        row = files[i:i + cols]
        cs = st.columns(cols)
        for j, (c, f) in enumerate(zip(cs, row)):
            with c:
                st.caption(Path(f).name[:30])
                waveform(str(f), f"{key}_{i + j}", height=40)
                st.audio(str(f))


# --------------------------------------------------------------------------- #
# 15 — waveform miniatures for the prompt gallery
# --------------------------------------------------------------------------- #
def gallery_miniatures(pairs: list, key: str, cols: int = 4) -> None:
    """pairs: list of (title, audio_path). Renders card-like miniatures."""
    for i in range(0, len(pairs), cols):
        cs = st.columns(cols)
        for j, (c, item) in enumerate(zip(cs, pairs[i:i + cols])):
            title, path = item
            with c:
                st.markdown(f"**{title[:32]}**")
                waveform(str(path), f"{key}_{i + j}", height=36)
                st.audio(str(path))


# --------------------------------------------------------------------------- #
# 17 — skeleton placeholder block
# --------------------------------------------------------------------------- #
def skeleton(n: int = 3, height: int = 60) -> None:
    for _ in range(n):
        st.skeleton(height=height)


# --------------------------------------------------------------------------- #
# 18 — animated count-up metric
# --------------------------------------------------------------------------- #
def countup_metric(label: str, value: float, key: str,
                   suffix: str = "", decimals: int | None = None) -> None:
    import streamlit.components.v1 as components

    if decimals is None:
        decimals = 0 if float(value).is_integer() else 2
    clean = str(key).replace("-", "_").replace(" ", "_")
    js = f"""
    <div style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);
      border-radius:12px;padding:12px 16px;text-align:center">
      <div style="font-size:.72rem;color:#9aa3c0;letter-spacing:.04em;text-transform:uppercase">{label}</div>
      <div id="cu_{clean}" style="font-size:1.7rem;font-weight:700;color:#eef1fb">0</div>
      <script>
        (function(){{
          var el=document.getElementById('cu_{clean}');
          var target={value}; var t0=null;
          function step(ts){{
            if(!t0) t0=ts;
            var p=Math.min(1,(ts-t0)/900);
            var eased=1-Math.pow(1-p,3);
            el.textContent=(target*eased).toFixed({decimals})+"{suffix}";
            if(p<1) requestAnimationFrame(step);
          }}
          requestAnimationFrame(step);
        }})();
      </script>
    </div>
    """
    components.html(js, height=78)


# --------------------------------------------------------------------------- #
# 20 — pulsing "live" dot
# --------------------------------------------------------------------------- #
def live_dot(text: str = "live") -> None:
    st.markdown(
        f"""
        <style>
        @keyframes mtpulse {{ 0% {{opacity:.25;transform:scale(.8)}}
          50% {{opacity:1;transform:scale(1.25)}} 100% {{opacity:.25;transform:scale(.8)}} }}
        .mt-live-dot {{ display:inline-block;width:9px;height:9px;border-radius:50%;
          background:#7ee2a8;margin-right:6px;animation:mtpulse 1.6s ease-in-out infinite; }}
        </style>
        <div><span class="mt-live-dot"></span><span style="color:#7ee2a8">{text}</span></div>
        """,
        unsafe_allow_html=True,
    )
