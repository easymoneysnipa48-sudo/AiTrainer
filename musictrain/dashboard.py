"""Streamlit dashboard for the musictrain toolkit.

A live, tidy control surface: loading skeletons, widget keys, animated
progress bars, dark/light themes, command palette + keyboard shortcuts,
breadcrumb history, global search, and auto-refreshing fragments.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from musictrain.config import Config

st.set_page_config(page_title="MusicTrain", page_icon="🎵", layout="wide")

ROOT = Path.cwd()


# feature 37 — curated prompt templates (section/energy/BPM angle)
_TEMPLATES = [
    {"name": "Sparse intro", "section": "intro", "energy": "low",
     "prompt": "sparse cinematic intro, 70 BPM, A minor, dark piano loop, airy pads, low energy, wide reverb"},
    {"name": "Trap verse", "section": "verse", "energy": "mid",
     "prompt": "melodic trap verse, 140 BPM, B minor, rolling 808 bass, trap hi-hats, pluck melody, aggressive"},
    {"name": "Melodic chorus", "section": "chorus", "energy": "high",
     "prompt": "melodic trap chorus, 96 BPM, A minor, dark piano, deep 808 bass, wide strings, powerful drums, emotional"},
    {"name": "Emotional pre-chorus", "section": "pre-chorus", "energy": "mid",
     "prompt": "emotional pre-chorus, 84 BPM, F minor, warm pads, soft piano, build tension, atmospheric"},
    {"name": "Bridge breakdown", "section": "bridge", "energy": "low",
     "prompt": "bridge breakdown, 90 BPM, C minor, stripped drums, ambient pads, melancholic, reflective"},
    {"name": "Outro fade", "section": "outro", "energy": "low",
     "prompt": "outro fade, 72 BPM, A minor, piano and pads fading out, dark, spacious, low energy"},
    {"name": "Orchestral intro", "section": "intro", "energy": "low",
     "prompt": "orchestral intro, 60 BPM, D minor, strings swells, timpani roll, cinematic, epic"},
    {"name": "Full-song demo", "section": "full-song", "energy": "high",
     "prompt": "full trap song demo, 140 BPM, E minor, intro verse chorus structure, 808 bass, trap hi-hats, dark piano, emotional"},
]


# --------------------------------------------------------------------------- #
# Theme (dark / light) — feature 1
# --------------------------------------------------------------------------- #
_DARK_CSS = """
<style>
  .stApp { background: linear-gradient(160deg, #0f1220 0%, #141a2e 55%, #0e1120 100%); }
  .stApp, [data-testid="stHeader"] { background: transparent; }
  html, body, [class*="css"], .stMarkdown, .stText, p, span, label { color: #e6e9f2; }
  [data-testid="stSidebar"] { background: rgba(20, 24, 40, 0.92); border-right: 1px solid rgba(255,255,255,0.06); }
  [data-testid="stSidebar"] * { color: #dfe3ee; }
  h1, h2, h3 { color: #f2f4fb !important; letter-spacing: -0.01em; }
  [data-testid="stMetric"] { background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 14px 16px; backdrop-filter: blur(6px); box-shadow: 0 2px 10px rgba(0,0,0,0.18); }
  [data-testid="stMetricLabel"] { color: #9aa3c0; }
  [data-testid="stMetricValue"] { color: #eef1fb; }
  [data-testid="stMetricDelta"] { color: #7ee2a8; }
  .stButton > button, .stDownloadButton > button { border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.06);
    color: #eef1fb; transition: all .15s ease; }
  .stButton > button:hover { border-color: #6ea8ff; background: rgba(110,168,255,0.12); }
  .stButton > button[kind="primary"] { background: linear-gradient(135deg, #5b8cff 0%, #7c5cff 100%);
    border: none; color: white; font-weight: 600; }
  .stButton > button[kind="primary"]:hover { filter: brightness(1.12); }
  [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
  [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06); color: #eef1fb; border-radius: 10px; }
  [data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; }
  [data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; background: rgba(255,255,255,0.03); }
  [data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important; background: rgba(255,255,255,0.025) !important; }
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 18px; }
  .mt-header { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; flex-wrap: wrap; }
  .mt-header .mt-emoji { font-size: 1.9rem; }
  .mt-header .mt-title { font-size: 1.6rem; font-weight: 700; color: #f2f4fb; letter-spacing: -0.02em; }
  .mt-caption { color: #9aa3c0; margin-bottom: 18px; }
  .mt-crumbs { color: #7d86a8; font-size: .8rem; margin-bottom: 10px; }
  .mt-crumbs a { color: #9db4ff; text-decoration: none; }
  .mt-crumbs a:hover { text-decoration: underline; }
  .mt-git { display: inline-block; font-family: ui-monospace, monospace; font-size: .72rem;
    color: #9db4ff; border: 1px solid rgba(157,180,255,.35); border-radius: 8px; padding: 2px 8px; }
  .mt-git.dirty { color: #ffd479; border-color: rgba(255,212,121,.4); }
  .mt-quicknav { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 14px; }
  .mt-quicknav .mt-qn { flex: 1; min-width: 90px; font-size: .78rem; padding: 6px 4px; }
  .mt-search-hit { font-size: .78rem; color: #aeb6d4; padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,.04); }
  .mt-search-hit b { color: #eef1fb; }

  /* responsive — feature 8 */
  @media (max-width: 720px) {
    .mt-header { gap: 8px; }
    .mt-header .mt-title { font-size: 1.25rem; }
    .mt-quicknav .mt-qn { min-width: 100%; }
    [data-testid="stMetric"] { padding: 10px 12px; }
  }
</style>
"""

_LIGHT_CSS = """
<style>
  .stApp { background: linear-gradient(160deg, #f4f6fb 0%, #eef1f8 55%, #f7f8fc 100%); }
  .stApp, [data-testid="stHeader"] { background: transparent; }
  html, body, [class*="css"], .stMarkdown, .stText, p, span, label { color: #1c2333; }
  [data-testid="stSidebar"] { background: rgba(255,255,255,0.85); border-right: 1px solid rgba(0,0,0,0.06); }
  [data-testid="stSidebar"] * { color: #2a3247; }
  h1, h2, h3 { color: #111827 !important; }
  [data-testid="stMetric"] { background: rgba(255,255,255,0.9); border: 1px solid rgba(0,0,0,0.07);
    border-radius: 14px; padding: 14px 16px; box-shadow: 0 2px 10px rgba(15,23,42,0.06); }
  [data-testid="stMetricLabel"] { color: #5b6478; }
  [data-testid="stMetricValue"] { color: #111827; }
  [data-testid="stMetricDelta"] { color: #0d7a3f; }
  .stButton > button, .stDownloadButton > button { border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.12); background: rgba(255,255,255,0.8);
    color: #1c2333; transition: all .15s ease; }
  .stButton > button:hover { border-color: #3b82f6; background: rgba(59,130,246,0.08); }
  .stButton > button[kind="primary"] { background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    border: none; color: white; font-weight: 600; }
  [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
  [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.9); color: #1c2333; border-radius: 10px; }
  [data-testid="stDataFrame"] { border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; overflow: hidden; }
  [data-testid="stExpander"] { border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; background: rgba(255,255,255,0.6); }
  [data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid rgba(0,0,0,0.07) !important;
    border-radius: 14px !important; background: rgba(255,255,255,0.7) !important; }
  .mt-header { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; flex-wrap: wrap; }
  .mt-header .mt-emoji { font-size: 1.9rem; }
  .mt-header .mt-title { font-size: 1.6rem; font-weight: 700; color: #111827; letter-spacing: -0.02em; }
  .mt-caption { color: #5b6478; margin-bottom: 18px; }
  .mt-crumbs { color: #6b7280; font-size: .8rem; margin-bottom: 10px; }
  .mt-crumbs a { color: #2563eb; text-decoration: none; }
  .mt-git { display: inline-block; font-family: ui-monospace, monospace; font-size: .72rem;
    color: #2563eb; border: 1px solid rgba(37,99,235,.35); border-radius: 8px; padding: 2px 8px; }
  .mt-git.dirty { color: #b45309; border-color: rgba(180,83,9,.4); }
  .mt-quicknav { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 14px; }
  .mt-quicknav .mt-qn { flex: 1; min-width: 90px; font-size: .78rem; padding: 6px 4px; }
  .mt-search-hit { font-size: .78rem; color: #4b5563; padding: 3px 0; border-bottom: 1px solid rgba(0,0,0,0.04); }
  .mt-search-hit b { color: #111827; }

  @media (max-width: 720px) {
    .mt-header .mt-title { font-size: 1.25rem; }
    .mt-quicknav .mt-qn { min-width: 100%; }
  }
</style>
"""


def _theme_css() -> str:
    light = st.session_state.get("mt_theme") == "light"
    return _LIGHT_CSS if light else _DARK_CSS


def _toggle_theme() -> None:
    """Dark/light theme toggle — feature 1 (persisted in session state)."""
    light = st.session_state.get("mt_theme") == "light"
    if st.toggle("☀️ Light mode", value=light, key="theme_toggle"):
        st.session_state["mt_theme"] = "light"
    else:
        st.session_state["mt_theme"] = "dark"


# --------------------------------------------------------------------------- #
# git status header — feature 10
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=60, show_spinner=False)
def _git_info() -> tuple:
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        )
        return head, dirty
    except Exception:  # noqa: BLE001 - no git, no problem
        return "", False


# --------------------------------------------------------------------------- #
# page header + breadcrumbs — feature 4 & 10
# --------------------------------------------------------------------------- #
def _page_header(emoji: str, title: str, caption: str = "") -> None:
    head, dirty = _git_info()
    git_chip = f'<span class="mt-git{" dirty" if dirty else ""}>{head or "no-git"}{"*" if dirty else ""}</span>' if head or dirty else ""
    st.markdown(
        f'<div class="mt-header"><span class="mt-emoji">{emoji}</span>'
        f'<span class="mt-title">{title}</span>{git_chip}</div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(f'<div class="mt-caption">{caption}</div>', unsafe_allow_html=True)


def _crumbs(history: list) -> None:
    """Breadcrumb trail of previously visited pages — feature 4."""
    if len(history) > 1:
        trail = "  /  ".join(html_escape(p) for p in history[-4:])
        st.markdown(f'<div class="mt-crumbs">↩ {trail}</div>', unsafe_allow_html=True)


def html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# command palette + keyboard shortcuts — features 5 & 6
# --------------------------------------------------------------------------- #
_PALETTE_HTML = """
<div id="mt-palette" style="display:none;position:fixed;top:72px;left:50%;transform:translateX(-50%);
  width:540px;max-width:92vw;z-index:99999;background:#141a2e;border:1px solid rgba(255,255,255,.16);
  border-radius:14px;padding:12px;box-shadow:0 14px 44px rgba(0,0,0,.55);">
  <input id="mt-palette-input" placeholder="Jump to a page…  (Ctrl/⌘+K to open · Esc to close)"
    style="width:100%;padding:11px;border-radius:9px;border:1px solid rgba(255,255,255,.22);
    background:#0f1220;color:#eef1fb;font-size:1rem;">
  <div id="mt-palette-results" style="margin-top:8px;max-height:320px;overflow:auto;"></div>
</div>
<script>
(function () {
  var PAGES = ["📋 Inventory","🔧 Normalize","🏷️ Metadata","✂️ Segment & Split","🎛️ Generate",
    "🪄 Prompt builder","📏 Check BPM","🏷️ Labels","📊 Compare","🧹 Hygiene","🏆 Leaderboard","🎧 Listening"];
  var SHORTCUTS = {g:"🎛️ Generate", l:"🏆 Leaderboard", c:"📊 Compare", h:"🧹 Hygiene",
    i:"📋 Inventory", n:"🔧 Normalize", m:"🏷️ Metadata", b:"📏 Check BPM"};
  var box = document.getElementById("mt-palette");
  var input = document.getElementById("mt-palette-input");
  var results = document.getElementById("mt-palette-results");
  var visible = false;

  function post(label) {
    window.parent.postMessage({type: "streamlit:setComponentValue", value: label}, "*");
  }
  function render(filter) {
    var q = (filter || "").toLowerCase();
    var hits = PAGES.filter(function (p) { return p.toLowerCase().indexOf(q) !== -1; });
    results.innerHTML = hits.map(function (p) {
      return '<div class="mt-pal-item" style="padding:9px 12px;border-radius:9px;cursor:pointer;color:#eef1fb">'
        + p + '</div>';
    }).join("");
    var items = results.querySelectorAll(".mt-pal-item");
    items.forEach(function (el) {
      el.addEventListener("click", function () { post(el.textContent); });
    });
  }
  function toggle(show) {
    visible = show; box.style.display = show ? "block" : "none";
    if (show) { input.value = ""; render(""); input.focus(); }
  }
  document.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); toggle(!visible); return; }
    if (e.key === "Escape") { toggle(false); return; }
    var tag = (e.target && e.target.tagName) || "";
    if (visible) { render(input.value); return; }
    if (tag === "INPUT" || tag === "TEXTAREA" || e.ctrlKey || e.metaKey || e.altKey) return;
    if (SHORTCUTS[e.key]) post(SHORTCUTS[e.key]);
  });
  render("");
})();
</script>
"""


def _command_palette() -> None:
    """Keyboard-driven command palette — features 5 & 6 (no-op outside browser)."""
    try:
        import streamlit.components.v1 as components

        choice = components.html(_PALETTE_HTML, height=0)
        if choice and choice in PAGES:  # noqa: F821 - PAGES defined below
            st.session_state["nav"] = choice
            st.rerun()
    except Exception:  # noqa: BLE001 - palette must never break the app
        pass


# --------------------------------------------------------------------------- #
# sidebar: stats, quick-nav, global search, theme — features 1, 2, 7
# --------------------------------------------------------------------------- #
def _sidebar_stats(cfg: Config) -> None:
    clips = len(list((ROOT / "outputs").glob("*.wav"))) if (ROOT / "outputs").exists() else 0
    evals = ROOT / "metadata" / "eval_results.jsonl"
    n_eval = sum(1 for _ in evals.open()) if evals.exists() else 0
    st.caption("📦 " + f"**{clips}** clips · **{n_eval}** eval rows")
    st.caption("🎚️ " + cfg.inference.model_name.split("/")[-1])
    lb = ROOT / "metadata" / "leaderboard.json"
    if lb.exists():
        data = json.loads(lb.read_text())
        if data.get("leaderboard"):
            st.caption("🏆 " + data["leaderboard"][0]["checkpoint"].split("/")[-1]
                       + f" · {data['leaderboard'][0]['score']:.2f}")


def _quicknav() -> None:
    """Quick-jump buttons for the most-used pages — feature 2."""
    st.markdown('<div class="mt-quicknav">', unsafe_allow_html=True)
    cols = st.columns(3)
    targets = ["🎛️ Generate", "📊 Compare", "🏆 Leaderboard"]
    for col, t in zip(cols, targets):
        if col.button(t, key=f"qn_{t}", width="stretch"):
            st.session_state["nav"] = t
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _global_search() -> None:
    """Search across inventory / labels / eval results — feature 7."""
    with st.expander("🔎 Global search", expanded=False):
        q = st.text_input("Search metadata", value="", key="gs_query", placeholder="e.g. chorus, piano, 808…")
        if not q.strip():
            st.caption("Type to search inventory, labels, and eval prompts.")
            return
        q = q.strip().lower()
        hits = 0
        inv = ROOT / "metadata" / "audio_inventory.json"
        if inv.exists():
            try:
                df = pd.read_json(inv)
                n = sum(1 for _, r in df.iterrows() if q in str(r.get("path", "")).lower())
                hits += n
                if n:
                    st.markdown(f'<div class="mt-search-hit">📋 inventory — <b>{n}</b> match(es)</div>', unsafe_allow_html=True)
            except Exception:  # noqa: BLE001
                pass
        lab = ROOT / "metadata" / "labels.csv"
        if lab.exists():
            try:
                ldf = pd.read_csv(lab)
                n = int(ldf.astype(str).apply(lambda c: c.str.lower().str.contains(q)).sum().sum())
                hits += n
                if n:
                    st.markdown(f'<div class="mt-search-hit">🏷️ labels — <b>{n}</b> match(es)</div>', unsafe_allow_html=True)
            except Exception:  # noqa: BLE001
                pass
        ev = ROOT / "metadata" / "eval_results.jsonl"
        if ev.exists():
            n = 0
            for ln in ev.open():
                if q in ln.lower():
                    n += 1
            hits += n
            if n:
                st.markdown(f'<div class="mt-search-hit">🧪 eval — <b>{n}</b> prompt(s)</div>', unsafe_allow_html=True)
        if hits == 0:
            st.caption("No matches.")


# --------------------------------------------------------------------------- #
# shared primitives
# --------------------------------------------------------------------------- #
def _skeleton(n: int = 3, height: int = 60) -> None:
    for _ in range(n):
        st.skeleton(height=height)


# feature 41-45 infra ---------------------------------------------------------
def _log_line(text: str) -> None:
    """Append a timestamped line to metadata/musictrain.log (feature 42)."""
    try:
        logf = ROOT / "metadata" / "musictrain.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        with logf.open("a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {text}\n")
    except Exception:  # noqa: BLE001 - logging must never break the app
        pass


def _log_tail(n: int = 300) -> str:
    logf = ROOT / "metadata" / "musictrain.log"
    if not logf.exists():
        return "(no activity logged yet — run a job from any page)"
    lines = logf.read_text().splitlines()
    return "\n".join(lines[-n:])


def _beep() -> None:
    """Feature 45: play a short two-tone chime in the browser on completion."""
    try:
        import streamlit.components.v1 as components

        components.html(
            """<script>
            try {
              const ctx = new (window.AudioContext || window.webkitAudioContext)();
              [660, 880].forEach((f, i) => {
                const o = ctx.createOscillator(), g = ctx.createGain();
                o.frequency.value = f; o.type = 'sine';
                g.gain.setValueAtTime(0.0001, ctx.currentTime + i * 0.15);
                g.gain.exponentialRampToValueAtTime(0.08, ctx.currentTime + i * 0.15 + 0.02);
                g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + i * 0.15 + 0.4);
                o.connect(g); g.connect(ctx.destination);
                o.start(ctx.currentTime + i * 0.15); o.stop(ctx.currentTime + i * 0.15 + 0.45);
              });
            } catch (e) {}
            </script>""",
            height=0,
        )
    except Exception:  # noqa: BLE001 - sound is best-effort
        pass


def _record_job(label: str, fn: Callable, args: tuple, kwargs: dict) -> None:
    """Feature 39: remember the last job for one-click replay, and log it."""
    st.session_state["mt_last_job"] = (label, fn, args, kwargs)
    _log_line(f"▶ {label}")


def _last_job_ui() -> None:
    """Feature 39: replay button for the last job run from the dashboard."""
    last = st.session_state.get("mt_last_job")
    if not last:
        return
    label, fn, args, kwargs = last
    st.markdown("---")
    st.caption("↻ Last job")
    if st.button(f"Replay: {label}", key="last_job_replay", width="stretch"):
        _run_job(label, fn, *args, **kwargs)


def _run_job(label: str, fn: Callable, *args, **kwargs):
    """Run `fn` on a worker thread while animating a progress bar."""
    _record_job(label, fn, args, kwargs)
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
        _log_line(f"✗ {label}: {out['error']}")
        _json_log("job", label=label, status="failed", error=str(out["error"]))
        st.error(str(out["error"]))
        raise out["error"]
    bar.progress(1.0, text=f"{label} — done")
    _log_line(f"✓ {label}")
    _json_log("job", label=label, status="done")
    try:
        st.toast(f"✅ {label} — done")  # feature 41
        _beep()                          # feature 45
    except Exception:  # noqa: BLE001
        pass
    return out.get("result")


def _run_job_cancellable(label: str, fn: Callable, *args, **kwargs):
    """Feature 23/28: like _run_job but with a Cancel button that stops the wait.

    The worker thread is daemon; cancellation stops the UI wait and marks the
    job aborted (the thread finishes or is abandoned in the background). The
    progress bar and cancel button live in one `st.empty()` slot, replaced each
    tick, so no duplicate-widget errors occur on long jobs.
    """
    out: dict = {}

    def _worker() -> None:
        try:
            out["result"] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            out["error"] = exc

    _record_job(label, fn, args, kwargs)
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    slot = st.empty()
    cancelled = False
    pct = 0.0
    while thread.is_alive():
        pct = min(pct + 0.03, 0.92)
        with slot.container():
            st.progress(pct, text=label)
            if st.button("⏹ Cancel", key="cancel_job_btn"):
                cancelled = True
                break
        time.sleep(0.12)
    if not cancelled:
        thread.join(timeout=5)

    if cancelled:
        with slot.container():
            st.progress(1.0, text=f"{label} — cancelled")
            st.info(f"{label} cancelled (worker finishing in background).")
        _log_line(f"■ {label} cancelled")
        _json_log("job", label=label, status="cancelled")
        return None
    if "error" in out:
        with slot.container():
            st.progress(1.0, text=f"{label} — failed")
            st.error(str(out["error"]))
        _log_line(f"✗ {label}: {out['error']}")
        _json_log("job", label=label, status="failed", error=str(out["error"]))
        raise out["error"]
    with slot.container():
        st.progress(1.0, text=f"{label} — done")
    _log_line(f"✓ {label}")
    _json_log("job", label=label, status="done")
    try:
        st.toast(f"✅ {label} — done")
        _beep()
    except Exception:  # noqa: BLE001
        pass
    return out.get("result")


def _run_live(label: str, fn: Callable, *args, progress_kw: str = "progress",
              cancel_kw: Optional[str] = None, **kwargs):
    """Features 27+43: run `fn` with live progress (and optional cancel).

    `fn` must accept a ``progress(done, total)`` callback (and optionally a
    ``cancel() -> bool`` callback, named by ``cancel_kw``). The bar shows real
    item counts instead of a fake animation. `cancel_kw` may be None when the
    target function has no cancellation support.
    """
    state = {"done": 0, "total": 0, "result": None, "error": None, "cancelled": False}

    def _cb(done: int, total: int) -> None:
        state["done"], state["total"] = done, total

    cb = {progress_kw: _cb}
    if cancel_kw:
        cb[cancel_kw] = lambda: state["cancelled"]

    def _worker() -> None:
        try:
            state["result"] = fn(*args, **cb, **kwargs)
        except Exception as exc:  # noqa: BLE001
            state["error"] = exc

    _record_job(label, fn, args, kwargs)
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    slot = st.empty()
    while thread.is_alive():
        done, total = state["done"], state["total"]
        frac = (done / total) if total else 0.0
        with slot.container():
            st.progress(frac, text=f"{label} — {done}/{total}")
            if cancel_kw and st.button("⏹ Cancel", key="cancel_live_btn"):
                state["cancelled"] = True
                break
        time.sleep(0.15)
    thread.join(timeout=5)

    if state["error"]:
        with slot.container():
            st.progress(1.0, text=f"{label} — failed")
            st.error(str(state["error"]))
        _log_line(f"✗ {label}: {state['error']}")
        _json_log("job", label=label, status="failed", error=str(state["error"]))
        raise state["error"]
    done, total = state["done"], state["total"]
    frac = (done / total) if total else 1.0
    with slot.container():
        st.progress(frac, text=f"{label} — {done}/{total} (done)")
    _log_line(f"✓ {label} ({done}/{total})")
    _json_log("job", label=label, status="done", progress=f"{done}/{total}")
    try:
        st.toast(f"✅ {label} — done")
        _beep()
    except Exception:  # noqa: BLE001
        pass
    return state["result"]


def _run_eval_queue(label: str, fn: Callable, *args, **kwargs):
    """Feature 27: eval wrapper over _run_live with cancel + prompt counts."""
    return _run_live(label, fn, *args, progress_kw="progress", cancel_kw="cancel", **kwargs)


def _zip_report() -> None:
    """Feature 24: one-click bundle of the eval + metrics reports as a zip."""
    import io
    import zipfile

    buf = io.BytesIO()
    meta = ROOT / "metadata"
    files = [
        ("eval_results.csv", meta / "eval_results.csv"),
        ("eval_report.html", meta / "eval_report.html"),
        ("leaderboard.json", meta / "leaderboard.json"),
        ("significance.json", meta / "significance.json"),
        ("metrics.json", meta / "metrics.json"),
        ("quality_report.json", meta / "quality_report.json"),
    ]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in files:
            if path.exists():
                zf.write(path, arcname=f"musictrain/{name}")
    st.download_button(
        "📦 Download report bundle (.zip)", data=buf.getvalue(),
        file_name="musictrain_report.zip", mime="application/zip", type="primary",
    )


def _read_json(rel: str):
    p = ROOT / "metadata" / rel
    return json.loads(p.read_text()) if p.exists() else None


def _json_log(event: str, **fields) -> None:
    """Append a structured event to metadata/runlog.jsonl (advanced #46)."""
    try:
        from musictrain.telemetry import json_log

        json_log(ROOT, event, **fields)
    except Exception:  # noqa: BLE001 - logging must never break the UI
        pass


def load_cfg() -> Config:
    p = ROOT / "configs" / "default.yaml"
    cfg = Config.load(p) if p.exists() else Config()
    cfg.project_root = ROOT
    return cfg


# --------------------------------------------------------------------------- #
# Data tables + charts — UI batch 2 (features 11-20)
# --------------------------------------------------------------------------- #
def _csv_bytes(df: pd.DataFrame) -> str:
    return df.to_csv(index=False).encode("utf-8").decode("utf-8")


def _csv_download(df: pd.DataFrame, filename: str, label: str = "⬇ CSV") -> None:
    st.download_button(
        label, data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename, mime="text/csv", width="stretch",
    )


def _table(
    df: pd.DataFrame,
    key: str,
    filename: str,
    progress_cols: Optional[list] = None,
    filter_col: Optional[str] = None,
    height: Optional[int] = None,
) -> None:
    """Feature 11-13: sortable table with optional progress columns, a filter,
    and a CSV download."""
    config = {}
    for c in progress_cols or []:
        if c in df.columns:
            config[c] = st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.2f")

    col_f, col_d = st.columns([4, 1])
    q = ""
    if filter_col and filter_col in df.columns:
        q = col_f.text_input(f"Filter by {filter_col}", value="", key=f"flt_{key}",
                             label_visibility="collapsed", placeholder=f"Filter by {filter_col}…")
    view = df
    if q.strip():
        mask = df[filter_col].astype(str).str.contains(q.strip(), case=False, na=False)
        view = df[mask]
    with col_d:
        _csv_download(view, filename)
    st.dataframe(view, width="stretch", hide_index=True, height=height,
                 column_config=config or None)


def _copy_button(text: str, key: str) -> None:
    """Feature 14: copy-to-clipboard button for prompts/JSON."""
    import streamlit.components.v1 as components

    payload = text.replace("`", "\\`").replace("${`, `}", "")
    js = f"""
    <button onclick="navigator.clipboard.writeText(`{payload}`);this.innerHTML='✓ Copied'"
      style="border-radius:8px;border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.06);
      color:#eef1fb;padding:5px 14px;cursor:pointer;font-size:.8rem">⧉ Copy</button>
    """
    components.html(js, height=36)


def _audio_grid(files: list, cols: int = 4) -> None:
    """Feature 15: inline audio player grid instead of stacked players."""
    if not files:
        return
    for i in range(0, len(files), cols):
        row = files[i:i + cols]
        cs = st.columns(cols)
        for c, f in zip(cs, row):
            with c:
                st.caption(str(Path(f).name)[:34])
                st.audio(str(f))


def _waveform_chart(path: str, key: str) -> None:
    """Feature 16: waveform thumbnail via librosa envelope + altair."""
    import altair as alt
    import librosa
    import numpy as np

    try:
        y, sr = librosa.load(str(path), sr=8000, mono=True)
        hop = max(1, len(y) // 240)
        env = np.abs(y[::hop])
        n = len(env)
        xs = np.linspace(0, len(y) / sr, n)
        pdf = pd.DataFrame({"t": xs, "amp": env})
        chart = (
            alt.Chart(pdf)
            .mark_area(opacity=0.5, color="#5b8cff")
            .encode(x=alt.X("t:Q", axis=None), y=alt.Y("amp:Q", axis=None))
            .properties(height=60)
        )
        st.altair_chart(chart, width="stretch", key=f"wav_{key}")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"waveform unavailable: {exc}")


def _spectrogram_chart(path: str, key: str) -> None:
    """Feature 17: spectrogram view for a clip."""
    import librosa
    import numpy as np

    try:
        y, sr = librosa.load(str(path), sr=16000, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 2.6))
        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                       sr=sr, x_axis="time", y_axis="mel", ax=ax)
        fig.colorbar(img, ax=ax, fraction=0.025)
        ax.set_title(Path(path).name)
        st.pyplot(fig)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"spectrogram unavailable: {exc}")


def _clap_sparkline(df: pd.DataFrame, key: str) -> None:
    """Feature 18: CLAP score trend over eval runs (ordered by run index)."""
    import altair as alt

    d = df.dropna(subset=["clap_score"])
    if d.empty:
        return
    d = d.reset_index().rename(columns={"index": "run"})
    chart = (
        alt.Chart(d)
        .mark_line(point=True, color="#7c5cff")
        .encode(x=alt.X("run:Q", axis=None, title="run"),
                y=alt.Y("clap_score:Q", scale=alt.Scale(zero=False)),
                tooltip=["run", "clap_score", "model"])
        .properties(height=90)
    )
    st.altair_chart(chart, width="stretch", key=f"spark_{key}")


def _bpm_heatmap(rows: list, key: str) -> None:
    """Feature 19: BPM deviation heatmap (prompt x checkpoint)."""
    import altair as alt

    data = []
    for r in rows:
        dev = r.get("deviation")
        if dev is None:
            continue
        short = (r.get("prompt") or "")[:26]
        data.append({"prompt": short, "checkpoint": (r.get("checkpoint") or "").split("/")[-1][:16],
                     "|dev|": abs(float(dev))})
    if len(data) < 2:
        st.caption("Not enough rows for a heatmap.")
        return
    pdf = pd.DataFrame(data)
    chart = (
        alt.Chart(pdf)
        .mark_rect()
        .encode(x=alt.X("checkpoint:N", sort=None),
                y=alt.Y("prompt:N", sort=None),
                color=alt.Color("|dev|:Q", scale=alt.Scale(scheme="viridis"), title="|dev|"))
        .properties(height=max(180, 12 * pdf["prompt"].nunique()))
    )
    st.altair_chart(chart, width="stretch", key=f"hm_{key}")


def _radar_chart(entries: list, key: str) -> None:
    """Feature 20: per-tag CLAP radar for the top checkpoint."""
    import numpy as np
    import matplotlib.pyplot as plt

    if not entries:
        return
    e = entries[0]
    tags = (e.get("clap_per_tag") or {})
    if not tags:
        return
    labels = sorted(tags)
    vals = [tags[t] for t in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, color="#5b8cff", linewidth=2)
    ax.fill(angles, vals, color="#5b8cff", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(e["checkpoint"].split("/")[-1], fontsize=10)
    st.pyplot(fig)


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
        _table(df, "inv", "inventory.csv", filter_col="path", height=360)

    # feature 33 — project wizard: bootstrap layout -> upload -> normalize -> features
    with st.expander("🚀 Project wizard", expanded=False):
        st.caption(
            "Bootstrap (or resume) the corpus pipeline: folder layout → raw audio → "
            "normalized clean → feature extraction. Steps already done are detected "
            "from disk, so it doubles as a resume helper."
        )
        steps = [
            ("Layout", "data/ + metadata/ folders", lambda: (ROOT / "data").exists()),
            ("Upload raw", "audio in data/raw", lambda: any((ROOT / "data" / "raw").glob("*"))),
            ("Normalize", "converted audio in data/clean", lambda: any((ROOT / "data" / "clean").glob("*"))),
            ("Features", "metadata/manifest.jsonl", lambda: (ROOT / "metadata" / "manifest.jsonl").exists()),
        ]
        done = [name for name, _d, check in steps if check()]
        st.progress(len(done) / len(steps), text=f"Pipeline steps done: {len(done)}/{len(steps)}")
        for name, desc, _check in steps:
            mark = "✅" if name in done else "⬜"
            st.markdown(f"{mark} **{name}** — {desc}")

        if st.button("▶ Run remaining steps", type="primary", key="wiz_run"):
            from musictrain.paths import ensure_layout

            ensure_layout(ROOT)
            if "Normalize" not in done:
                from musictrain.audio.normalize import normalize

                converted, skipped, failed = _run_job_cancellable(
                    "Normalizing audio", normalize, ROOT, load_cfg(), force=False,
                )
                if converted is not None:
                    st.success(f"{converted} converted · {skipped} skipped · {failed} failed")
            if "Features" not in done:
                from musictrain.metadata import extract

                records = _run_job_cancellable(
                    "Extracting features", extract, ROOT, load_cfg(), which="clean",
                )
                if records is not None:
                    st.success(f"Processed {len(records)} files -> metadata/manifest.jsonl")
            st.rerun()


# --------------------------------------------------------------------------- #
# 🔧 Normalize
# --------------------------------------------------------------------------- #
def page_normalize() -> None:
    _page_header("🔧", "Normalize audio", "Converts data/raw/* to data/clean/* (mono, 32 kHz, PCM).")
    cfg = load_cfg()
    force = st.checkbox("Force overwrite", value=False, key="norm_force")

    # feature 30 — drag & drop upload straight into data/raw, then normalize
    with st.expander("⬆ Upload audio (drag & drop)", expanded=False):
        st.caption(
            "Drop files here — they land in data/raw and the normalizer picks "
            "them up on the next run (mono, 32 kHz PCM → data/clean)."
        )
        up = st.file_uploader(
            "Audio files", type=["wav", "mp3", "flac", "ogg", "m4a"],
            accept_multiple_files=True, key="norm_upload",
        )
        if up and st.button("Save uploads to data/raw", type="secondary", key="norm_save_uploads"):
            from musictrain.paths import ensure_layout

            ensure_layout(ROOT)
            raw = ROOT / "data" / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in up:
                dest = raw / f.name
                dest.write_bytes(f.getbuffer())
                n += 1
            st.success(f"Saved {n} file(s) -> data/raw — run normalization to convert them.")
            st.rerun()

    if st.button("Run normalization", type="primary", key="norm_run"):
        from musictrain.audio.normalize import normalize
        from musictrain.paths import ensure_layout

        ensure_layout(ROOT)

        def _go():
            return normalize(ROOT, cfg, force=force)

        converted, skipped, failed = _run_job_cancellable("Normalizing audio", _go)
        if converted is None:
            st.info("Normalization cancelled.")
        else:
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

        records = _run_job_cancellable("Extracting features", _go)
        if records is None:
            st.info("Feature extraction cancelled.")
        else:
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

            segs = _run_live("Segmenting audio", segment, ROOT, cfg, progress_kw="progress")
            if segs is not None:
                st.success(f"{len(segs)} segments written")
    with col2:
        st.subheader("Train/val/test split")
        st.write(
            f"Ratios: {cfg.split.train:.0%}/{cfg.split.val:.0%}/{cfg.split.test:.0%} "
            f"· seed {cfg.split.seed}"
        )
        if st.button("Run split", type="primary", key="split_run"):
            from musictrain.split import split

            _run_job_cancellable("Splitting corpus", split, ROOT, cfg)
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

    # feature 37 — template library (apply fills the prompt area)
    t_names = ["— custom —"] + [t["name"] for t in _TEMPLATES]
    tc1, tc2 = st.columns([3, 1])
    tpl = tc1.selectbox("📚 Template", t_names, index=0, key="gen_tpl")
    if tc2.button("Apply template", key="gen_tpl_apply", width="stretch"):
        hit = next((t for t in _TEMPLATES if t["name"] == tpl), None)
        if hit:
            st.session_state["gen_prompt"] = hit["prompt"]
            st.rerun()

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

    with st.popover("⚙️ Advanced sampling", width="stretch"):
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.05, key="gen_temp")
        top_k = st.slider("Top-k", 1, 1000, 250, 10, key="gen_topk")
        top_p = st.slider("Top-p", 0.5, 1.0, 1.0, 0.01, key="gen_topp")
        preset = st.selectbox("Preset", ["", "standard", "creative", "precise"], key="gen_preset")
        negative = st.text_input("Negative prompt (CLAP-checked)", value="", key="gen_negative")

    # feature 16 — melody-from-audio conditioning (best with musicgen-melody)
    melody_file = st.file_uploader(
        "🎵 Condition on a melody (optional, use musicgen-melody)",
        type=["wav", "mp3", "flac", "ogg"], key="gen_melody",
    )
    melody_path = None
    if melody_file is not None:
        tmp = ROOT / "outputs" / f"_melody_in_{melody_file.name}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(melody_file.getbuffer())
        melody_path = tmp

    if st.button("Generate", type="primary", key="gen_run"):
        from musictrain.inference import generate

        cfg.inference.model_name = model
        cfg.inference.guidance_scale = guidance
        cfg.inference.max_new_tokens = int(tokens)
        cfg.inference.temperature = temperature
        cfg.inference.top_k = int(top_k)
        cfg.inference.top_p = top_p
        cfg.inference.preset = preset
        cfg.inference.negative_prompt = negative

        t0 = time.monotonic()

        def _go():
            return generate(
                cfg, prompt, out_dir=ROOT / "outputs", seed=int(seed) or None,
                melody_from=melody_path,
            )

        result = _run_job_cancellable("Generating audio (MPS)", _go)
        if result is None:
            st.info("Generation cancelled.")
        else:
            elapsed = max(time.monotonic() - t0, 0.001)
            tps = result.get("max_new_tokens", 0) / elapsed
            st.success(f"Saved {result['path']} ({result['duration']}s, {result['device']})")
            st.caption(f"⚡ ~{tps:.0f} tokens/s (model+audio pipeline wall time) — feature 44")
            st.subheader("Result")
            st.audio(str(result["path"]))
            st.json(result)

    # feature 26 — batch prompt editor (one prompt per line, sequential generate)
    with st.expander("🧪 Batch prompts", expanded=False):
        st.caption(
            "Generate several prompts in one run — one prompt per line, each "
            "saved to outputs/ with the advanced settings above."
        )
        batch = st.text_area(
            "Batch prompts (one per line)", value="", height=140, key="gen_batch",
            placeholder="dark piano intro, 70 BPM, A minor\nverse, 84 BPM, trap hats, airy pads",
        )
        if st.button("Generate batch", type="primary", key="gen_batch_run"):
            prompts = [ln.strip() for ln in batch.splitlines() if ln.strip()]
            if not prompts:
                st.warning("Enter at least one prompt.")
            else:
                from musictrain.inference import generate

                cfg.inference.model_name = model
                cfg.inference.guidance_scale = guidance
                cfg.inference.max_new_tokens = int(tokens)
                cfg.inference.temperature = temperature
                cfg.inference.top_k = int(top_k)
                cfg.inference.top_p = top_p
                cfg.inference.preset = preset
                cfg.inference.negative_prompt = negative

                saved, failed = [], 0
                slot = st.empty()
                for i, p in enumerate(prompts, 1):
                    with slot.container():
                        st.progress(i / len(prompts), text=f"Generating {i}/{len(prompts)} — {p[:48]}…")
                    try:
                        res = generate(cfg, p, out_dir=ROOT / "outputs", seed=int(seed) or None)
                        saved.append(res["path"])
                    except Exception as exc:  # noqa: BLE001 - one bad prompt must not kill the batch
                        failed += 1
                        st.error(f"Prompt {i} failed: {exc}")
                with slot.container():
                    st.success(f"Batch done: {len(saved)} saved, {failed} failed")
                if saved:
                    _audio_grid(saved, cols=4)

    # recent outputs grid (feature 15)
    recent = sorted((ROOT / "outputs").glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    if recent:
        st.markdown("---")
        st.subheader("Recent outputs")
        _audio_grid(recent, cols=4)


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

    with st.popover("⚙️ Tolerance", width="stretch"):
        tol = st.slider("BPM tolerance", 0.01, 0.20, 0.05, 0.01, key="chk_tol",
                        format="%.2f", help="Fractional deviation allowed before reject")
        max_stretch = st.slider("Max time-stretch", 0.02, 0.30, 0.10, 0.01, key="chk_stretch",
                                format="%.2f")

    if st.button("Check", type="primary", key="chk_run"):
        from musictrain.evaluate import check

        cfg = load_cfg()
        cfg.check.bpm_tolerance = tol
        cfg.check.max_time_stretch = max_stretch

        report = _run_job_cancellable(
            "Checking BPM",
            check, cfg, ROOT / pick,
            target_bpm=float(target) if target > 0 else None,
            fix=fix,
        )
        if report is None:
            st.info("Check cancelled.")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.json(report)
                _copy_button(json.dumps(report, indent=2), "chk_copy")
            with c2:
                t_w, t_s = st.tabs(["Waveform", "Spectrogram"])
                with t_w:
                    _waveform_chart(str(ROOT / pick), "chk")
                with t_s:
                    _spectrogram_chart(str(ROOT / pick), "chk")
            st.subheader("Audio")
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

    # advanced #6-#9 — prompt difficulty + calibration, read from CLI output
    diff = _read_json("prompt_difficulty.json")
    if diff:
        with st.expander("🎯 Prompt difficulty & calibration (run `musictrain difficulty`)", expanded=False):
            cal = diff.get("calibration", {})
            if cal.get("suggested_max_abs_deviation") is not None:
                a1, a2, a3 = st.columns(3)
                a1.metric("Suggested max |dev|", cal["suggested_max_abs_deviation"],
                          help=f"rejects {cal.get('would_reject_dev')} prompts")
                a2.metric("Suggested min CLAP", cal["suggested_min_clap_score"],
                          help=f"rejects {cal.get('would_reject_clap')} prompts")
                a3.metric("Prompts", cal["n_prompts"])
            hard = diff.get("difficulty", [])[:8]
            if hard:
                st.caption("Hardest prompts")
                st.dataframe(
                    pd.DataFrame(hard)[["difficulty", "section", "bpm_target", "status", "prompt"]],
                    width="stretch", hide_index=True,
                )
            inter = diff.get("section_bpm_interaction")
            if inter:
                st.caption("Section × BPM interaction (bpm_dev_corr: >0 means faster BPM drifts more)")
                st.dataframe(pd.DataFrame(inter), width="stretch", hide_index=True)

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
    _table(view, "cmp", "mlflow_runs.csv", progress_cols=["clap_score"], height=320)

    st.subheader("CLAP trend over runs")
    _clap_sparkline(view, "cmp")

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
        drift = _read_json("drift.json")
        curation = _read_json("curation_scores.json")
        leakage = _read_json("leakage.json")

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

    drift_count = len(drift.get("drifted_features", [])) if drift else None
    top_cur = (curation.get("tracks") or [{}])[0].get("score") if curation else None
    leak_count = leakage.get("cross_split_duplicates", 0) if leakage else None
    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "Drifted features",
        f"{drift_count}" if drift_count is not None else "—",
        "ref→cur" if drift else None,
    )
    d2.metric("Top curation score", f"{top_cur}" if top_cur is not None else "—")
    d3.metric(
        "Cross-split leaks",
        f"{leak_count}" if leak_count is not None else "—",
        "train/val/test" if leak_count else None,
    )
    d4.metric(
        "Files checked (leakage)",
        f"{leakage['files_checked']}" if leakage else "—",
    )

    cfg = load_cfg()
    b1, b2, b3, b4, b5, b6, b7 = st.columns(7)
    if b1.button("Run quality", width="stretch", key="hyg_quality"):
        from musictrain.audio.quality import quality as run_q

        _run_job("Scoring quality", run_q, ROOT, cfg)
        st.rerun()
    if b2.button("Run dedup", width="stretch", key="hyg_dedup"):
        from musictrain.dedup import find_duplicates

        _run_job("Finding duplicates", find_duplicates, ROOT, cfg)
        st.rerun()
    if b3.button("Run corpus", width="stretch", key="hyg_corpus"):
        from musictrain.corpus import corpus as run_c

        _run_job("Computing corpus stats", run_c, ROOT, cfg)
        st.rerun()
    if b4.button("Run OOD", width="stretch", key="hyg_ood"):
        from musictrain.ood import curate_ood

        _run_job("Flagging OOD tracks", curate_ood, ROOT, cfg)
        st.rerun()
    if b5.button("Run drift", width="stretch", key="hyg_drift"):
        from musictrain.drift import drift_report

        _run_job("Computing drift", drift_report, ROOT, cfg,
                 reference="clean", current="train")
        st.rerun()
    if b6.button("Run curation", width="stretch", key="hyg_curation"):
        from musictrain.curation import curation_score

        _run_job("Scoring curation", curation_score, ROOT, cfg)
        st.rerun()
    if b7.button("Run leakage", width="stretch", key="hyg_leakage"):
        from musictrain.labelprop import leakage_check

        _run_job("Checking leakage", leakage_check, ROOT, cfg)
        st.rerun()

    st.markdown("---")

    t_qual, t_dedup, t_corp, t_ood, t_drift, t_cur, t_leak = st.tabs(
        ["🔊 Quality", "👯 Duplicates", "📈 Corpus", "🚫 OOD",
         "📉 Drift", "⭐ Curation", "🔓 Leakage"]
    )

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

    with t_drift:
        st.subheader("Feature drift")
        if not drift:
            st.info("No drift report yet — run the sweep above or `musictrain drift`.")
        else:
            st.caption(
                f"{drift['reference']} vs {drift['current']} · "
                f"{drift['reference_n']} → {drift['current_n']} tracks · "
                f"KS p < {drift.get('threshold', 0.05)} or PSI > 0.25 flags drift"
            )
            if drift.get("drifted_features"):
                st.warning(f"Drifted: {', '.join(drift['drifted_features'])}")
            else:
                st.success("No drifted features detected.")

            cont = drift.get("continuous") or {}
            if cont:
                cdf = pd.DataFrame(cont).T.reset_index().rename(columns={"index": "feature"})
                cdf = cdf.rename(
                    columns={
                        "reference_mean": "ref mean", "current_mean": "cur mean",
                        "delta": "Δ", "ks_pvalue": "KS p", "psi": "PSI",
                        "drifted": "drifted",
                    }
                )
                st.caption("Continuous features")
                st.dataframe(cdf, width="stretch", hide_index=True)

            disc = drift.get("discrete") or {}
            if disc:
                ddf = pd.DataFrame(disc).T.reset_index().rename(columns={"index": "feature"})
                ddf = ddf.rename(columns={"psi": "PSI", "drifted": "drifted"})
                st.caption("Discrete features")
                st.dataframe(ddf, width="stretch", hide_index=True)

    with t_cur:
        st.subheader("Curation scores")
        if not curation:
            st.info("No curation scores yet — run the sweep above or `musictrain curation`.")
        else:
            tracks = curation.get("tracks") or []
            st.caption(f"Scored {curation.get('scored', len(tracks))} track(s) · higher = keep first")
            if tracks:
                tdf = pd.DataFrame(tracks)
                cols = [c for c in ("path", "score", "quality", "novelty", "coverage", "dup_penalty", "genre") if c in tdf]
                st.bar_chart(tdf.set_index("path")["score"].head(25))
                st.dataframe(tdf[cols].head(100), width="stretch", hide_index=True)

    with t_leak:
        st.subheader("Cross-split leakage")
        if not leakage:
            st.info("No leakage check yet — run the sweep above or `musictrain labelprop --check-leakage`.")
        else:
            leaks = leakage.get("leaks") or []
            st.caption(
                f"Checked {leakage.get('files_checked', 0)} file(s) across "
                f"{', '.join(leakage.get('splits', []))}"
            )
            if not leaks:
                st.success("No cross-split near-duplicates — training/eval are clean.")
            else:
                st.warning(f"{len(leaks)} cross-split near-duplicate pair(s) found")
                ldf = pd.DataFrame(leaks)
                cols = [c for c in ("split_a", "a", "split_b", "b", "similarity") if c in ldf]
                st.dataframe(ldf[cols], width="stretch", hide_index=True)


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

    # features 34-35 — label editor with vocab dropdowns + batch assign
    st.subheader("✏️ Label editor")
    if df is None:
        if st.button("Scaffold labels.csv template", key="lab_scaffold"):
            from musictrain.labels import scaffold

            scaffold(ROOT)
            st.rerun()
        st.info("No labels.csv yet — scaffold one above, then edit in place.")
    else:
        custom_vocab: dict = {}
        cv_path = ROOT / "metadata" / "custom_vocab.json"
        if cv_path.exists():
            custom_vocab = json.loads(cv_path.read_text())

        options: dict = {}
        for dim in ("genre", "mood", "instruments", "section", "section_type"):
            terms = set(VOCAB.get(dim, set())) | set(custom_vocab.get(dim, []))
            options[dim] = sorted(terms)
        editor_cfg = {
            "genre": st.column_config.SelectboxColumn("genre", options=options["genre"]),
            "section": st.column_config.SelectboxColumn("section", options=options["section"]),
            "section_type": st.column_config.SelectboxColumn("section_type", options=options["section_type"]),
            "mood": st.column_config.TextColumn("mood", help="pipe-separated, e.g. dark|emotional"),
            "instruments": st.column_config.TextColumn("instruments", help="pipe-separated"),
        }
        edited = st.data_editor(
            df, key="lab_editor", num_rows="dynamic", hide_index=True,
            width="stretch", column_config=editor_cfg,
            disabled=["source_id"],
        )

        c_new, c_save, c_assign = st.columns([2, 1, 2])
        with c_new:
            new_term = st.text_input(
                "Add vocab term (dim:term)", value="", key="lab_newterm",
                placeholder="mood:euphoric",
            )
            if st.button("➕ Add term", key="lab_addterm") and new_term:
                dim, _, term = new_term.partition(":")
                dim, term = dim.strip(), term.strip()
                if dim in VOCAB and term:
                    custom_vocab.setdefault(dim, [])
                    if term not in custom_vocab[dim]:
                        custom_vocab[dim].append(term)
                    cv_path.write_text(json.dumps(custom_vocab, indent=2))
                    st.success(f"Added {dim}:{term} (custom vocab -> metadata/custom_vocab.json). "
                               "The CLI vocab check won't know it yet — extend labels.py to persist it.")
                    st.rerun()
                else:
                    st.error("Format: dim:term with a known dim (genre/mood/instruments/section).")
        with c_save:
            if st.button("💾 Save labels.csv", type="primary", key="lab_save"):
                edited.to_csv(labels_csv, index=False)
                st.success(f"Saved {len(edited)} rows -> metadata/labels.csv")
                st.rerun()
        with c_assign:
            sel = st.multiselect(
                "Batch-assign to rows", edited.index.tolist(), key="lab_sel",
                format_func=lambda i: f"#{i} {edited.loc[i, 'source_id']}",
            )
            with st.popover("Apply values to selected", width="stretch"):
                b_genre = st.selectbox("genre", [""] + options["genre"], key="lab_bg")
                b_mood = st.text_input("mood (pipe-separated)", "", key="lab_bm")
                b_inst = st.text_input("instruments (pipe-separated)", "", key="lab_bi")
                b_sec = st.selectbox("section", [""] + options["section"], key="lab_bs")
                if st.button("Apply", type="primary", key="lab_bapply") and sel:
                    for i in sel:
                        if b_genre:
                            edited.loc[i, "genre"] = b_genre
                        if b_mood:
                            edited.loc[i, "mood"] = b_mood
                        if b_inst:
                            edited.loc[i, "instruments"] = b_inst
                        if b_sec:
                            edited.loc[i, "section"] = b_sec
                    edited.to_csv(labels_csv, index=False)
                    st.success(f"Applied to {len(sel)} row(s) and saved labels.csv")
                    st.rerun()

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
    _table(pd.DataFrame(rows), "lb", "leaderboard.csv",
           progress_cols=["score", "ok %", "mean CLAP"], height=280)

    c_rad, c_hm = st.columns([1, 2])
    with c_rad:
        st.subheader("Per-tag CLAP radar")
        _radar_chart(entries, "lb")
    with c_hm:
        st.subheader("BPM deviation heatmap")
        from musictrain.report import load_results

        _bpm_heatmap(load_results(ROOT), "lb")


def page_leaderboard() -> None:
    _page_header("🏆", "Leaderboard", "Ranks checkpoints by adherence, BPM fidelity, verdict share, human rating.")
    cfg = load_cfg()

    c_btn, c_zip = st.columns([2, 1])
    with c_btn:
        if st.button("Rebuild leaderboard", type="primary", key="lb_rebuild"):
            from musictrain.leaderboard import build

            _run_job_cancellable("Ranking checkpoints", build, cfg)
            st.rerun()
    with c_zip:
        _zip_report()  # feature 29 — one-click report bundle

    with st.spinner("Loading leaderboard…"):
        _leaderboard_view(cfg)


def _undo_ratings_ui(ratings_path: Path) -> None:
    """Feature 38: undo the last rating save (session undo stack of line counts)."""
    undo = st.session_state.setdefault("mt_rating_undo", [])
    if undo and st.button("↩ Undo last rating save", key="lst_undo"):
        target = undo.pop()
        lines = ratings_path.read_text().splitlines()[:target]
        ratings_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        st.success("Undid last rating save.")
        st.rerun()


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

    checkpoints = sorted({r.get("checkpoint", "?") for r in rows})
    picked = st.multiselect("Checkpoints", checkpoints, default=checkpoints[:1], key="lst_ckpts")
    if picked:
        rows = [r for r in rows if r.get("checkpoint") in picked]
    if not rows:
        st.info("No clips for the selected checkpoint(s).")
        return

    c_ab, c_kp = st.columns(2)
    ab = c_ab.toggle("👥 A/B compare (needs 2+ checkpoints)", value=False, key="lst_ab")
    keypad = c_kp.toggle("⚡ Rapid keypad ratings", value=False, key="lst_keypad",
                         help="One-click 1–5 buttons per clip instead of sliders.")

    if ab and len(picked) >= 2:
        _ab_compare(rows, picked, existing, ratings_path)
        return

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
                _waveform_chart(str(ap), f"lst_{i}")
                st.audio(str(ap))
            else:
                st.warning("Audio file missing")
            c1, c2 = st.columns([1, 3])
            if keypad:
                # feature 32 — rapid-fire 1–5 keypad, one click per rating
                chosen = st.session_state.get(f"kp_{i}", int(prev.get("rating") or 3))
                c1.caption("Rating")
                for v in range(1, 6):
                    if c1.button(
                        str(v), key=f"kp_btn_{i}_{v}",
                        type="primary" if v == chosen else "secondary",
                    ):
                        st.session_state[f"kp_{i}"] = v
                rating = chosen
            else:
                rating = c1.slider(
                    f"Rating {i + 1}", 1, 5, int(prev.get("rating") or 3),
                    key=f"rating_{i}", label_visibility="collapsed",
                )
            note = c2.text_input(
                "Note (optional)", value=prev.get("note", ""),
                key=f"note_{i}", label_visibility="collapsed",
            )
            ratings[key] = {"rating": rating, "note": note}

    c_save, c_undo = st.columns([1, 1])
    if c_save.button("Save ratings", type="primary", key="lst_save"):
        saved = 0
        before = sum(1 for _ in ratings_path.open())
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
        st.session_state.setdefault("mt_rating_undo", []).append(before + saved)  # feature 38
        st.success(f"Saved {saved} rating(s) -> metadata/human_ratings.jsonl")
        st.rerun()
    _undo_ratings_ui(ratings_path)

    st.caption(f"Already rated: {len(existing)} prompt/checkpoint pairs")


def _ab_compare(rows: list, picked: list, existing: dict, ratings_path: Path) -> None:
    """Feature 31 — side-by-side A/B listening: same prompt across checkpoints."""
    from collections import defaultdict

    by_prompt: dict = defaultdict(dict)
    for r in rows:
        by_prompt[(r["prompt"], r.get("section"), r.get("bpm_target"))][r.get("checkpoint")] = r
    pairs = [(k, v) for k, v in by_prompt.items() if len(v) >= 2]
    if not pairs:
        st.info("No prompts generated by 2+ of the selected checkpoints — run the "
                "same prompt set on another checkpoint first.")
        return

    pairs = pairs[:20]
    st.caption(f"{len(pairs)} shared prompt(s) · one column per checkpoint — toggle audio to A/B them.")
    ratings = {}
    for i, ((prompt, section, bpm), ck_rows) in enumerate(pairs):
        with st.container(border=True):
            st.caption(f"**{section}** · {bpm} BPM")
            st.write(prompt)
            cols = st.columns(len(ck_rows))
            for (ck, r), col in zip(ck_rows.items(), cols):
                with col:
                    st.markdown(f"**{ck.split('/')[-1]}**")
                    ap = r.get("audio_path")
                    if ap and Path(ap).exists():
                        st.audio(str(ap))
                    else:
                        st.warning("missing audio")
                    st.caption(f"CLAP {r.get('clap_score')} · dev {r.get('deviation')}")
                    key = (prompt, ck)
                    prev = existing.get(key, {})
                    chosen = st.session_state.get(f"ab_{i}_{ck}", int(prev.get("rating") or 3))
                    row_btns = st.columns(5)
                    for v in range(1, 6):
                        if row_btns[v - 1].button(
                            str(v), key=f"ab_btn_{i}_{ck}_{v}",
                            type="primary" if v == chosen else "secondary",
                        ):
                            st.session_state[f"ab_{i}_{ck}"] = v
                    ratings[key] = {"rating": chosen, "note": prev.get("note", "")}

    c_save, c_undo = st.columns([1, 1])
    if c_save.button("Save A/B ratings", type="primary", key="ab_save"):
        saved = 0
        before = sum(1 for _ in ratings_path.open())
        with ratings_path.open("a") as fh:
            for (prompt, checkpoint), rr in ratings.items():
                if rr["rating"] != existing.get((prompt, checkpoint), {}).get("rating", 3):
                    fh.write(json.dumps({"prompt": prompt, "checkpoint": checkpoint,
                                         "rating": rr["rating"], "note": rr["note"]}) + "\n")
                    saved += 1
        st.session_state.setdefault("mt_rating_undo", []).append(before + saved)
        st.success(f"Saved {saved} A/B rating(s) -> metadata/human_ratings.jsonl")
        st.rerun()
    _undo_ratings_ui(ratings_path)


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

    # feature 36 — prompt gallery (save favorites, one-click reuse)
    gallery_path = ROOT / "metadata" / "prompt_gallery.json"
    gallery: list = json.loads(gallery_path.read_text()) if gallery_path.exists() else []
    gc1, gc2 = st.columns([3, 1])
    gtags = gc1.text_input("Gallery tags (comma-separated)", value="", key="pb_gtags",
                           placeholder="chorus, dark, bpm140")
    if gc2.button("💾 Save to gallery", key="pb_gsave") and prompt.strip():
        gallery.append({
            "prompt": prompt,
            "tags": [t.strip() for t in gtags.split(",") if t.strip()],
            "section": section,
            "created": time.strftime("%Y-%m-%d %H:%M"),
        })
        gallery_path.write_text(json.dumps(gallery, indent=2))
        st.success("Saved to metadata/prompt_gallery.json")
        st.rerun()
    if gallery:
        with st.expander(f"🗂️ Prompt gallery ({len(gallery)} saved)", expanded=False):
            gal_filter = st.text_input("Filter by tag", value="", key="pb_gfilter")
            q = gal_filter.strip().lower()
            for i, g in enumerate(gallery):
                if q and not any(q in t.lower() for t in g.get("tags", [])):
                    continue
                st.caption(f"{g.get('created', '')} · tags: {', '.join(g.get('tags', [])) or '—'}")
                st.code(g["prompt"], language=None)
                if st.button("↩ Use", key=f"pb_guse_{i}"):
                    st.session_state["pb_prompt"] = g["prompt"]
                    st.rerun()

    if st.button("Generate with MusicGen", type="primary", key="pb_gen"):
        from musictrain.inference import generate

        result = _run_job_cancellable("Generating audio (MPS)", generate, cfg, prompt, out_dir=ROOT / "outputs")
        if result is not None:
            st.success(f"Saved {result['path']} ({result['duration']}s)")
            st.audio(str(result["path"]))


@st.fragment(run_every=15)
def _sched_eval(cfg: Config, secs: list, seeds: int, limit: int) -> None:
    """Feature 40: auto-run the selected eval on a timer while armed & overdue.

    The last-run timestamp is set *before* starting, so interrupted fragment
    re-runs can never stack a second eval on top of a live one.
    """
    if not st.session_state.get("ev_sched"):
        return
    nmin = int(st.session_state.get("ev_nmin", 10) or 10)
    last = st.session_state.get("ev_last_run", 0.0)
    if time.time() - last < nmin * 60:
        return

    from musictrain.evalset import run_eval as _run_eval

    evf = ROOT / "metadata" / "eval_results.jsonl"
    if evf.exists():
        (evf.parent / "eval_results.jsonl.bak").write_bytes(evf.read_bytes())
    st.session_state["ev_last_run"] = time.time()
    st.info(f"🕐 Scheduled eval starting ({time.strftime('%H:%M:%S')}) — watch the Logs page.")
    section = ",".join(secs) if secs else None
    results = _run_live(
        "Scheduled eval", _run_eval, cfg,
        limit=limit, section=section, seeds=seeds,
        progress_kw="progress", cancel_kw="cancel",
    )
    ok = sum(1 for r in (results or []) if r.get("status") == "ok")
    st.success(f"Scheduled eval done: {len(results or [])} prompts, {ok} ok.")


# --------------------------------------------------------------------------- #
# 🎯 Eval queue
# --------------------------------------------------------------------------- #
def page_eval() -> None:
    _page_header(
        "🎯", "Eval queue",
        "Run the fixed prompt set with live per-prompt progress; cancel or "
        "resume any time. Results stream to metadata/eval_results.jsonl.",
    )
    from musictrain.evalset import load as load_evalset, run_eval as _run_eval

    cfg = load_cfg()
    prompts = load_evalset(cfg.project_root)
    if not prompts:
        st.info("No eval prompt set yet — run `musictrain evalset` on the CLI first.")
        return

    sections = sorted({p.get("section", "?") for p in prompts})
    c1, c2, c3 = st.columns(3)
    secs = c1.multiselect("Sections", sections, default=sections[:2], key="ev_secs")
    seeds = c2.number_input("Seeds (majority verdict)", 1, 5, 1, key="ev_seeds")
    limit = c3.number_input("Limit (0 = all)", 0, len(prompts), 0, key="ev_limit")

    # feature 40 — scheduled auto-run (while this page is open)
    sc1, sc2 = st.columns([1, 2])
    sched = sc1.toggle("🕐 Auto-run eval", value=False, key="ev_sched",
                       help="Re-runs the selected eval on a timer while this page is open.")
    nmin = sc2.number_input("Interval (minutes)", 1, 120, 10, key="ev_nmin")
    if sched:
        st.caption(f"⏱ Armed: auto-run every {int(nmin)} min (fires on page refresh while this page is open).")

    st.caption(
        f"{len(prompts)} prompts in the set · current result file has "
        f"{sum(1 for _ in open(ROOT / 'metadata' / 'eval_results.jsonl')) if (ROOT / 'metadata' / 'eval_results.jsonl').exists() else 0} rows"
    )

    if st.button("▶ Start eval", type="primary", key="ev_run"):
        # protect the current baseline before run_eval overwrites the result file
        evf = ROOT / "metadata" / "eval_results.jsonl"
        if evf.exists():
            (evf.parent / "eval_results.jsonl.bak").write_bytes(evf.read_bytes())

        section = ",".join(secs) if secs else None
        results = _run_eval_queue(
            "Running eval",
            _run_eval,
            cfg,
            limit=int(limit),
            section=section,
            seeds=int(seeds),
        )
        if results is None:
            st.info("Eval cancelled — completed prompts were saved to the result file.")
        else:
            ok = sum(1 for r in results if r.get("status") == "ok")
            st.success(
                f"Eval finished: {len(results)} prompts, {ok} in-tolerance by majority "
                "-> metadata/eval_results.jsonl (previous file backed up to .bak)"
            )
            cols = [c for c in ("prompt", "section", "bpm_target", "detected_bpm", "deviation", "clap_score", "status") if c in results[0]]
            st.dataframe(pd.DataFrame(results)[cols], width="stretch")

    _sched_eval(cfg, secs, int(seeds), int(limit))


# --------------------------------------------------------------------------- #
# 🪵 Logs
# --------------------------------------------------------------------------- #
@st.fragment(run_every=10)
def _logs_view() -> None:
    """Auto-refreshing tail of metadata/musictrain.log (feature 42)."""
    text = _log_tail(400)
    st.caption("Tails metadata/musictrain.log · auto-refreshes every 10s")
    st.code(text, language=None)
    st.download_button(
        "⬇ Download log", data=_log_tail(10000).encode("utf-8"),
        file_name="musictrain.log", mime="text/plain", width="stretch",
    )


@st.fragment(run_every=10)
def _runlog_view() -> None:
    """Auto-refreshing structured event log (metadata/runlog.jsonl, #46)."""
    try:
        from musictrain.telemetry import read_runlog

        events = read_runlog(ROOT, limit=200)
    except Exception:  # noqa: BLE001
        events = []

    if not events:
        st.info("No structured events yet — run any dashboard job and it lands here.")
        return

    kinds = sorted({e.get("event", "?") for e in events})
    kind = st.selectbox(
        "Event type", ["all", *kinds], index=0, key="runlog_kind",
    )
    shown = events if kind == "all" else [e for e in events if e.get("event") == kind]

    rows = []
    for e in reversed(shown[-200:]):
        payload = {k: v for k, v in e.items() if k not in ("at", "event")}
        rows.append(
            {
                "at": (e.get("at") or "")[11:19],
                "event": e.get("event"),
                "details": json.dumps(payload, default=str)[:200],
            }
        )
    st.caption(f"{len(shown)} event(s) from metadata/runlog.jsonl · auto-refreshes every 10s")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if rows:
        st.download_button(
            "⬇ Download runlog",
            data="\n".join(json.dumps(e) for e in shown).encode("utf-8"),
            file_name="runlog.jsonl", mime="application/json", width="stretch",
        )


def page_logs() -> None:
    _page_header("🪵", "Activity log", "Live tail of every dashboard job — started, done, cancelled, failed.")
    t_cli, t_runlog = st.tabs(["🪵 CLI log", "🧾 Runlog events"])
    with t_cli:
        _logs_view()
    with t_runlog:
        _runlog_view()



def _analyze_clip(path: str, cfg: Config) -> dict:
    """Deep-analyze one clip on demand (light path: no CLAP/vocal)."""
    from musictrain.audio.analysis import beat_grid, detect_structure, extract_chords, onset_stats
    from musictrain.audio.features import load_audio

    cache = st.session_state.setdefault("viz_analysis", {})
    if path in cache:
        return cache[path]
    rec: dict = {}
    try:
        with st.spinner("Analyzing clip (chords, beat grid, structure)..."):
            y, sr = load_audio(Path(path), sr=cfg.analysis.sr)
            rec = {
                "chords": extract_chords(y, sr, hop_length=cfg.analysis.hop_length,
                                         frame_seconds=cfg.analysis.chord_frame),
                "beat_grid": beat_grid(y, sr, hop_length=cfg.analysis.hop_length,
                                       beats_per_bar=cfg.analysis.beats_per_bar),
                "onsets": onset_stats(y, sr, hop_length=cfg.analysis.hop_length),
                "structure": detect_structure(
                    y, sr, hop_length=cfg.analysis.hop_length,
                    min_segments=cfg.analysis.structure_min_segments,
                    max_segments=cfg.analysis.structure_max_segments,
                    segment_seconds=cfg.analysis.structure_segment_seconds,
                ),
            }
    except Exception as exc:  # noqa: BLE001
        st.caption(f"analysis failed: {exc}")
    cache[path] = rec
    return rec


def page_visualize() -> None:
    """Batch 1 - previews & audio: waveforms, spectrograms, chords, structure, stems."""
    from musictrain import viz

    _page_header("🎬", "Visualize",
                 "Waveforms, spectrograms, chords, beat grids, structure and stem mixing.")
    cfg = load_cfg()
    files = viz.scan_audio(ROOT, ["data/clean", "data/raw", "data/segments", "outputs"])
    if not files:
        st.info("No audio found under data/ or outputs/. Generate or normalize some clips first.")
        return

    names = {str(p): f"{p.relative_to(ROOT)}" for p in files}
    sel = st.selectbox("Clip", list(names), format_func=lambda k: names[k], key="viz_clip")

    t_wave, t_spect, t_music, t_struct, t_stems = st.tabs(
        ["🌊 Waveform", "🔬 Spectrogram", "🎼 Beat & chords", "🧩 Structure", "🎚️ Stems"])

    with t_wave:
        viz.waveform(sel, "page")
        rec = _analyze_clip(sel, cfg)
        viz.onset_overlay(sel, rec.get("beat_grid", {}), rec.get("onsets", {}), "page")
    with t_spect:
        kind = st.radio("Kind", ["mel", "chroma"], horizontal=True, key="viz_kind")
        viz.spectrogram(sel, "page", kind=kind)
    with t_music:
        rec = _analyze_clip(sel, cfg)
        viz.chord_beat_strip(rec.get("chords", []), rec.get("beat_grid", {}), "page")
        viz.chromagram(sel, "page")
    with t_struct:
        rec = _analyze_clip(sel, cfg)
        viz.structure_timeline(sel, rec.get("structure", {}).get("segments", []), "page")
    with t_stems:
        stem_files = viz.scan_audio(ROOT, ["data/stems"])
        viz.stem_mixer(stem_files, "page")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Before / after")
        clean = viz.scan_audio(ROOT, ["data/clean"])
        raw = viz.scan_audio(ROOT, ["data/raw"])
        if clean and raw:
            viz.before_after(str(raw[0]), str(clean[0]), "page")
        else:
            st.caption("run normalize to compare raw vs clean")
    with c2:
        st.subheader("Live generation")
        viz.live_generation_view(str(ROOT / "outputs"), "page")


PAGES = {
    "📋 Inventory": page_inventory,
    "🔧 Normalize": page_normalize,
    "🏷️ Metadata": page_features,
    "✂️ Segment & Split": page_split,
    "🎛️ Generate": page_generate,
    "🪄 Prompt builder": page_promptbuilder,
    "📏 Check BPM": page_check,
    "🎬 Visualize": page_visualize,
    "🏷️ Labels": page_labels,
    "📊 Compare": page_compare,
    "🧹 Hygiene": page_hygiene,
    "🏆 Leaderboard": page_leaderboard,
    "🎯 Eval": page_eval,
    "🎧 Listening": page_listening,
    "🪵 Logs": page_logs,
}


def main() -> None:
    st.markdown(_theme_css(), unsafe_allow_html=True)
    _command_palette()

    history = st.session_state.setdefault("mt_history", [])
    with st.sidebar:
        st.markdown("### 🎵 MusicTrain")
        st.caption(f"Project: `{ROOT.name}`")
        _toggle_theme()
        _quicknav()
        _global_search()
        st.markdown("---")
        _sidebar_stats(load_cfg())
        _last_job_ui()  # feature 39 — replay the last job
        st.markdown("---")
        choice = st.radio("Go to", list(PAGES.keys()), key="nav")

    # breadcrumb history — feature 4
    if not history or history[-1] != choice:
        history.append(choice)
        st.session_state["mt_history"] = history[-12:]
    _crumbs(history)
    PAGES[choice]()


main()
