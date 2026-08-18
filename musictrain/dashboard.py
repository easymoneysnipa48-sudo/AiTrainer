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
from typing import Callable, Optional

import pandas as pd
import streamlit as st

from musictrain.config import Config
from musictrain.logging import get_logger, setup as setup_logging

st.set_page_config(page_title="MusicTrain", page_icon="🎵", layout="wide")

ROOT = Path.cwd()
setup_logging(ROOT)
log = get_logger("dashboard")


# feature 37 — curated prompt templates (section/energy/BPM angle)
from musictrain.templates import PROMPT_TEMPLATES as _TEMPLATES
from musictrain.templates import MODELS as _TEMPLATES_MODELS


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
  .mt-header { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; flex-wrap: wrap;
    position: sticky; top: 0; z-index: 50; background: rgba(15,18,32,.92); backdrop-filter: blur(8px); }
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
  .mt-header { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; flex-wrap: wrap;
    position: sticky; top: 0; z-index: 50; background: rgba(244,246,251,.92); backdrop-filter: blur(8px); }
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


def _theme_vars(light: bool) -> str:
    """Override Streamlit's own theme CSS variables so every native widget
    (markdown text, headings, links, labels, inputs) follows the toggle.
    This is what makes the toggle actually change text colors — Streamlit
    paints most text via ``var(--text-color)`` and friends."""
    if light:
        return """
  :root, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    --background-color: #f4f6fb !important;
    --secondary-background-color: #ffffff !important;
    --quiet-background-color: #eef1f8 !important;
    --text-color: #1c2333 !important;
    --heading-color: #111827 !important;
    --link-color: #2563eb !important;
    --small-link-color: #2563eb !important;
  }
"""
    return """
  :root, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    --background-color: #0f1220 !important;
    --secondary-background-color: #141a2e !important;
    --quiet-background-color: #141a2e !important;
    --text-color: #e6e9f2 !important;
    --heading-color: #f2f4fb !important;
    --link-color: #9db4ff !important;
    --small-link-color: #9db4ff !important;
  }
"""


_FONTS = {
    "System": "'-apple-system', BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "Inter": "'Inter', system-ui, sans-serif",
    "Space Grotesk": "'Space Grotesk', system-ui, sans-serif",
    "Lexend": "'Lexend', system-ui, sans-serif",
    "Sora": "'Sora', system-ui, sans-serif",
    "JetBrains Mono": "'JetBrains Mono', ui-monospace, monospace",
    "Serif": "Georgia, 'Times New Roman', serif",
    "Mono": "ui-monospace, 'SF Mono', Menlo, monospace",
}

# Google Fonts specs, loaded only when the matching font is picked.
_WEB_FONTS = {
    "Inter": "Inter:wght@400;500;600;700;800",
    "Space Grotesk": "Space+Grotesk:wght@400;500;600;700",
    "Lexend": "Lexend:wght@400;500;600;700",
    "Sora": "Sora:wght@400;500;600;700",
    "JetBrains Mono": "JetBrains+Mono:wght@400;500;700",
}


def _theme_css() -> str:
    mode = st.session_state.get("mt_theme_mode", "dark")
    light = (_os_theme() if mode == "system" else mode) == "light"
    base = _LIGHT_CSS if light else _DARK_CSS
    accent = _ACCENTS.get(st.session_state.get("mt_accent_label"), "#5b8cff")
    font_name = st.session_state.get("mt_font_name", "System")
    font = _FONTS.get(font_name, _FONTS["System"])
    import_url = _WEB_FONTS.get(font_name, "")
    fonts_import = (
        f"@import url('https://fonts.googleapis.com/css2?family={import_url}&display=swap');"
        if import_url else ""
    )
    vars_css = _theme_vars(light)
    return base + f"""
<style>
{fonts_import}
{vars_css}
  :root, .stApp {{ --mt-accent: {accent}; --mt-accent-2: #7c5cff; }}
  .stApp, [data-testid="stSidebar"], [data-testid="stMetric"], .stButton > button,
  [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
  [data-testid="stMarkdownContainer"] {{ transition: background-color .3s ease, color .3s ease, border-color .3s ease; }}
  .stButton > button[kind="primary"] {{ background: linear-gradient(135deg, var(--mt-accent) 0%, var(--mt-accent-2) 100%); }}
  /* font picker (feature: custom theming) */
  :root, .stApp {{ --mt-font: {font}; }}
  html, body, .stApp {{ font-family: var(--mt-font) !important; }}
  /* shimmer skeletons + hover lift + player styles */
  @keyframes mt-shimmer {{ 0% {{ background-position: -400px 0; }} 100% {{ background-position: 400px 0; }} }}
  [data-testid="stSkeleton"] {{ background: linear-gradient(90deg, rgba(128,128,128,.10) 25%, rgba(128,128,128,.22) 37%, rgba(128,128,128,.10) 63%);
    background-size: 400px 100%; animation: mt-shimmer 1.4s ease-in-out infinite; border-radius: 10px; }}
  [data-testid="stMetric"] {{ transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease, background-color .3s ease; }}
  [data-testid="stMetric"]:hover {{ transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,.22); }}
  .stButton > button {{ transition: transform .12s ease, box-shadow .12s ease, border-color .15s ease, background-color .3s ease; }}
  .stButton > button:hover {{ transform: translateY(-1px); }}
  .stTabs [data-baseweb="tab"] {{ transition: color .15s ease, background-color .15s ease; }}
  .mt-player-wrap {{ border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 10px 12px; background: rgba(128,128,128,.06); }}
  .mt-player-canvas {{ width: 100%; border-radius: 9px; cursor: pointer; background: #0b0e1a; display: block; }}
  .mt-card-sec {{ color: #9db4ff; font-size: .72rem; letter-spacing: .18em; text-transform: uppercase; margin: 14px 0 6px; }}
  .mt-card-line {{ font-size: 1.05rem; line-height: 1.7; }}
  /* uikit components */
  .mt-chip {{ display:inline-block; font-size:.72rem; color:#9aa3c0;
    border:1px solid rgba(255,255,255,.12); border-radius:999px; padding:2px 9px; margin:2px 4px 2px 0;
    transition: transform .12s ease, border-color .12s ease, background-color .2s ease; }}
  .mt-chip b {{ color:#eef1fb; }}
  .mt-chip:hover {{ transform: translateY(-1px); border-color: var(--mt-accent); background: color-mix(in srgb, var(--mt-accent) 10%, transparent); }}
  .mt-tile {{ background: rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.08);
    border-radius:12px; padding:12px 14px; margin:4px 0; }}
  .mt-tile-l {{ font-size:.74rem; color:#9aa3c0; }}
  .mt-tile-v {{ font-size:1.35rem; font-weight:700; color:#eef1fb; }}
  .mt-tile-d {{ font-size:.72rem; color:#7ee2a8; }}

  /* ---- loader & polish upgrades ---- */
  /* gradient page title */
  .mt-header .mt-title {{ background: linear-gradient(120deg, var(--mt-accent), var(--mt-accent-2));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }}
  /* animated gradient progress bars */
  [data-testid="stProgress"] > div {{ border-radius: 999px; overflow: hidden;
    background: rgba(128,128,128,.16); height: 10px !important; }}
  [data-testid="stProgress"] > div > div {{ background: linear-gradient(90deg, var(--mt-accent), var(--mt-accent-2)) !important;
    border-radius: 999px; position: relative; overflow: hidden; transition: width .25s ease; }}
  [data-testid="stProgress"] > div > div::after {{ content: ''; position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.4), transparent);
    animation: mt-sweep 1.2s linear infinite; }}
  @keyframes mt-sweep {{ 0% {{ transform: translateX(-100%); }} 100% {{ transform: translateX(100%); }} }}
  /* skeleton loading cards (avataar + lines) */
  .mt-sk-wrap {{ display: flex; flex-direction: column; gap: 10px; }}
  .mt-sk-card {{ display: flex; gap: 12px; align-items: center; border: 1px solid rgba(128,128,128,.15);
    border-radius: 14px; padding: 14px; background: rgba(128,128,128,.05); }}
  .mt-sk-avatar, .mt-sk-line {{ background: linear-gradient(90deg, rgba(128,128,128,.12) 25%, rgba(128,128,128,.26) 37%, rgba(128,128,128,.12) 63%);
    background-size: 400px 100%; animation: mt-shimmer 1.4s ease-in-out infinite; }}
  .mt-sk-avatar {{ width: 38px; height: 38px; border-radius: 50%; flex-shrink: 0; }}
  .mt-sk-body {{ flex: 1; display: flex; flex-direction: column; gap: 8px; }}
  .mt-sk-line {{ height: 12px; border-radius: 8px; }}
  .mt-sk-line.w80 {{ width: 80%; }} .mt-sk-line.w60 {{ width: 60%; }} .mt-sk-line.w40 {{ width: 40%; }}
  /* custom scrollbars */
  ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
  ::-webkit-scrollbar-thumb {{ background: rgba(128,128,128,.3); border-radius: 8px;
    border: 2px solid transparent; background-clip: content-box; }}
  ::-webkit-scrollbar-thumb:hover {{ background-color: rgba(128,128,128,.5); }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  /* selection + focus rings */
  ::selection {{ background: color-mix(in srgb, var(--mt-accent) 38%, transparent); }}
  button:focus-visible, input:focus-visible, textarea:focus-visible, [role="button"]:focus-visible {{
    outline: 2px solid var(--mt-accent) !important; outline-offset: 2px; }}
  /* primary button glow + press */
  .stButton > button[kind="primary"] {{ box-shadow: 0 4px 20px -4px color-mix(in srgb, var(--mt-accent) 60%, transparent); }}
  .stButton > button:active {{ transform: translateY(0) scale(.98); }}
  .stDownloadButton > button {{ transition: transform .12s ease, box-shadow .12s ease, border-color .15s ease; }}
  .stDownloadButton > button:hover {{ transform: translateY(-1px); border-color: var(--mt-accent); }}
  /* tabs: accent underline on active */
  .stTabs [data-baseweb="tab"][aria-selected="true"] {{ color: var(--mt-accent) !important;
    box-shadow: inset 0 -2px 0 var(--mt-accent); }}
  .stTabs [data-baseweb="tab"]:hover {{ color: var(--mt-accent); }}
  /* expanders + dataframes + links */
  [data-testid="stExpander"] summary:hover {{ background: rgba(128,128,128,.06); border-radius: 12px; }}
  [data-testid="stDataFrame"] tbody tr:hover {{ background: rgba(128,128,128,.07); }}
  a {{ color: var(--mt-accent); }}
  a:hover {{ text-decoration: underline; }}
  [data-baseweb="popover"] [data-baseweb="menu"] {{ border-radius: 12px; }}
  /* toggle switch accent */
  [data-testid="stWidgetLabel"] {{ color: var(--mt-accent); }}
  @media (max-width: 768px) {{
    .mt-header {{ flex-direction: column; align-items: flex-start; gap: 4px; }}
    [data-testid="stMetric"] {{ padding: 10px 12px; }}
  }}
</style>
"""


_OS_THEME_HTML = """
<script>
(function(){
  var m = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
  var dark = m ? m.matches : true;
  window.parent.postMessage({type:'streamlit:setComponentValue', value: dark ? 'dark' : 'light'}, '*');
})();
</script>
"""


def _os_theme() -> str:
    """Detect the OS color scheme (feature 67) — one-shot, cached in session state."""
    cached = st.session_state.get("mt_os_theme")
    if cached in ("dark", "light"):
        return cached
    try:
        import streamlit.components.v1 as components

        val = components.html(_OS_THEME_HTML, height=0)
        if val in ("dark", "light"):
            st.session_state["mt_os_theme"] = val
            return val
    except Exception:  # noqa: BLE001
        pass
    return "dark"


_ACCENTS = {
    "💙 Blue": "#5b8cff", "💜 Violet": "#7c5cff", "🩵 Cyan": "#22c1dc",
    "💚 Green": "#2fbf71", "🩷 Pink": "#ff5c8a", "🧡 Amber": "#ffa53c",
}


def _toggle_theme() -> None:
    """Theme selector (dark / light / system) + accent + font pickers.

    The sidebar and the Settings page share one canonical widget key per
    control (``mt_theme_mode`` / ``mt_accent_label`` / ``mt_font_name``), so
    the widget value *is* the app state — no defaults derived from state and
    no key pops, which previously churned widget IDs and forced a second
    click. The sidebar copy is hidden on the Settings page, which renders
    its own controls with the same keys.
    """
    st.segmented_control(
        "🎨 Theme", ["dark", "light", "system"], key="mt_theme_mode",
        help="Dark / light / follow your OS — applies instantly.",
    )
    st.selectbox("🖌️ Accent", list(_ACCENTS.keys()), key="mt_accent_label")
    st.selectbox("✒️ Font", list(_FONTS.keys()), key="mt_font_name")


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
    "🎤 Lyrics","🪄 Prompt builder","📏 Check BPM","🎬 Visualize","🏷️ Labels","📊 Compare","🧹 Hygiene",
    "🏆 Leaderboard","📈 Training","🔬 Analytics","🎯 Eval","🎧 Listening","✂️ Annotate","🧪 Campaign",
    "🪵 Logs","🧮 Metrics Lab","📦 Model Ops","📡 Ops & Alerts","⚙️ Settings"];
  var SHORTCUTS = {g:"🎛️ Generate", l:"🏆 Leaderboard", c:"📊 Compare", h:"🧹 Hygiene",
    i:"📋 Inventory", n:"🔧 Normalize", m:"🏷️ Metadata", b:"📏 Check BPM",
    t:"📈 Training", a:"🔬 Analytics", v:"🎬 Visualize", e:"🎯 Eval", s:"⚙️ Settings"};
  var box = document.getElementById("mt-palette");
  var input = document.getElementById("mt-palette-input");
  var results = document.getElementById("mt-palette-results");
  var visible = false;
  var selected = 0;

  function post(label) {
    window.parent.postMessage({type: "streamlit:setComponentValue", value: label}, "*");
  }
  function render(filter) {
    var q = (filter || "").toLowerCase();
    var hits = PAGES.filter(function (p) { return p.toLowerCase().indexOf(q) !== -1; });
    if (selected >= hits.length) selected = Math.max(0, hits.length - 1);
    results.innerHTML = hits.map(function (p, idx) {
      var bg = idx === selected ? "background:#5b8cff33;" : "";
      return '<div class="mt-pal-item" data-i="' + idx + '" style="padding:9px 12px;border-radius:9px;cursor:pointer;color:#eef1fb;' + bg + '">'
        + p + '</div>';
    }).join("");
    var items = results.querySelectorAll(".mt-pal-item");
    items.forEach(function (el) {
      el.addEventListener("click", function () { post(el.textContent); });
    });
    return hits;
  }
  function toggle(show) {
    visible = show; box.style.display = show ? "block" : "none";
    if (show) { input.value = ""; selected = 0; render(""); input.focus(); }
  }
  document.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); toggle(!visible); return; }
    if (e.key === "Escape") { toggle(false); return; }
    if (visible) {
      if (e.key === "ArrowDown") { e.preventDefault(); selected++; render(input.value); return; }
      if (e.key === "ArrowUp") { e.preventDefault(); selected = Math.max(0, selected - 1); render(input.value); return; }
      if (e.key === "Enter") {
        e.preventDefault();
        var hits = render(input.value);
        if (hits[selected]) post(hits[selected]);
        return;
      }
      render(input.value); return;
    }
    var tag = (e.target && e.target.tagName) || "";
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

    # feature 12 — tiny CLAP trend sparkline in the sidebar
    _sidebar_sparkline()


def _sidebar_sparkline() -> None:
    """Feature 12 — mini CLAP trend over recent eval rows."""
    import altair as alt

    ev = ROOT / "metadata" / "eval_results.jsonl"
    if not ev.exists():
        return
    pts = []
    for ln in ev.open():
        try:
            r = json.loads(ln)
            if r.get("clap_score") is not None:
                pts.append({"run": len(pts), "clap": float(r["clap_score"])})
        except Exception:  # noqa: BLE001
            continue
    if len(pts) < 2:
        return
    df = pd.DataFrame(pts)
    chart = (
        alt.Chart(df)
        .mark_line(color="#7c5cff", point=False)
        .encode(x=alt.X("run:Q", axis=None), y=alt.Y("clap:Q", scale=alt.Scale(zero=False), axis=None),
                tooltip=["run", "clap"])
        .properties(height=46)
    )
    st.altair_chart(chart, width="stretch")
    st.caption("CLAP trend (last eval rows)")


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


def _skeleton_cards(n: int = 2) -> None:
    """Richer shimmer loader: skeleton cards with an avatar circle + lines."""
    cards = "".join(
        '<div class="mt-sk-card"><div class="mt-sk-avatar"></div>'
        '<div class="mt-sk-body"><div class="mt-sk-line w80"></div>'
        '<div class="mt-sk-line w60"></div><div class="mt-sk-line w40"></div></div></div>'
        for _ in range(n)
    )
    st.markdown(f'<div class="mt-sk-wrap">{cards}</div>', unsafe_allow_html=True)


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


# --------------------------------------------------------------------------- #
# pro player (features 1/2/13) — canvas waveform/spec + click-to-seek + live FFT
# --------------------------------------------------------------------------- #
def _pro_player(path: str, key: str, mode: str = "wave") -> None:
    """Canvas audio player: click-to-seek waveform or spectrogram, live FFT bars."""
    import base64

    import numpy as np

    try:
        import librosa

        y, sr = librosa.load(str(path), sr=16000, mono=True)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"player unavailable: {exc}")
        return
    try:
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
    except Exception:  # noqa: BLE001
        b64 = ""

    env = (np.abs(y[:: max(1, len(y) // 400)]) * 400).clip(0, 255).astype(int).tolist()
    dur = len(y) / sr

    spec_img: list = []
    if mode != "wave":
        win, hop_s, n_bins = 1024, 512, 24
        cols = []
        for start in range(0, len(y) - win, hop_s):
            mag = np.abs(np.fft.rfft(y[start:start + win] * np.hanning(win)))
            seg = int(len(mag) / n_bins)
            cols.append([float(mag[k * seg:(k + 1) * seg].max()) for k in range(n_bins)])
            if len(cols) >= 96:
                break
        if cols:
            arr = np.array(cols).T
            spec_img = (arr / (arr.max() or 1.0) * 255).astype(int).tolist()

    _JS = (
        "<div class='mt-player-wrap'>"
        "<audio id='mt-a-KEY' src='data:audio/wav;base64,B64' preload='metadata'></audio>"
        "<canvas id='mt-c-KEY' class='mt-player-canvas' height='120'></canvas>"
        "<div style='display:flex;gap:10px;align-items:center;margin-top:6px;flex-wrap:wrap;'>"
        "<button id='mt-p-KEY' style='border-radius:9px;border:1px solid rgba(255,255,255,.2);"
        "background:rgba(255,255,255,.08);color:#eef1fb;padding:4px 14px;cursor:pointer;'>▶</button>"
        "<span id='mt-t-KEY' style='color:#9aa3c0;font-size:.78rem;font-family:ui-monospace,monospace;'></span>"
        "<label style='color:#9aa3c0;font-size:.78rem;margin-left:auto;cursor:pointer;'>"
        "<input type='checkbox' id='mt-f-KEY'> ⚡ Live FFT</label></div>"
        "<canvas id='mt-fc-KEY' class='mt-player-canvas' height='42' style='display:none;margin-top:6px;'></canvas>"
        "</div><script>"
        "(function(){"
        "var a=document.getElementById('mt-a-KEY'),cv=document.getElementById('mt-c-KEY'),c2=cv.getContext('2d');"
        "var env=ENV,spec=SPEC,dur=DUR;"
        "function draw(){"
        "var w=cv.width=cv.clientWidth*2,h=cv.height=120;c2.fillStyle='#0b0e1a';c2.fillRect(0,0,w,h);"
        "var pct=a.currentTime/dur;"
        "if(spec.length){"
        "var bw=w/spec[0].length,bh=h/spec.length;"
        "for(var c=0;c<spec[0].length;c++){for(var r=0;r<spec.length;r++){"
        "var v=spec[r][c];c2.fillStyle='rgb('+Math.round(v)+','+Math.round(v*0.6)+',255)';"
        "c2.fillRect(c*bw,r*bh,bw+0.5,bh+0.5);}}"
        "}else{"
        "var n=env.length,bw=w/n;"
        "for(var i=0;i<n;i++){var hh=env[i]/255*h*0.9;c2.fillStyle='#5b8cff';c2.fillRect(i*bw,h-hh,bw+0.5,hh);}"
        "}"
        "c2.fillStyle='rgba(255,255,255,.85)';c2.fillRect(pct*w-1,0,2,h);}"
        "cv.addEventListener('click',function(e){var r=cv.getBoundingClientRect();a.currentTime=(e.clientX-r.left)/r.width*dur;draw();});"
        "document.getElementById('mt-p-KEY').addEventListener('click',function(){if(a.paused){a.play();}else{a.pause();}});"
        "a.addEventListener('timeupdate',function(){var m=Math.floor(a.currentTime/60),s=Math.floor(a.currentTime%60);"
        "document.getElementById('mt-t-KEY').textContent=m+':'+(s<10?'0':'')+s+' / '+Math.floor(dur)+':00';draw();});"
        "var routed=false;"
        "document.getElementById('mt-f-KEY').addEventListener('change',function(e){"
        "var ffc=document.getElementById('mt-fc-KEY');ffc.style.display=e.target.checked?'block':'none';if(!e.target.checked){return;}"
        "var AC=window.AudioContext||window.webkitAudioContext;if(!AC){return;}"
        "if(!routed){var ac=new AC(),src=ac.createMediaElementSource(a),an=ac.createAnalyser();an.fftSize=128;"
        "src.connect(an);an.connect(ac.destination);routed=true;}"
        "var w=ffc.width=ffc.clientWidth*2,h=ffc.height=42,data=new Uint8Array(128);"
        "(function loop(){if(ffc.style.display==='none'){return;}requestAnimationFrame(loop);"
        "an.getByteFrequencyData(data);var f2=ffc.getContext('2d');f2.fillStyle='#0b0e1a';f2.fillRect(0,0,w,h);var bw=w/data.length;"
        "for(var i=0;i<data.length;i++){f2.fillStyle='rgb(91,140,255)';f2.fillRect(i*bw,h-data[i]/255*h,bw,data[i]/255*h);}"
        "})();});"
        "draw();"
        "})();</script>"
    )
    html = (
        _JS.replace("KEY", key)
        .replace("B64", b64)
        .replace("ENV", json.dumps(env))
        .replace("SPEC", json.dumps(spec_img))
        .replace("DUR", repr(dur))
    )
    try:
        import streamlit.components.v1 as components

        components.html(html, height=236)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"player unavailable: {exc}")


def _fullscreen_audio(path: str, key: str) -> None:
    """Feature 11 — fullscreen review of a clip with native controls."""
    import base64

    try:
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
    except Exception as exc:  # noqa: BLE001
        st.caption(f"fullscreen unavailable: {exc}")
        return
    html = (
        f'<audio id="mt-fs-{key}" src="data:audio/wav;base64,{b64}" controls style="width:100%;"></audio>'
        f'<script>(function(){{var a=document.getElementById("mt-fs-{key}");'
        f'var b=document.createElement("button");b.textContent="⛶ Fullscreen review";'
        f'b.style.cssText="width:100%;margin-top:6px;padding:6px;border-radius:9px;'
        f'border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.06);color:#eef1fb;cursor:pointer;";'
        f'b.onclick=function(){{var el=a.parentElement;if(el.requestFullscreen)el.requestFullscreen();'
        f'else if(el.webkitRequestFullscreen)el.webkitRequestFullscreen();}};'
        f'a.parentElement.appendChild(b);}})();</script>'
    )
    try:
        import streamlit.components.v1 as components

        components.html(html, height=120)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"fullscreen unavailable: {exc}")


def _tempo_tap(key: str = "tap") -> float:
    """Feature 8 — tap-along BPM estimator (median interval)."""
    taps = st.session_state.setdefault(f"mt_taps_{key}", [])
    c1, c2 = st.columns([1, 2])
    if c1.button("👆 Tap", key=f"tap_btn_{key}", width="stretch"):
        now = time.time()
        if taps and now - taps[-1] > 2.5:
            taps.clear()
        taps.append(now)
        st.session_state[f"mt_taps_{key}"] = taps[-8:]
        st.rerun()
    if len(taps) >= 2:
        import statistics

        iv = [b - a for a, b in zip(taps, taps[1:]) if b - a > 0.15]
        if iv:
            bpm = max(40.0, min(240.0, statistics.median(60.0 / i for i in iv)))
            c2.metric("Tempo (tapped)", f"{bpm:.1f} BPM", f"{len(taps)} taps")
            return bpm
    c2.caption("Tap along with the beat — the median interval gives your BPM.")
    return 0.0


def _segment_plays(audio_path: str, segs: list, key: str) -> None:
    """Feature 20 — pick a structure segment and play it from its start time."""
    if not segs:
        st.caption("no segments detected")
        return
    opts = {}
    for i, s in enumerate(segs):
        start = float(s.get("start") or 0.0)
        dur = float(s.get("duration") or 0.0)
        opts[f"#{i} {s.get('role', '?')} · {start:.1f}s (+{dur:.0f}s)"] = s
    pick = st.selectbox("▶ Play a section", list(opts), key=f"segpick_{key}")
    s = opts[pick]
    start = float(s.get("start") or 0.0)
    st.audio(audio_path, start_time=start)
    st.caption(f"role **{s.get('role', '?')}** · starts {start:.1f}s · {float(s.get('duration') or 0):.1f}s long")


def _segment_inspector() -> None:
    """Feature 21 — inspect, play, rename and delete segment files on disk."""
    seg_dir = ROOT / "data" / "segments"
    if not seg_dir.exists():
        st.caption("no data/segments directory yet")
        return
    files = sorted(seg_dir.glob("*.wav"))
    if not files:
        st.caption("no segment files on disk")
        return
    st.caption(f"{len(files)} segment file(s) on disk")
    pick = st.selectbox("Segment", [f.name for f in files], key="seg_insp_pick")
    p = seg_dir / pick
    st.audio(str(p))
    c1, c2 = st.columns(2)
    new_name = c1.text_input("Rename to", value=pick, key="seg_insp_rename")
    if c1.button("✏️ Rename", key="seg_insp_ren_btn", width="stretch") and new_name and new_name != pick:
        p.rename(seg_dir / new_name)
        st.toast(f"renamed → {new_name}")
        st.rerun()
    if c2.button("🗑 Delete", key="seg_insp_del_btn", width="stretch"):
        p.unlink(missing_ok=True)
        st.toast("segment deleted")
        st.rerun()


def _provenance_view() -> None:
    """Feature 46 — every segment traces back to its source beat."""
    segs = _read_json("segments.json")
    if isinstance(segs, dict):
        segs = segs.get("segments", [])
    if not segs:
        st.caption("no segment records — run `segment` first")
        return
    rows = []
    for s in segs:
        src = s.get("source") or s.get("source_file") or ""
        rows.append({
            "segment": Path(str(s.get("path") or s.get("file") or "?")).name,
            "source beat": Path(str(src)).name if src else "?",
            "start (s)": s.get("start"),
            "duration (s)": s.get("duration"),
            "role": s.get("role", ""),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"{len(rows)} segment(s) · every row links to its source beat in data/clean")


# --------------------------------------------------------------------------- #
# lyrics tools (features 22/23/26/28/29/30)
# --------------------------------------------------------------------------- #
_RHYME_COLORS = ["#5b8cff", "#ff5c8a", "#2fbf71", "#ffa53c", "#c084fc", "#22c1dc", "#f472b6", "#a3e635"]


def _rhyme_key(word: str) -> str:
    w = word.lower().strip(".,!?;:'\"()[]-—_")
    if not w:
        return ""
    v = "aeiouy"
    idx = max([i for i, ch in enumerate(w) if ch in v], default=-1)
    if idx == -1:
        return w[-2:]
    return (w[idx:].rstrip("estd") or w[-1])


def _rhyme_preview(text: str) -> None:
    """Feature 23 — color-code end-of-line rhymes in the editor."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        st.caption("type lines to see rhyme groups")
        return
    groups: dict = {}
    order: list = []
    for ln in lines:
        words = ln.split()
        k = _rhyme_key(words[-1]) if words else ""
        if k and k not in groups:
            groups[k] = _RHYME_COLORS[len(order) % len(_RHYME_COLORS)]
            order.append(k)
    out = []
    for ln in lines:
        words = ln.split()
        if words:
            last = words[-1]
            col = groups.get(_rhyme_key(last), "#9aa3c0")
            out.append(html_escape(" ".join(words[:-1]))
                       + f' <span style="color:{col};font-weight:700;">{html_escape(last)}</span>')
        else:
            out.append(html_escape(ln))
    st.markdown("<br>".join(out), unsafe_allow_html=True)
    st.caption(f"🎨 {len(order)} rhyme group(s) — end-words sharing a color rhyme")


def _syllable_stats(text: str) -> None:
    """Feature 22 — live per-line syllable counts while editing."""
    from musictrain.lyrictools import count_syllables

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return
    counts = [count_syllables(ln) for ln in lines]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lines", len(lines))
    c2.metric("Avg syll/line", f"{sum(counts) / len(counts):.1f}")
    c3.metric("Max syll", max(counts))
    c4.metric("Total syll", sum(counts))
    chips = " ".join(f'<span class="mt-chip">{c} syll</span>' for c in counts[:28])
    st.markdown(chips, unsafe_allow_html=True)


_STYLE_PRESETS = {
    "🔥 Aggressive banger": {"artist": "dababy", "mood": "aggressive", "topic": "clout",
                            "weights": {"topic": 1.0, "mood": 1.4, "flow": 1.2, "ad_libs": 1.0}},
    "🌧️ Emotional pain": {"artist": "lil durk", "mood": "emotional", "topic": "pain",
                          "weights": {"topic": 1.2, "mood": 1.4, "flow": 0.8, "ad_libs": 0.4}},
    "💰 Money flex": {"artist": "future", "mood": "confident", "topic": "money",
                       "weights": {"topic": 1.0, "mood": 1.0, "flow": 1.0, "ad_libs": 0.8}},
    "🕊️ Melodic sad": {"artist": "juice wrld", "mood": "melancholic", "topic": "heartbreak",
                        "weights": {"topic": 1.3, "mood": 1.2, "flow": 0.9, "ad_libs": 0.3}},
    "🚀 Trap come-up": {"artist": "gunna", "mood": "smooth", "topic": "come-up",
                        "weights": {"topic": 1.1, "mood": 1.0, "flow": 1.1, "ad_libs": 0.7}},
    "👑 King energy": {"artist": "kanye west", "mood": "confident", "topic": "boss",
                       "weights": {"topic": 1.2, "mood": 1.1, "flow": 0.9, "ad_libs": 0.5}},
}


_SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _melody_hints(key: str) -> None:
    """Feature 28 — singable scale notes for the detected key."""
    import re as _re

    m = _re.match(r"([A-Ga-g][#b]?)\s+(major|minor|m)", key or "")
    if m:
        root_s, kind = m.group(1), ("minor" if m.group(2) == "m" else m.group(2))
    else:
        m2 = _re.match(r"([A-Ga-g][#b]?)", key or "")
        root_s, kind = (m2.group(1) if m2 else "A"), "minor"
    root_s = root_s[0].upper() + (root_s[1:] if len(root_s) > 1 else "")
    root = _NOTE_NAMES.index(root_s) if root_s in _NOTE_NAMES else 9
    notes = [_NOTE_NAMES[(root + s) % 12] for s in _SCALES.get(kind, _SCALES["minor"])]
    chips = " ".join(f'<span class="mt-chip"><b>{n}</b></span>' for n in notes)
    st.markdown(f"🎵 {kind} scale — singable notes: {chips}", unsafe_allow_html=True)
    st.caption("hint: anchor vocal melodies on these notes; the root and 5th are the safest targets.")


def _song_assembler(result) -> None:
    """Feature 29 — assemble sections into a song order, then copy/download."""
    sections = result.sections
    roles = [s["role"] for s in sections]
    order = st.multiselect("Section order (song assembly)", roles, default=roles, key="ly_assembly")
    if not order:
        st.caption("pick at least one section")
        return
    parts = []
    for role in order:
        sec = next((s for s in sections if s["role"] == role), None)
        if sec is None:
            continue
        parts.append(f"[{role}] {sec.get('artist', result.artist)}")
        parts.extend(sec.get("lines", []))
    full = "\n".join(parts)
    st.text(full)
    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇ Assembly .txt", full, file_name=f"assembly_{result.artist}_{result.seed}.txt",
        mime="text/plain", key="ly_assembly_dl", width="stretch",
    )
    _copy_button(full, "ly_assembly_copy")


def _lyric_card_html(result) -> str:
    """Feature 30 — Instagram-style lyric card as a standalone HTML file."""
    parts = []
    for sec in result.sections:
        parts.append(f'<div class="mt-card-sec">{html_escape(str(sec["role"]).upper())} — {html_escape(str(sec.get("artist", result.artist)))}</div>')
        for ln in sec.get("lines", []):
            parts.append(f'<div class="mt-card-line">{html_escape(str(ln))}</div>')
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html_escape(str(result.artist))} — lyric card</title><style>"
        "body{margin:0;font-family:Georgia,serif;background:#0f1220;display:flex;justify-content:center;}"
        ".card{max-width:520px;width:92vw;margin:24px auto;padding:44px 30px;border-radius:22px;"
        "background:linear-gradient(160deg,#151b30 0%,#0e1120 60%,#1a1028 100%);"
        "border:1px solid rgba(255,255,255,.12);box-shadow:0 18px 50px rgba(0,0,0,.5);}"
        ".mt-card-sec{color:#9db4ff;font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;margin:16px 0 8px;}"
        ".mt-card-line{color:#f2f4fb;font-size:1.06rem;line-height:1.75;}"
        "</style></head><body><div class=\"card\">" + "".join(parts) + "</div></body></html>"
    )


def _lyric_card_download(result) -> None:
    st.download_button(
        "🖼 Lyric card (.html)", _lyric_card_html(result),
        file_name=f"card_{result.artist}_{result.seed}.html", mime="text/html",
        key="ly_card_dl", width="stretch",
    )


# --------------------------------------------------------------------------- #
# training / eval / ops helpers (features 31/32/33/35/37/38/39)
# --------------------------------------------------------------------------- #
def _mlflow_train_runs(cfg):
    try:
        from musictrain.experiments import _configure

        ml = _configure(cfg)
        if ml is None:
            return pd.DataFrame()
        exp = ml.get_experiment_by_name(cfg.mlflow.experiment_name)
        if exp is None:
            return pd.DataFrame()
        return ml.search_runs(experiment_ids=[exp.experiment_id])
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.fragment(run_every=5)
def _live_training(cfg) -> None:
    """Feature 31 — auto-refreshing training telemetry."""
    df = _mlflow_train_runs(cfg)
    c = st.columns(4)
    if df.empty:
        c[0].metric("MLflow runs", 0)
        c[1].metric("last train loss", "—")
        c[2].metric("best eval loss", "—")
        c[3].metric("recent job", "—")
    else:
        c[0].metric("MLflow runs", len(df))
        tr = df[df.get("tags.task") == "train"] if "tags.task" in df else df
        tloss = tr.get("metrics.train_loss") if "metrics.train_loss" in tr else None
        eloss = tr.get("metrics.eval_loss") if "metrics.eval_loss" in tr else None
        c[1].metric("last train loss",
                    f"{pd.to_numeric(tloss, errors='coerce').dropna().iloc[-1]:.3f}"
                    if tloss is not None and pd.to_numeric(tloss, errors="coerce").notna().any() else "—")
        c[2].metric("best eval loss",
                    f"{pd.to_numeric(eloss, errors='coerce').dropna().min():.3f}"
                    if eloss is not None and pd.to_numeric(eloss, errors="coerce").notna().any() else "—")
        recent = _log_tail(1)
        c[3].metric("recent job", (recent[14:60] if recent.startswith(("▶", "✓", "✗", "■")) else "—"))
    tail = [ln for ln in _log_tail(30).splitlines() if any(k in ln for k in ("step", "loss", "adapter", "Epoch"))][-6:]
    st.caption("live tail (auto-refresh 5s)")
    if tail:
        st.code("\n".join(tail), language=None)


def _restore_best(cfg) -> None:
    """Feature 32 — copy the best adapter (lowest eval loss) to checkpoints/best."""
    import shutil

    cands = []
    lr = ROOT / "checkpoints" / "lyrics"
    if lr.exists():
        for d in sorted(lr.glob("*")):
            if not d.is_dir():
                continue
            ts = d / "trainer_state.json"
            try:
                el = float(json.loads(ts.read_text()).get("eval_loss") or float("inf"))
                cands.append((el, str(d)))
            except Exception:  # noqa: BLE001
                continue
    audio = ROOT / "adapters"
    if audio.exists() and any(f.is_file() for f in audio.iterdir()):
        cands.append((float("inf"), str(audio)))
    if not cands:
        st.info("No adapters yet — train `musictrain finetune` or `train-lyrics` first.")
        return
    cands.sort(key=lambda kv: kv[0])
    el, src = cands[0]
    dst = ROOT / "checkpoints" / "best"
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.glob("*"):
        if f.is_file():
            f.unlink()
    if Path(src).is_dir():
        for f in Path(src).glob("*"):
            if f.is_file():
                shutil.copy2(f, dst / f.name)
    st.success(f"Restored best adapter ({el:.3f} eval loss) → checkpoints/best")
    st.code(str(dst), language=None)


def _checkpoint_gallery(cfg) -> None:
    """Feature 33 — browse MLflow runs with metrics + one-click adoption."""
    from musictrain.experiments import search_runs

    df = search_runs(cfg)
    if df.empty:
        st.info("No MLflow runs yet — run an eval or finetune first.")
        return
    cols = [c for c in ("run_id", "name", "task", "model", "device", "clap_score", "deviation", "duration_s", "seed") if c in df]
    st.dataframe(df[cols].head(40), width="stretch", hide_index=True)
    st.caption(f"{len(df)} run(s) · {len(df[df.get('task') == 'eval'] if 'task' in df else df)} eval")
    models = [m for m in df["model"].dropna().unique() if str(m).strip()] if "model" in df else []
    if not models:
        st.caption("no model ids logged yet — run an eval first")
        return
    pick = st.selectbox("Adopt a model for the Generate page", [""] + sorted(models), key="ckpt_gallery_pick")
    if pick:
        st.session_state["mt_model_hint"] = pick
        st.session_state["nav"] = "🎛️ Generate"
        st.toast(f"{pick} → Generate page")
        st.rerun()


def _eval_drilldown(rows: list) -> None:
    """Feature 35 — drill into failing prompts with a fix suggestion."""
    fails = [
        r for r in rows
        if r.get("status") != "ok"
        or (r.get("detected_bpm") and r.get("bpm_target")
            and abs(float(r["detected_bpm"]) - float(r["bpm_target"])) > 8.0)
    ]
    if not fails:
        st.success("No failing prompts — every clip is in tolerance.")
        return
    opts = {
        f"#{i} {r.get('section', '?')} {r.get('bpm_target')}BPM → {r.get('detected_bpm', '?')} (dev {r.get('deviation', '?')})": r
        for i, r in enumerate(fails)
    }
    pick = st.selectbox("Failing prompt", list(opts), key="ev_drill_pick")
    r = opts[pick]
    st.write(r.get("prompt") or r.get("description") or "")
    c1, c2 = st.columns(2)
    c1.metric("Target BPM", r.get("bpm_target"))
    c2.metric("Detected BPM", r.get("detected_bpm", "—"))
    dev = float(r.get("deviation") or 0.0)
    target = float(r.get("bpm_target") or 0.0)
    det = float(r.get("detected_bpm") or 0.0)
    if target and det:
        ratio = det / target
        if ratio > 1.5:
            sug = "detected ≈ target × 2 — try a slower/denser tempo anchor in the prompt"
        elif ratio < 0.66:
            sug = "detected ≈ target ÷ 2 — try an explicit faster phrasing or fewer long notes"
        else:
            sug = f"off by {dev:.1f} BPM — tighten the tempo phrasing (e.g. 'exactly {target:g} BPM')"
        st.info(f"💡 {sug}")
    ap = r.get("audio_path")
    if ap and Path(ap).exists():
        st.audio(str(ap))


@st.fragment(run_every=10)
def _adapter_watch() -> None:
    """Feature 37 — detect new adapters and prompt a gate run."""
    adapters = ROOT / "adapters"
    if not adapters.exists():
        return
    mtime = max((f.stat().st_mtime for f in adapters.glob("*") if f.is_file()), default=0.0)
    last = st.session_state.get("mt_adapter_mtime", 0.0)
    if last == 0.0:
        st.session_state["mt_adapter_mtime"] = mtime
        st.caption("👀 Watching adapters/ — new adapters prompt a gate run here.")
        return
    if mtime > last:
        st.warning(f"🆕 New adapter in adapters/ ({time.strftime('%H:%M', time.localtime(mtime))}) — refresh the leaderboard.")
        if st.button("▶ Run eval gate now", key="ev_watch_run"):
            st.session_state["mt_adapter_mtime"] = mtime
            st.session_state["ev_gate_requested"] = True
            st.rerun()
    else:
        st.session_state["mt_adapter_mtime"] = mtime
        st.caption("👀 Watching adapters/ — no new adapters since last check.")


def _prompt_set_builder() -> None:
    """Feature 38 — edit the eval prompt set in the UI."""
    evp = ROOT / "metadata" / "eval_prompts.jsonl"
    current = evp.read_text() if evp.exists() else ""
    edited = st.text_area(
        "eval_prompts.jsonl — one JSON object per line", value=current,
        height=200, key="ev_builder_edit",
        placeholder='{"id": "chorus_140_A#min", "section": "chorus", "genre": "melodic trap", "bpm": 140, "key": "A# minor", "mood": "dark", "instruments": "piano, 808 bass", "energy": 0.8, "seed": 42, "description": "chorus, 140 BPM, ..."}',
    )
    c1, c2 = st.columns(2)
    if c1.button("💾 Save prompt set", key="ev_builder_save", width="stretch"):
        lines = [ln for ln in edited.splitlines() if ln.strip()]
        bad = [ln for ln in lines if not _try_json(ln)]
        if bad:
            st.error(f"{len(bad)} invalid JSON line(s) — fix and retry")
        else:
            evp.parent.mkdir(parents=True, exist_ok=True)
            evp.write_text(edited)
            st.toast(f"Saved {len(lines)} prompt(s) → metadata/eval_prompts.jsonl")
            st.rerun()
    if c2.button("🔄 Rebuild standard set", key="ev_builder_rebuild", width="stretch"):
        from musictrain.evalset import build

        build(ROOT, force=True)
        st.toast("Standard eval set rebuilt")
        st.rerun()
    st.caption("fields: id · section · genre · bpm · key · mood · instruments · energy · seed · description")


def _try_json(line: str):
    try:
        return json.loads(line)
    except Exception:  # noqa: BLE001
        return None


def _run_cost_table(cfg) -> None:
    """Feature 39 — per-run cost/elapsed breakdown from MLflow."""
    from musictrain.experiments import search_runs

    df = search_runs(cfg)
    if df.empty:
        st.caption("No MLflow runs yet — costs appear after the first eval/finetune.")
        return
    show = df.copy()
    if "duration_s" in show:
        show["duration_s"] = pd.to_numeric(show["duration_s"], errors="coerce")
    cols = [c for c in ("run_id", "name", "task", "model", "device", "duration_s", "clap_score") if c in show]
    st.dataframe(show[cols].head(30), width="stretch", hide_index=True)
    if "duration_s" in show and show["duration_s"].notna().any():
        st.caption(f"⏱ total logged time: {show['duration_s'].sum():.1f}s across {len(show)} run(s)")


# --------------------------------------------------------------------------- #
# data helpers (features 42/44/45/49/50)
# --------------------------------------------------------------------------- #
def _dataset_browser() -> None:
    """Feature 42 — filter labeled segments + batch re-label in place."""
    lab = ROOT / "metadata" / "labels.csv"
    if not lab.exists():
        st.caption("no labels.csv yet — run `beatlabels` or scaffold one on the Labels page")
        return
    df = pd.read_csv(lab)
    man = ROOT / "metadata" / "manifest.jsonl"
    merged = df.copy()
    if man.exists():
        try:
            mdf = pd.DataFrame([json.loads(ln) for ln in man.read_text().splitlines() if ln.strip()])
            if {"path", "bpm", "key"}.issubset(mdf.columns):
                mdf = mdf.copy()
                mdf["stem"] = mdf["path"].apply(lambda p: Path(str(p)).stem)
                df2 = df.copy()
                df2["stem"] = df2["source_id"].astype(str)
                merged = df2.merge(mdf[["stem", "bpm", "key"]], on="stem", how="left")
        except Exception:  # noqa: BLE001
            merged = df.copy()

    f1, f2, f3, f4 = st.columns(4)
    genres = sorted({str(v) for v in merged["genre"].dropna().astype(str) if str(v).strip()})
    moods = sorted({str(v) for v in merged["mood"].dropna().astype(str) if str(v).strip()})
    g = f1.multiselect("Genre", genres, key="dsb_genre")
    m = f2.multiselect("Mood", moods, key="dsb_mood")
    secs = sorted({str(v) for v in merged["section"].dropna().astype(str) if str(v).strip()})
    s = f3.multiselect("Section", secs, key="dsb_sec")
    q = f4.text_input("Search source_id", key="dsb_q")

    view = merged.copy()
    if g:
        view = view[view["genre"].astype(str).apply(lambda v: any(x in v for x in g))]
    if m:
        view = view[view["mood"].astype(str).apply(lambda v: any(x in v for x in m))]
    if s:
        view = view[view["section"].astype(str).isin(s)]
    if q.strip():
        view = view[view["source_id"].astype(str).str.contains(q.strip(), case=False)]

    st.caption(f"{len(view)}/{len(merged)} labeled segment(s) match")
    st.dataframe(view, width="stretch", hide_index=True)

    sel = st.multiselect("Batch re-label rows", view.index.tolist(), key="dsb_sel",
                         format_func=lambda i: f"{i}: {view.loc[i, 'source_id']}")
    if sel:
        with st.popover("Apply values", width="stretch"):
            b_mood = st.text_input("mood (pipe-separated)", "", key="dsb_bm")
            b_inst = st.text_input("instruments (pipe-separated)", "", key="dsb_bi")
            b_sec = st.selectbox("section", [""] + secs, key="dsb_bs")
            if st.button("Apply & save", type="primary", key="dsb_apply") and sel:
                for i in sel:
                    if b_mood:
                        merged.loc[i, "mood"] = b_mood
                    if b_inst:
                        merged.loc[i, "instruments"] = b_inst
                    if b_sec:
                        merged.loc[i, "section"] = b_sec
                merged.to_csv(lab, index=False)
                st.toast(f"Re-labeled {len(sel)} row(s) → labels.csv")
                st.rerun()


def _label_check_ui() -> None:
    """Feature 44 — vocabulary consistency check with one click."""
    lab = ROOT / "metadata" / "labels.csv"
    if not lab.exists():
        st.info("no labels.csv to check yet")
        return
    if st.button("🔍 Run consistency check", key="lab_check_run"):
        from musictrain.labels import check

        st.session_state["lab_issues"] = check(lab)
    issues = st.session_state.get("lab_issues")
    if issues is None:
        st.caption(
            "Checks the vocabulary: unknown genre/section/mood/instrument terms, "
            "missing source_id/license/description, and duplicate ids."
        )
        return
    if not issues:
        st.success("labels.csv is fully consistent with the vocabulary ✓")
        return
    st.warning(f"{len(issues)} issue(s)")
    st.dataframe(pd.DataFrame({"issue": issues}), width="stretch", hide_index=True)


def _dataset_export_import() -> None:
    """Feature 45 — one-click dataset zip export + import."""
    import io
    import zipfile

    st.markdown("#### 📦 Dataset export / import")
    e1, e2 = st.columns(2)
    with e1:
        if st.button("⬇ Export dataset (.zip)", key="set_ds_export", width="stretch"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for rel in ("metadata/labels.csv", "metadata/manifest.jsonl", "metadata/segments.json"):
                    p = ROOT / rel
                    if p.exists():
                        z.write(p, rel)
                for split in ("train", "val", "test"):
                    d = ROOT / "data" / split
                    if d.exists():
                        for f in sorted(d.glob("*.wav"))[:100]:
                            z.write(f, f"data/{split}/{f.name}")
            st.session_state["mt_ds_zip"] = buf.getvalue()
        z = st.session_state.get("mt_ds_zip")
        if z:
            st.download_button(
                "💾 Download dataset.zip", data=z, file_name="dataset.zip",
                mime="application/zip", key="set_ds_dl", width="stretch",
            )
    with e2:
        zf = st.file_uploader("Import dataset.zip (labels/manifest/segments + splits)", type=["zip"], key="set_ds_imp")
        if zf and st.button("📂 Import & extract", key="set_ds_imp_btn", width="stretch"):
            with zipfile.ZipFile(io.BytesIO(zf.getbuffer())) as z:
                z.extractall(ROOT)
            st.toast("Dataset imported → data/ + metadata/")
            st.rerun()


def _crash_panel() -> None:
    """Feature 50 — recent exceptions with stack traces + copy."""
    cands = [ROOT / "logs" / "musictrain.log", ROOT / "metadata" / "musictrain.log"]
    text = ""
    for c in cands:
        if c.exists():
            t = c.read_text()
            if len(t) > len(text):
                text = t
    if not text:
        st.caption("no log file yet — errors will surface here")
        return
    blocks, cur = [], []
    for ln in text.splitlines():
        if "Traceback" in ln or "ERROR" in ln:
            cur = [ln]
        elif cur:
            cur.append(ln)
            if not ln.strip() and len(cur) > 3:
                blocks.append("\n".join(cur))
                cur = []
    if cur:
        blocks.append("\n".join(cur))
    blocks = blocks[-5:]
    if not blocks:
        st.caption("no errors/tracebacks in the log — clean run")
        return
    st.warning(f"{len(blocks)} recent error block(s)")
    for b in blocks:
        with st.expander(b.splitlines()[0][:90]):
            st.code(b, language=None)
    _copy_button("\n\n".join(blocks), "crash_copy")


def _session_resume() -> None:
    """Feature 49 — restore the last page/theme/pins from metadata/session.json."""
    sfile = ROOT / "metadata" / "session.json"
    try:
        if sfile.exists():
            data = json.loads(sfile.read_text())
            for k in ("nav", "mt_theme_mode", "mt_accent_label", "mt_font_name", "mt_focus", "mt_pinned", "mt_lang"):
                if k not in st.session_state and k in data:
                    st.session_state[k] = data[k]
    except Exception:  # noqa: BLE001
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
            log.exception("dashboard job %r failed", label)
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
    Cancel button is rendered once (outside the polling loop): a widget key must
    be unique per script run, and re-rendering it each tick previously raised
    StreamlitDuplicateElementKey.
    """
    out: dict = {}

    def _worker() -> None:
        try:
            out["result"] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            log.exception("dashboard cancellable job %r failed", label)
            out["error"] = exc

    _record_job(label, fn, args, kwargs)
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    slot = st.empty()
    cancel_slot = st.empty()
    cancelled = False
    # Render the Cancel button once, outside the polling loop.
    with cancel_slot.container():
        if st.button("⏹ Cancel", key="cancel_job_btn"):
            cancelled = True

    pct = 0.0
    while thread.is_alive() and not cancelled:
        pct = min(pct + 0.03, 0.92)
        slot.progress(pct, text=label)
        time.sleep(0.12)
    cancel_slot.empty()
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
            log.exception("dashboard live job %r failed", label)
            state["error"] = exc

    _record_job(label, fn, args, kwargs)
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    slot = st.empty()
    cancel_slot = st.empty()
    if cancel_kw:
        # Render the Cancel button once, outside the polling loop.
        with cancel_slot.container():
            if st.button("⏹ Cancel", key="cancel_live_btn"):
                state["cancelled"] = True
    while thread.is_alive() and not state["cancelled"]:
        done, total = state["done"], state["total"]
        frac = (done / total) if total else 0.0
        slot.progress(frac, text=f"{label} — {done}/{total}")
        time.sleep(0.15)
    if cancel_kw:
        cancel_slot.empty()
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
        _skeleton_cards(2)
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

    st.markdown("---")
    st.subheader("🗂️ Dataset browser")
    _dataset_browser()

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

    with st.expander("🔍 Segment inspector — play / rename / delete", expanded=False):
        _segment_inspector()
    with st.expander("🧬 Source provenance", expanded=False):
        _provenance_view()


# --------------------------------------------------------------------------- #
# 🎛️ Generate
# --------------------------------------------------------------------------- #
def page_generate() -> None:
    _page_header("🎛️", "Generate audio", "MusicGen on MPS — prompt, guidance, and sampling control.")
    cfg = load_cfg()

    # feature 33 — adopt a model picked in the checkpoint gallery
    hint = st.session_state.pop("mt_model_hint", None)
    if hint:
        h1, h2 = st.columns([4, 1])
        h1.info(f"🖼 Checkpoint gallery → adopt model **{hint}**")
        if h2.button("✅ Adopt as default", key="gen_adopt", width="stretch"):
            cfg.inference.model_name = hint
            cfg.settings.default_model = hint
            cfg.save(ROOT / "configs" / "default.yaml")
            st.session_state["gen_model"] = hint
            st.toast("default model updated")
            st.rerun()

    # feature 37 — template library (apply fills the prompt area)
    t_names = ["— custom —"] + [t.name for t in _TEMPLATES]
    tc1, tc2 = st.columns([3, 1])
    tpl = tc1.selectbox("📚 Template", t_names, index=0, key="gen_tpl")
    if tc2.button("Apply template", key="gen_tpl_apply", width="stretch"):
        hit = next((t for t in _TEMPLATES if t.name == tpl), None)
        if hit:
            st.session_state["gen_prompt"] = hit.prompt
            st.rerun()

    prompt = st.text_area(
        "Prompt",
        value="cinematic hip hop chorus, 96 BPM, A minor, dark piano, "
        "deep 808 bass, wide strings, powerful drums",
        key="gen_prompt",
    )
    c1, c2, c3, c4 = st.columns(4)
    _model_ids = [m.model_id for m in _TEMPLATES_MODELS]
    _id_to_name = {m.model_id: m.name for m in _TEMPLATES_MODELS}
    model = c1.selectbox(
        "Model",
        _model_ids,
        index=0, key="gen_model",
        format_func=lambda mid: _id_to_name.get(mid, mid),
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

    # recent outputs grid (feature 15) + scrolling ticker (feature 24)
    recent = sorted((ROOT / "outputs").glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    if recent:
        st.markdown("---")
        st.subheader("Recent outputs")
        from musictrain import viz

        viz.ticker([p.name for p in recent], "gen")
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
                if report.get("target_bpm"):
                    from musictrain import viz

                    viz.bpm_gauge(float(report["detected_bpm"]),
                                  float(report["target_bpm"]), "chk")
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

    st.markdown("---")
    _label_check_ui()

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
        _skeleton_cards(2)
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

    with st.expander("👆 Tap-to-BPM", expanded=False):
        _tempo_tap("listen")

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
                _pro_player(str(ap), f"lst_{i}", mode="wave")
                _fullscreen_audio(str(ap), f"lst_{i}")
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

    c_save, _c_undo = st.columns([1, 1])
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

    c_save, _c_undo = st.columns([1, 1])
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
# ✂️ Annotate (review / trim / label) — #15
# --------------------------------------------------------------------------- #
def page_annotate() -> None:
    _page_header(
        "✂️", "Annotate",
        "Review a clip, scrub a trim window, and save a labelled slice for curation.",
    )
    from musictrain.report import load_results

    with st.spinner("Loading clips…"):
        rows = load_results(ROOT)
    candidates = [r for r in rows if r.get("audio_path") and Path(r["audio_path"]).exists()]
    if not candidates:
        _skeleton_cards(2)
        st.info("No generated clips yet — run `musictrain eval` first.")
        return

    labels = [f"{r.get('section') or '?'} · {Path(r['audio_path']).name}" for r in candidates]
    idx = st.selectbox("Clip", range(len(candidates)), format_func=lambda i: labels[i], key="ann_clip")
    row = candidates[idx]
    ap = Path(row["audio_path"])

    st.caption(f"**{row.get('prompt')}**")
    _waveform_chart(str(ap), "ann_wave")
    st.audio(str(ap))

    import soundfile as sf

    try:
        info = sf.info(ap)
        dur = float(info.duration)
    except Exception:  # noqa: BLE001
        dur = 0.0

    c1, c2 = st.columns(2)
    start = c1.number_input("Trim start (s)", 0.0, max(dur, 0.0), 0.0, 0.5, key="ann_start")
    end = c2.number_input("Trim end (s)", 0.0, max(dur, 0.0), min(dur, 30.0), 0.5, key="ann_end")

    label = st.text_input("Label (comma-separated tags)", value="", key="ann_label")

    if st.button("💾 Save trimmed slice", type="primary", key="ann_save"):
        if end <= start or dur <= 0:
            st.error("Trim end must be after start and within the clip.")
        else:
            out_dir = ROOT / "data" / "reviewed"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{ap.stem}_trim_{start:.1f}-{end:.1f}.wav"
            try:
                data, fs = sf.read(ap)
                lo = int(start * fs)
                hi = int(end * fs)
                sf.write(out_path, data[lo:hi], fs)
                ann = ROOT / "metadata" / "annotations.jsonl"
                with ann.open("a") as fh:
                    fh.write(json.dumps({
                        "source": str(ap), "trimmed": str(out_path),
                        "start": start, "end": end, "label": label,
                        "prompt": row.get("prompt"),
                    }) + "\n")
                st.success(f"Saved {out_path.name} -> data/reviewed/")
                st.audio(str(out_path))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Trim failed: {exc}")


# --------------------------------------------------------------------------- #
# 🎧 Listening campaign — #7
# --------------------------------------------------------------------------- #
def page_campaign() -> None:
    _page_header(
        "🧪", "Listening campaign",
        "Blind A/B or MOS listening with persisted sessions and rater agreement.",
    )
    from musictrain import listening_campaign as lc

    c1, c2, c3 = st.columns([2, 1, 1])
    name = c1.text_input("Campaign name", value="campaign1", key="camp_name")
    mode = c2.selectbox("Mode", ["ab", "mos"], key="camp_mode")
    seed = c3.number_input("Seed", 0, 9999, 0, key="camp_seed")

    if st.button("🆕 Create campaign", key="camp_create"):
        out = lc.start(ROOT, name, mode=mode, seed=seed)
        if out.get("n_items"):
            st.success(f"Created {name!r} with {out['n_items']} item(s).")
        else:
            st.error(str(out))

    camp = lc.load_campaign(ROOT, name)
    if not camp:
        _skeleton_cards(2)
        st.info("Create a campaign (needs prior eval results) to begin rating.")
        return

    st.caption(f"Campaign **{name}** ({camp['mode']}) — {camp['n_items']} item(s)")
    rater = st.text_input("Rater id", value="rater1", key="camp_rater")

    existing = {(r["rater"], r["item_id"]) for r in lc.load_ratings(ROOT, name)}
    pending = [it for it in camp["items"] if (rater, it["id"]) not in existing]
    if not pending:
        st.success(f"All {camp['n_items']} item(s) rated by {rater} — thank you!")
    else:
        it = pending[0]
        st.markdown(f"**Item {it['id']}** — {it.get('prompt')}")
        with st.container(border=True):
            if it["mode"] == "ab":
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**X**")
                    st.audio(it["x"]["audio_path"])
                with cb:
                    st.markdown("**Y**")
                    st.audio(it["y"]["audio_path"])
                choice = st.radio("Prefer", ["X", "Y", "tie"], horizontal=True, key="camp_choice")
            else:
                st.audio(it["clip"]["audio_path"])
                choice = str(st.slider("MOS score", 1, 5, 3, key="camp_mos"))
            if st.button("Save judgement", type="primary", key="camp_save_rating"):
                rating = None if it["mode"] == "ab" else int(choice)
                lc.record(ROOT, name, rater, it["id"], "X" if it["mode"] == "ab" else str(choice),
                          rating=rating)
                st.rerun()

    a = lc.agreement(ROOT, name)
    if a["n_ratings"]:
        st.metric("Rater agreement (mean majority)", a["agreement"] if a["agreement"] is not None else "—")
        st.caption(f"{a['n_ratings']} rating(s) · {a['n_raters']} rater(s) · {a['n_items']} item(s)")
        if camp["mode"] == "ab":
            with st.expander("🔓 Unblind X/Y"):
                st.json(lc.unblind(ROOT, name))


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

    _evf = ROOT / "metadata" / "eval_results.jsonl"
    _nrows = 0
    if _evf.exists():
        try:
            with _evf.open() as _fh:
                _nrows = sum(1 for _ in _fh)
        except OSError:
            _nrows = 0
    st.caption(f"{len(prompts)} prompts in the set · current result file has {_nrows} rows")

    if st.button("▶ Start eval", type="primary", key="ev_run") or st.session_state.pop("ev_gate_requested", False):
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
            from musictrain import viz

            viz.confetti("eval")
            cols = [c for c in ("prompt", "section", "bpm_target", "detected_bpm", "deviation", "clap_score", "status") if c in results[0]]
            st.dataframe(pd.DataFrame(results)[cols], width="stretch")

    _sched_eval(cfg, secs, int(seeds), int(limit))

    st.markdown("---")
    with st.expander("🔬 Failing-prompt drill-down", expanded=False):
        from musictrain.report import load_results

        _eval_drilldown(load_results(ROOT))
    with st.expander("📝 Prompt-set builder", expanded=False):
        _prompt_set_builder()
    with st.expander("👀 Adapter watcher", expanded=False):
        _adapter_watch()


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


def _eval_rows() -> list:
    """Read eval_results.jsonl into a list of dicts ([] if missing)."""
    p = ROOT / "metadata" / "eval_results.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def page_visualize() -> None:
    """Batch 1+2 - previews & audio: waveforms, spectrograms, chords, structure, stems, CLAP."""
    from musictrain import viz

    _page_header("🎬", "Visualize",
                 "Waveforms, spectrograms, chords, beat grids, structure, stems and prompt adherence.")
    cfg = load_cfg()
    files = viz.scan_audio(ROOT, ["data/clean", "data/raw", "data/segments", "outputs"])
    if not files:
        st.info("No audio found under data/ or outputs/. Generate or normalize some clips first.")
        return

    names = {str(p): f"{p.relative_to(ROOT)}" for p in files}
    sel = st.selectbox("Clip", list(names), format_func=lambda k: names[k], key="viz_clip")

    ev = _eval_rows()
    mean_clap = sum(r.get("clap_score") or 0.0 for r in ev) / len(ev) if ev else 0.0
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        viz.countup_metric("clips", len(files), "viz_clips")
    with m2:
        viz.countup_metric("stems", len(viz.scan_audio(ROOT, ["data/stems"])), "viz_stems")
    with m3:
        viz.countup_metric("eval rows", len(ev), "viz_eval")
    with m4:
        viz.countup_metric("mean CLAP", mean_clap, "viz_clap", decimals=2)

    t_wave, t_spect, t_music, t_struct, t_stems, t_clap, t_player = st.tabs(
        ["🌊 Waveform", "🔬 Spectrogram", "🎼 Beat & chords", "🧩 Structure", "🎚️ Stems", "🔖 CLAP", "🎛 Player"])

    with t_wave:
        viz.waveform(sel, "page")
        rec = _analyze_clip(sel, cfg)
        viz.onset_overlay(sel, rec.get("beat_grid", {}), rec.get("onsets", {}), "page")
    with t_spect:
        kind = st.radio("Kind", ["mel", "chroma"], horizontal=True, key="viz_kind")
        viz.spectrogram(sel, "page", kind=kind)
        with st.expander("🖱 Clickable spectrogram — click anywhere to seek", expanded=False):
            _pro_player(sel, "viz_spec", mode="spec")
    with t_music:
        rec = _analyze_clip(sel, cfg)
        viz.chord_beat_strip(rec.get("chords", []), rec.get("beat_grid", {}), "page")
        viz.chromagram(sel, "page")
    with t_struct:
        rec = _analyze_clip(sel, cfg)
        segs = rec.get("structure", {}).get("segments", [])
        viz.segmented_waveform(sel, segs, "page")
        viz.structure_timeline(sel, segs, "page")
        _segment_plays(sel, segs, "viz_struct")
    with t_player:
        pmode = st.radio("View", ["wave", "spectrogram"], horizontal=True, key="viz_pmode")
        _pro_player(sel, "viz_pro", mode=pmode)
        st.caption("🎛 pro player — click the canvas to seek · ⚡ Live FFT for a realtime spectrum")
    with t_stems:
        stem_files = viz.scan_audio(ROOT, ["data/stems"])
        viz.stem_mixer(stem_files, "page")
    with t_clap:
        if not ev:
            st.caption("run an eval to populate per-tag CLAP scores")
        else:
            ckpt = st.selectbox("Checkpoint", sorted({r.get("checkpoint", "?") for r in ev}),
                                key="viz_ckpt")
            viz.clap_heat_strip([r for r in ev if r.get("checkpoint") == ckpt], "page")

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
        viz.live_dot("watching outputs/")
        viz.live_generation_view(str(ROOT / "outputs"), "page")



def page_training() -> None:
    """Batch 4 - training visuals: HUD, CLAP trend, MLflow matrix, cost, coverage, drift."""
    from musictrain import trainviz

    _page_header("📈", "Training",
                 "Model training and experiment health — read live from MLflow and local metadata.")
    cfg = load_cfg()
    with st.expander("⚡ Live training telemetry", expanded=True):
        _live_training(cfg)
    if st.button("🏆 Restore best checkpoint → checkpoints/best", key="tr_restore_best"):
        _restore_best(cfg)

    trainviz.training_hud(cfg)

    st.markdown("---")
    st.subheader("CLAP trend (eval runs)")
    trainviz.clap_trend(cfg)

    t_ml, t_cost, t_cov, t_drift = st.tabs(
        ["🧪 Experiments", "⚡ Cost", "🗺️ Coverage", "📉 Drift"])

    with t_ml:
        trainviz.matrix_heatmap(cfg)
        st.markdown("#### Recent runs")
        trainviz.metrics_panel(cfg)
    with t_cost:
        trainviz.cost_chart(cfg)
    with t_cov:
        trainviz.coverage_heatmap(cfg)
        st.markdown("#### Split")
        trainviz.split_donut(cfg)
    with t_drift:
        trainviz.drift_timeline(cfg)
        st.markdown("#### Weight delta")
        trainviz.weight_diff_heatmap(cfg)

    st.markdown("#### LR schedule")
    trainviz.lr_schedule()


def page_metricslab() -> None:
    """Metrics Lab: every headline metric as animated gauges/sparklines/tiles."""
    from musictrain import uikit
    from musictrain.report import load_results

    _page_header("🧮", "Metrics Lab",
                 "All headline metrics in one live view — gauges, sparklines and tiles.")
    cfg = load_cfg()
    rows = load_results(cfg.project_root)
    if not rows:
        st.info("No eval results yet — run `musictrain eval` to populate.")
        return

    claps = [r["clap_score"] for r in rows if r.get("clap_score") is not None]
    devs = [abs(r["deviation"]) for r in rows if r.get("deviation") is not None]
    ok = sum(1 for r in rows if r.get("status") == "ok")
    mean_clap = sum(claps) / len(claps) if claps else 0.0
    mean_dev = sum(devs) / len(devs) if devs else 0.0
    ok_rate = ok / len(rows) if rows else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        uikit.gauge(mean_clap, "mean CLAP", 1.0, key="g_clap")
    with c2:
        uikit.gauge(1.0 - min(1.0, mean_dev * 5), "BPM adherence", 1.0, key="g_dev")
    with c3:
        uikit.gauge(ok_rate, "ok-rate", 1.0, key="g_ok")
    with c4:
        uikit.metric_tile("Prompts", f"{len(rows)}", f"{ok} ok")

    st.markdown("#### CLAP distribution")
    uikit.sparkline(sorted(claps), label="CLAP (sorted)")
    uikit.sparkline(sorted(devs), label="|BPM deviation| (sorted)")

    st.markdown("#### Per-genre tiles")
    by_g = {}
    for r in rows:
        g = (r.get("genre") or "default").strip() or "default"
        by_g.setdefault(g, []).append(r.get("clap_score"))
    cols = st.columns(min(4, max(1, len(by_g))))
    for i, (g, vals) in enumerate(by_g.items()):
        vals = [v for v in vals if v is not None]
        with cols[i % len(cols)]:
            uikit.metric_tile(g, f"{sum(vals) / len(vals):.3f}" if vals else "—", f"{len(vals)} clip(s)")


def page_modelops() -> None:
    """Model Ops: registry, backup, lineage, checksums and rollback."""
    _page_header("📦", "Model Ops",
                 "Registry, backups, lineage and artifact integrity.")
    cfg = load_cfg()
    t_reg, t_backup, t_lineage, t_gallery = st.tabs(["📚 Registry", "🗄️ Backups", "🧬 Lineage", "🖼️ Checkpoint gallery"])

    with t_gallery:
        _checkpoint_gallery(cfg)

    with t_reg:
        try:
            from musictrain.registry_ml import list_models

            models = list_models(cfg)
            if not models:
                st.info("No registered models — run `musictrain register <checkpoint>`.")
            else:
                st.dataframe(pd.DataFrame(models), width="stretch")
        except Exception as exc:  # noqa: BLE001 - registry is best-effort
            st.info(f"Registry unavailable: {exc}")

    with t_backup:
        from musictrain.backup import list_backups

        bks = list_backups(cfg)
        if not bks:
            st.info("No backups yet — run `musictrain backup --task snapshot`.")
        else:
            st.dataframe(pd.DataFrame(bks), width="stretch")
        if st.button("🔄 Snapshot now", key="bk_snap"):
            from musictrain.backup import snapshot

            snapshot(cfg)
            st.toast("Backup created.")

    with t_lineage:
        from musictrain.modelops import lineage_graph

        lg = lineage_graph(cfg)
        edges = lg.get("edges", [])
        if not edges:
            st.info("No lineage recorded — run `musictrain modelops --task lineage`.")
        else:
            for e in edges:
                st.markdown(f"- `{e.get('parent', '?')}` → `{e.get('child', '?')}`")


def page_ops() -> None:
    """Ops & Alerts: cost, runlog, alerts and config lint."""
    _page_header("📡", "Ops & Alerts",
                 "Cost attribution, runlog, alert thresholds and config health.")
    cfg = load_cfg()
    t_cost, t_log, t_alert, t_lint = st.tabs(["⚡ Cost", "🪵 Runlog", "🚨 Alerts", "🧾 Config lint"])

    with t_cost:
        from musictrain.cost import cost_summary

        s = cost_summary(cfg)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total energy", f"{s.get('total_kwh', 0):.3f} kWh")
        with c2:
            st.metric("Runs", s.get("runs", 0))
        st.markdown("#### Per-run breakdown")
        _run_cost_table(cfg)

    with t_log:
        from musictrain.telemetry import read_runlog

        logs = read_runlog(cfg.project_root, limit=50)
        if not logs:
            st.info("No runlog entries.")
        else:
            st.dataframe(pd.DataFrame(logs), width="stretch")

    with t_alert:
        from musictrain.alerts import check_alerts

        violations = check_alerts(cfg)
        if not violations:
            st.success("No threshold violations — all clear.")
        else:
            st.warning(f"{len(violations)} violation(s)")
            st.dataframe(pd.DataFrame(violations), width="stretch")

    with t_lint:
        from musictrain.modelops import lint

        res = lint(cfg)
        issues = res.get("issues", [])
        if not issues:
            st.success("Config lint passed.")
        else:
            st.dataframe(pd.DataFrame(issues), width="stretch")


def page_analytics() -> None:
    """Batch 5 - analytics: embeddings, active learning, curation, leaderboard, checkpoints."""
    from musictrain import analyteviz

    _page_header("🔬", "Analytics",
                 "Corpus and model analytics — embeddings, active learning, curation, leaderboard and checkpoints.")
    cfg = load_cfg()

    st.subheader("Leaderboard")
    analyteviz.leaderboard_bar(cfg)
    analyteviz.two_run_overlay(cfg)

    t_emb, t_active, t_aug, t_cur, t_ckpt, t_train = st.tabs(
        ["🧬 Embeddings", "🎯 Active learning", "🔀 Augmentation",
         "⭐ Curation", "🗂️ Checkpoints", "⏱️ Training"])

    with t_emb:
        analyteviz.umap_scatter(cfg)
    with t_active:
        analyteviz.active_scatter(cfg)
    with t_aug:
        analyteviz.augmentation_panel()
    with t_cur:
        analyteviz.curation_histogram(cfg)
    with t_ckpt:
        analyteviz.checkpoint_timeline(cfg)
    with t_train:
        analyteviz.early_stop_curve()
        analyteviz.token_gauge(cfg)
        analyteviz.model_size_cost(cfg)


# --------------------------------------------------------------------------- #
# 🎤 Lyrics (rap/lyrics pivot — beat analysis -> lyrics generation)
# --------------------------------------------------------------------------- #
_LY_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _analyze_beat_for_lyrics(path: str, cfg: Config) -> dict:
    """Deep-analyze a beat for lyric shaping: key + swing + structure."""
    from musictrain.audio.analysis import key_candidates, swing_ratio

    cache = st.session_state.setdefault("ly_analysis", {})
    if path in cache:
        return cache[path]
    rec = _analyze_clip(path, cfg)
    try:
        from musictrain.audio.features import load_audio

        y, sr = load_audio(Path(path), sr=cfg.analysis.sr)
        rec = dict(rec)
        rec["key"] = key_candidates(y, sr, top_k=cfg.analysis.key_top_k)
        rec["swing"] = swing_ratio(y, sr, hop_length=cfg.analysis.hop_length)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"key/swing analysis failed: {exc}")
    cache[path] = rec
    return rec


def _ly_ctx(rec: dict, artist: str, mood: str, topic: str, seed: int,
            roles: list, negative: list, weights: dict):
    from musictrain import lyrics as L

    key = (rec.get("key") or {}).get("key", "A minor")
    bpm = float((rec.get("beat_grid") or {}).get("tempo", 140.0))
    swing = (rec.get("swing") or {}).get("feel", "straight")
    energies = [s.get("energy", 0.0) for s in (rec.get("structure") or {}).get("segments", [])]
    energy = sum(energies) / len(energies) if energies else 0.5
    bars = {"intro": 4, "verse": 16, "hook": 8, "chorus": 8, "bridge": 8,
            "pre-chorus": 4, "outro": 4, "full-song": 16}
    structure = [L.SectionSpec(role=r, bars=bars.get(r, 8)) for r in roles if r]
    return L.BeatContext(
        bpm=bpm, key=key, swing=swing, energy=energy,
        artist=artist, mood=mood, topic=topic, structure=structure,
        negative=negative, weights=weights, seed=seed,
    )


def _ly_push_version(result) -> None:
    """Keep the last N generations for undo/restore (feature #8)."""
    versions = st.session_state.setdefault("ly_versions", [])
    label = f"{result.artist} · {result.mood} · seed {result.seed}"
    versions.insert(0, (label, result))
    st.session_state["ly_versions"] = versions[:10]


def _ly_arrangement_bar(sections: list) -> None:
    """Feature 36 — colored arrangement map of the generated sections."""
    palette = {
        "intro": "#5b8cff", "verse": "#34d399", "hook": "#f59e0b",
        "chorus": "#f472b6", "bridge": "#a78bfa", "outro": "#64748b",
        "pre-chorus": "#22d3ee", "full-song": "#94a3b8",
    }
    total = max(1, sum(s.get("bars", 8) for s in sections))
    cells = "".join(
        f'<div style="flex:{s.get("bars", 8)};background:{palette.get(s.get("role", "verse"), "#888")};'
        f'border-radius:6px;text-align:center;font-size:.72rem;color:#0b0e17;'
        f'padding:8px 2px;margin:1px;overflow:hidden;white-space:nowrap;">'
        f'{s.get("role", "?")}</div>'
        for s in sections
    )
    st.markdown(
        f'<div style="display:flex;width:100%;">{cells}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"🧭 arrangement map · {total} bars total")


def page_lyrics() -> None:
    """Beat analysis → rapper-style lyrics: structure editor, per-section slots,
    style presets, refinement (re-roll / regenerate / restyle), rating + profile."""
    from musictrain import artists as A
    from musictrain import lyrics as L
    from musictrain import lyrictools as LT
    from musictrain import lyricrating as rating
    from musictrain import lyricsprefs as prefs

    _page_header("🎤", "Lyrics",
                 "Upload a beat, read its tempo/key/structure, and write rapper-style lyrics to it.")
    cfg = load_cfg()

    # ---------------- beat source ---------------- #
    up = st.file_uploader("Upload a beat", type=["wav", "mp3", "flac", "ogg", "m4a"], key="ly_up")
    clean_dir = ROOT / "data" / "clean"
    clean_files = sorted(
        p for p in (clean_dir.iterdir() if clean_dir.exists() else [])
        if p.is_file() and p.suffix.lower() in _LY_AUDIO_EXT
    )
    pick = st.selectbox(
        "…or pick from data/clean", ["(none)"] + [p.name for p in clean_files], key="ly_pick",
    )

    beat_path: Path | None = None
    if up is not None:
        udir = ROOT / "data" / "lyric_uploads"
        udir.mkdir(parents=True, exist_ok=True)
        beat_path = udir / up.name
        beat_path.write_bytes(up.getbuffer())
    elif pick != "(none)":
        beat_path = clean_dir / pick

    if beat_path is None or not beat_path.exists():
        _skeleton_cards(2)
        st.info("Upload a beat or pick one from data/clean to start writing to it.")
        return

    st.audio(str(beat_path))
    with st.spinner("Analyzing beat (tempo, key, swing, structure)…"):
        rec = _analyze_beat_for_lyrics(str(beat_path), cfg)

    key = (rec.get("key") or {}).get("key", "?")
    bpm = (rec.get("beat_grid") or {}).get("tempo", 0.0)
    swing = (rec.get("swing") or {}).get("feel", "?")
    segs = (rec.get("structure") or {}).get("segments", [])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BPM", f"{bpm:.1f}")
    m2.metric("Key", key)
    m3.metric("Swing", swing)
    m4.metric("Sections", len(segs))
    _melody_hints(key)

    detected_roles = [s.get("role", "verse") for s in segs]
    st.caption("detected structure: " + " → ".join(detected_roles))

    # chord → mood/topic suggestion (feature #8)
    sug = LT.suggest_from_chords(rec.get("chords", []), key)
    sc1, sc2, sc3 = st.columns([4, 1, 1])
    sc1.caption(f"💡 chord suggestion: **{sug['mood']}** / **{sug['topic']}** ({sug['reason']})")
    if sc2.button("Apply suggestion", key="ly_apply_sug", width="stretch"):
        st.session_state["ly_sug"] = (sug["mood"], sug["topic"])
    if st.session_state.get("ly_sug"):
        if sc3.button("Clear", key="ly_clear_sug", width="stretch"):
            st.session_state.pop("ly_sug", None)
            st.rerun()

    # vocal-detection gate (feature #7) — warn if the beat already has vocals
    vocal = rec.get("vocal") or st.session_state.get("ly_vocal") or {}
    if vocal.get("verdict") == "vocal":
        st.warning("🎙️ This beat already contains vocals — you may be writing over a hook.")
    elif vocal.get("verdict") == "instrumental":
        st.success("🎹 Instrumental beat — clean space for your lyrics.")
    elif cfg.clap.enabled:
        if st.button("🔍 Check for vocals", key="ly_vocal_check"):
            with st.spinner("Detecting vocals (CLAP)…"):
                from musictrain.audio.analysis import vocal_instrumental

                v = vocal_instrumental(cfg, beat_path)
                if v:
                    st.session_state["ly_vocal"] = v
                    st.rerun()

    # ---------------- style ---------------- #
    st.subheader("🎛️ Style")
    preset_name = st.selectbox(
        "⚡ Style preset", ["(custom)"] + list(_STYLE_PRESETS), key="ly_style_preset",
    )
    if preset_name != "(custom)":
        p = _STYLE_PRESETS[preset_name]
        fa = A.get_artist(p["artist"])
        if fa:
            st.session_state["ly_artist"] = fa.name
        if p["mood"] in A.MOODS:
            st.session_state["ly_mood"] = p["mood"]
        st.session_state["ly_topic"] = p["topic"]
    a1, a2, a3, a4 = st.columns(4)
    artist = a1.selectbox("Artist", A.artist_names(), key="ly_artist")
    genre_names = A.genre_names()
    genre = a2.selectbox("Genre template", ["(none)"] + genre_names, key="ly_genre")
    g = A.get_genre(genre)
    mood = a3.selectbox("Mood", ["(auto)"] + list(A.MOODS), key="ly_mood")
    topic = a4.text_input("Topic", value="", placeholder="pain / loyalty / success…", key="ly_topic")

    if st.button("🎲 Surprise me", key="ly_surprise"):
        r = prefs.random_recipe(ROOT, seed=int(time.time()) % 10**6)
        st.session_state["ly_recipe"] = r
        st.rerun()

    # apply a random recipe if present
    rr = st.session_state.get("ly_recipe")
    if rr:
        artist = next((a.name for a in A.ARTISTS if a.id == rr.get("artist")), artist)
        mood = rr.get("mood", mood)
        topic = rr.get("topic", topic)

    if g and mood == "(auto)" and not topic:
        mood = g.moods[0]
        topic = g.topics[0]

    # ---------------- structure editor (features #1/#2/#5/#6) ---------------- #
    with st.expander("🧩 Structure editor, presets & features", expanded=True):
        preset = st.selectbox(
            "📐 Arrangement preset", ["(custom)"] + LT.arrangement_names(), key="ly_preset",
        )
        if preset != "(custom)":
            arrangement = ",".join(r for r, _ in LT.ARRANGEMENTS[preset])
        else:
            default_arr = ",".join(detected_roles) or "intro,verse,hook,verse,hook,outro"
            arrangement = st.text_input(
                "Arrangement (roles, comma-separated)", value=default_arr, key="ly_arrangement",
            )
        roles = [r.strip().lower() for r in arrangement.split(",") if r.strip()]
        st.caption("roles: intro · verse · hook · chorus · bridge · pre-chorus · outro · full-song")

        overrides = st.text_input(
            "Per-section topic slots (role=topic; …)",
            placeholder="verse=pain; hook=success; outro=reflection", key="ly_overrides",
        )
        override_map = {}
        for part in overrides.split(";"):
            if "=" in part:
                r, t = part.split("=", 1)
                override_map[r.strip().lower()] = t.strip()

        # multi-artist feature mode (#6) + duet (#5)
        features = st.text_input(
            "Feature artists (role=artist; …)",
            placeholder="verse=drake; verse=future+gunna (duet)", key="ly_features",
        )
        feature_map = {}   # role -> artist id
        duet_map = {}      # role -> second artist id
        for part in features.split(";"):
            if "=" in part:
                r, aid = part.split("=", 1)
                parts = [p.strip() for p in aid.split("+") if p.strip()]
                fa = A.get_artist(parts[0]) if parts else None
                if fa:
                    feature_map[r.strip().lower()] = fa.id
                    if len(parts) > 1:
                        fa2 = A.get_artist(parts[1])
                        if fa2:
                            duet_map[r.strip().lower()] = fa2.id

    # ---------------- weights + negatives (features #28/#29) ---------------- #
    with st.expander("⚖️ Prompt weights & negatives"):
        w1, w2, w3 = st.columns(3)
        w_topic = w1.slider("topic weight", 0.0, 2.0, 1.0, 0.1, key="ly_w_topic")
        w_mood = w2.slider("mood weight", 0.0, 2.0, 1.0, 0.1, key="ly_w_mood")
        w_flow = w3.slider("flow weight", 0.0, 2.0, 1.0, 0.1, key="ly_w_flow")
        w_adlibs = st.slider("ad-lib weight", 0.0, 2.0, 0.6, 0.1, key="ly_w_adlibs")
        neg_text = st.text_input(
            "Negative (banned words/topics, comma-separated)", value="", key="ly_neg",
        )

    # ---------------- generate ---------------- #
    seed = st.number_input("Seed", 0, 10**6, int(st.session_state.get("ly_seed", 42)), key="ly_seed")
    variation = st.slider(
        "🎲 Variation", 1, 10, 3, key="ly_variation",
        help="Higher = more seed drift for wider template choice.",
    )
    gen_seed = seed + (variation - 3) * 11
    st.caption(f"effective seed: **{gen_seed}** (base {seed} + variation {(variation - 3) * 11:+d})")
    weights = {"topic": w_topic, "mood": w_mood, "flow": w_flow, "ad_libs": w_adlibs}
    negatives = [n.strip() for n in neg_text.split(",") if n.strip()]
    negatives += prefs.negatives(ROOT)

    artist_id = (A.get_artist(artist) or A.ARTISTS[0]).id
    sug_applied = st.session_state.get("ly_sug")
    if sug_applied:
        eff_mood, eff_topic = sug_applied
    else:
        eff_mood = mood if mood != "(auto)" else "dark"
        eff_topic = topic or (g.topics[0] if g else "struggle")
    ctx = _ly_ctx(rec, artist_id, eff_mood, eff_topic, seed, roles, negatives, weights)
    # apply per-section topic slots (feature #2) + feature artists (#6) + duets (#5)
    if override_map or feature_map:
        ctx.structure = [
            L.SectionSpec(
                role=s.role, bars=s.bars,
                topic=override_map.get(s.role, ""),
                artist=feature_map.get(s.role, ""),
                artist2=duet_map.get(s.role, ""),
            )
            for s in ctx.structure
        ]

    gcol1, gcol2, gcol3 = st.columns([3, 1, 1])
    if gcol1.button("✍️ Write lyrics", type="primary", key="ly_generate", width="stretch"):
        ctx.seed = gen_seed
        result = L.generate(ctx)
        st.session_state["ly_result"] = result
        st.session_state["ly_ctx"] = ctx
        _ly_push_version(result)
        prefs.record_history(ROOT, result.as_dict())
        _log_line(f"lyrics: {result.artist} {int(result.bpm)}bpm {result.mood}/{result.topic} seed={result.seed}")

    if gcol2.button("🎲 Re-roll", key="ly_reroll", width="stretch"):
        ctx.seed = gen_seed + int(time.time()) % 97
        result = L.generate(ctx)
        st.session_state["ly_result"] = result
        st.session_state["ly_ctx"] = ctx
        _ly_push_version(result)

    # style-profile autopilot (feature #10)
    if gcol3.button("🤖 Auto-write", key="ly_autowrite", width="stretch"):
        r = prefs.autopilot(ROOT, seed=seed)
        auto_ctx = _ly_ctx(rec, r.get("artist", artist_id), r.get("mood", "dark"),
                           r.get("topic", "struggle"), seed, roles, negatives, weights)
        result = L.generate(auto_ctx)
        st.session_state["ly_result"] = result
        st.session_state["ly_ctx"] = auto_ctx
        _ly_push_version(result)
        st.rerun()

    result = st.session_state.get("ly_result")

    # ---------------- refinement (features #39/#40/#41) ---------------- #
    if result is not None:
        r1, r2, r3 = st.columns(3)
        sec_role = r1.selectbox(
            "Regenerate section", ["(none)"] + [s["role"] for s in result.sections], key="ly_resec",
        )
        if r1.button("🔁 Regen section", key="ly_resec_btn", width="stretch") and sec_role != "(none)":
            ctx = st.session_state.get("ly_ctx") or ctx
            new_sec = L.regenerate_section(ctx, sec_role, seed=seed + 1)
            for s in st.session_state["ly_result"].sections:
                if s["role"] == sec_role:
                    s.update(new_sec)
            st.rerun()

        restyle_to = r2.selectbox("Style transfer", A.artist_names(), key="ly_restyle")
        if r2.button("🎨 Restyle", key="ly_restyle_btn", width="stretch"):
            target = A.get_artist(restyle_to)
            if target and target.id != result.artist.lower():
                st.session_state["ly_result"] = L.restyle(result, target.id, seed=seed)
                st.rerun()

        if r3.button("⭐ Save favorite", key="ly_fav_btn", width="stretch"):
            prefs.add_favorite(ROOT, f"{result.artist}-{result.mood}-{seed}", {
                "artist": result.artist, "mood": result.mood, "topic": result.topic,
                "seed": seed,
            })
            st.success("favorite saved")

    # ---------------- display ---------------- #
    if result is None:
        _skeleton_cards(3)
        st.info("Write lyrics to generate a version against this beat.")
        return

    _ly_arrangement_bar(result.sections)
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        st.caption(f"**{result.artist}** · {result.mood} · {result.topic} · {int(result.bpm)} BPM · {result.key} · seed {result.seed} · backend {result.backend}")

    # quality-gate report (surfaces retries/fallbacks instead of hiding them)
    if getattr(result, "gate_issues", None):
        st.warning("Quality gate: " + "; ".join(result.gate_issues))
    with c2:
        slug = f"{result.artist}_{int(result.bpm)}bpm_{result.key.replace(' ', '')}_{result.mood}_{result.topic}_seed{result.seed}".replace("/", "-")
        st.download_button("⬇ .txt", result.full_text(), file_name=slug + ".txt", mime="text/plain", key="ly_dl")
    with c3:
        # studio sheet export (feature #9)
        st.download_button("⬇ Studio sheet (.md)", result.to_sheet(), file_name=slug + ".md", mime="text/markdown", key="ly_dl_sheet")
    with c4:
        # LRC karaoke export (feature #9)
        st.download_button("⬇ .lrc", LT.lrc(result), file_name=slug + ".lrc", mime="text/plain", key="ly_dl_lrc")

    # lyrical metrics (feature #6)
    m = LT.metrics(result)
    mm1, mm2, mm3, mm4 = st.columns(4)
    mm1.metric("Flow score", f"{m['flow_score']}/100")
    mm2.metric("Rhyme density", f"{m['rhyme_density']:.0%}")
    mm3.metric("Avg syllables", m["avg_syllables"])
    mm4.metric("Bars", m["bars"])

    for sec in result.sections:
        with st.expander(f"[{sec['role']}] {sec['bars']} bars · {sec['flow']} @ {sec['cadence']} · {sec.get('artist', result.artist)}", expanded=(sec["role"] in ("hook", "chorus"))):
            for ann in LT.annotate_section(sec):
                st.markdown(f"> {ann['line']}  `{ann['syllables']} syll`")
            if sec["ad_libs"]:
                st.caption("ad-libs: " + ", ".join(f"({a})" for a in sec["ad_libs"]))

    # ---------------- in-place line editor (feature #3) ---------------- #
    with st.expander("✏️ Edit lines", expanded=False):
        edited = st.text_area(
            "Full text (edit freely)", value=result.full_text(), height=280, key="ly_editor",
        )
        _syllable_stats(edited)
        _rhyme_preview(edited)
        e1, e2 = st.columns(2)
        if e1.button("💾 Save edits", key="ly_editor_save", width="stretch"):
            st.session_state["ly_edited"] = edited
            st.success("edits saved — download below")
        if st.session_state.get("ly_edited"):
            e2.download_button(
                "⬇ edited .txt", st.session_state["ly_edited"],
                file_name=slug + "_edited.txt", mime="text/plain", key="ly_dl_edited",
                width="stretch",
            )

    # ---------------- song assembly + lyric card (features 29/30) ---------------- #
    with st.expander("🧩 Song assembler & lyric card", expanded=False):
        _song_assembler(result)
        _lyric_card_download(result)

    # ---------------- side-by-side diff (feature #7) ---------------- #
    with st.expander("🔍 Compare to another seed", expanded=False):
        c1, c2 = st.columns(2)
        other_seed = c1.number_input("Other seed", 0, 10**6, seed + 1, key="ly_diff_seed")
        if c2.button("Generate & diff", key="ly_diff_btn", width="stretch"):
            ctx2 = st.session_state.get("ly_ctx") or ctx
            ctx2.seed = int(other_seed)
            st.session_state["ly_diff_result"] = L.generate(ctx2)
        diff_result = st.session_state.get("ly_diff_result")
        if diff_result is not None:
            d1, d2 = st.columns(2)
            with d1:
                st.caption(f"**seed {result.seed}**")
                st.text(result.full_text())
            with d2:
                st.caption(f"**seed {diff_result.seed}**")
                st.text(diff_result.full_text())
            diff_lines = LT.diff_results(result, diff_result)
            if diff_lines:
                st.code("\n".join(diff_lines), language=None)

    # ---------------- version history + undo (feature #8) ---------------- #
    versions = st.session_state.get("ly_versions", [])
    if len(versions) > 1:
        with st.expander("🕘 Version history & undo", expanded=False):
            labels = [v[0] for v in versions]
            pick = st.selectbox("Previous versions", labels, key="ly_ver_pick")
            if st.button("↩️ Restore selected", key="ly_ver_restore", width="stretch"):
                idx = labels.index(pick)
                st.session_state["ly_result"] = versions[idx][1]
                st.rerun()

    # ---------------- project save/load (feature #10) ---------------- #
    with st.expander("💾 Project save / load", expanded=False):
        from musictrain import lyricproject as proj

        p1, p2 = st.columns(2)
        proj_name = p1.text_input("Project name", value="session1", key="ly_proj_name")
        if p1.button("💾 Save project", key="ly_proj_save", width="stretch"):
            proj.save_project(ROOT, proj_name, {
                "beat": str(beat_path),
                "recipe": {"artist": result.artist, "mood": result.mood,
                           "topic": result.topic, "seed": result.seed},
                "structure": [{"role": s["role"], "bars": s["bars"],
                               "artist": s.get("artist", "")}
                              for s in result.sections],
                "result": result.as_dict(),
            })
            st.success(f"project {proj_name!r} saved")
        existing = proj.list_projects(ROOT)
        if existing:
            load_name = p2.selectbox("Load project", existing, key="ly_proj_load_sel")
            if p2.button("📂 Load", key="ly_proj_load", width="stretch"):
                data = proj.load_project(ROOT, load_name)
                if data and data.get("result"):
                    st.session_state["ly_loaded_project"] = data
                    st.success(f"loaded {load_name!r} — its result is below")
                    st.json(data.get("recipe", {}))

    # ---------------- rating + style profile (features #49/#50) ---------------- #
    st.markdown("---")
    st.subheader("⭐ Rate it (builds your style profile)")
    s1, s2 = st.columns([3, 1])
    score = s1.slider("How hard does this hit?", 1, 5, 3, key="ly_rate")
    if s2.button("Log rating", key="ly_rate_btn", width="stretch"):
        rating.record_rating(ROOT, {
            "item": slug, "artist": result.artist, "mood": result.mood,
            "topic": result.topic, "genre": genre if genre != "(none)" else "",
            "score": score / 5.0,
        })
        st.success("rating logged")

    profile = rating.load_profile(ROOT)
    if profile.get("n_ratings"):
        st.caption(f"👤 {profile['n_ratings']} ratings · top artist: "
                   f"{rating.top_preference(profile, 'artists') or '—'} · top mood: "
                   f"{rating.top_preference(profile, 'moods') or '—'}")


# --------------------------------------------------------------------------- #
# ⚙️ Settings
# --------------------------------------------------------------------------- #
def page_settings() -> None:
    _page_header("⚙️", "Settings", "Appearance, pretrained templates, and file transfers.")
    from musictrain import transfer
    from musictrain.templates import MODELS, find_model_by_name

    cfg = load_cfg()
    cfg_path = ROOT / "configs" / "default.yaml"

    # ---------------- Appearance ---------------- #
    st.subheader("🎨 Appearance")
    ac1, ac2 = st.columns(2)
    # Same canonical keys as the sidebar controls (sidebar copy is hidden on
    # this page), so one click applies — no key pops, no drift between them.
    mode_before = st.session_state.get("mt_theme_mode", "dark")
    mode = ac1.segmented_control("Theme", ["dark", "light", "system"], key="mt_theme_mode")
    if mode != mode_before:
        st.rerun()

    accent_before = st.session_state.get("mt_accent_label") or "💙 Blue"
    accent = ac2.selectbox("🖌️ Accent", list(_ACCENTS.keys()), key="mt_accent_label")
    if accent != accent_before:
        st.rerun()

    font_before = st.session_state.get("mt_font_name") or "System"
    font = ac2.selectbox("✒️ Font", list(_FONTS.keys()), key="mt_font_name")
    if font != font_before:
        st.rerun()
    st.caption(f"Aa Bb Cc 123 — {font} preview")

    lang = st.selectbox(
        "🌐 Language", ["en", "es"],
        index=0 if st.session_state.get("mt_lang", "en") == "en" else 1,
        key="set_lang",
    )
    if lang != st.session_state.get("mt_lang", "en"):
        st.session_state["mt_lang"] = lang
        st.rerun()

    # ---------------- Pretrained templates ---------------- #
    st.subheader("🧠 Pretrained templates")
    model_names = [m.name for m in MODELS]
    cur_model = cfg.inference.model_name
    idx = next((i for i, m in enumerate(MODELS) if m.model_id == cur_model), 0)
    pick = st.selectbox("Default model", model_names, index=idx, key="set_def_model")
    picked = find_model_by_name(pick)
    if picked:
        cap = f"{picked.description} {picked.size}"
        if picked.melody_capable:
            cap += " · 🎵 melody conditioning"
        if picked.stereo:
            cap += " · 🔉 stereo"
        st.caption(cap)
        st.code(picked.model_id, language=None)

    c_set, c_cat = st.columns([1, 1])
    if c_set.button("Set as default", key="set_def_apply"):
        cfg.inference.model_name = picked.model_id
        cfg.settings.default_model = picked.model_id
        cfg.save(cfg_path)
        st.success(f"Default model → {picked.model_id}")
        st.rerun()
    with c_cat.popover("📚 Model catalog", width="stretch"):
        rows = [
            {"name": m.name, "id": m.model_id, "size": m.size,
             "melody": "✓" if m.melody_capable else "",
             "stereo": "✓" if m.stereo else ""}
            for m in MODELS
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.caption(
        f"{len(_TEMPLATES)} prompt templates are available on the 🎛️ Generate and 🪄 Prompt builder pages."
    )

    # ---------------- Files & transfer ---------------- #
    st.subheader("📂 Files & transfer")
    allow_ext = st.toggle(
        "🔓 Allow external paths", value=cfg.settings.allow_external_paths,
        key="set_allow_ext",
        help="Permit upload/download directories outside the project root.",
    )
    fu1, fu2 = st.columns(2)
    up_dir_raw = fu1.text_input("Upload directory", value=cfg.settings.upload_dir, key="set_up_dir")
    dn_dir_raw = fu2.text_input("Download directory", value=cfg.settings.download_dir, key="set_dn_dir")

    up_dir = dn_dir = None
    try:
        up_dir = transfer.resolve_dir(up_dir_raw, ROOT, "data/raw", allow_ext)
    except PermissionError as exc:
        st.error(str(exc))
    try:
        dn_dir = transfer.resolve_dir(dn_dir_raw, ROOT, "downloads", allow_ext)
    except PermissionError as exc:
        st.error(str(exc))

    if st.button("💾 Save settings", key="set_save"):
        cfg.settings.allow_external_paths = allow_ext
        cfg.settings.upload_dir = up_dir_raw
        cfg.settings.download_dir = dn_dir_raw
        cfg.settings.theme = st.session_state.get("mt_theme_mode", "dark")
        cfg.settings.accent = _ACCENTS.get(
            st.session_state.get("mt_accent_label"), cfg.settings.accent or "#5b8cff"
        )
        cfg.settings.default_model = cfg.inference.model_name
        cfg.settings.lang = st.session_state.get("mt_lang", "en")
        cfg.save(cfg_path)
        st.success(f"Saved → {cfg_path}")
        st.rerun()

    _dataset_export_import()

    st.markdown("#### ⬆ Upload")
    up_files = st.file_uploader(
        "Upload files to the upload directory",
        type=["wav", "mp3", "flac", "ogg", "m4a", "aiff", "aif",
              "csv", "json", "jsonl", "yaml", "pt", "bin", "safetensors"],
        accept_multiple_files=True, key="set_upload",
    )
    if up_files and st.button("Save uploads", key="set_upload_save"):
        if up_dir is None:
            st.error("Upload directory is invalid — fix the path/permission above.")
        else:
            transfer.ensure_dir(up_dir)
            n = 0
            for f in up_files:
                transfer.save_upload(f.getbuffer(), up_dir, f.name)
                n += 1
            st.success(f"Saved {n} file(s) → {up_dir}")
            st.rerun()

    st.markdown("#### ⬇ Download")
    artifacts = transfer.list_artifacts(ROOT, ["outputs", "checkpoints", "metadata"])
    if not artifacts:
        st.info("No artifacts yet — generate some audio or train a checkpoint first.")
    else:
        labels = [str(p.relative_to(ROOT)) for p in artifacts]
        sel = st.selectbox("Artifact", labels, key="set_dl_file")
        src = ROOT / sel
        c_dl, c_cp = st.columns(2)
        c_dl.download_button(
            "🌐 Download in browser", data=src.read_bytes(), file_name=src.name,
            mime="application/octet-stream", key="set_dl_browser", width="stretch",
        )
        if c_cp.button("💾 Copy to directory", key="set_dl_copy", width="stretch"):
            if dn_dir is None:
                st.error("Download directory is invalid — fix the path/permission above.")
            else:
                dest, overwrote = transfer.copy_download(src, dn_dir)
                st.success(f"Copied → {dest}" + (" (overwrote)" if overwrote else ""))


PAGES = {
    "📋 Inventory": page_inventory,
    "🔧 Normalize": page_normalize,
    "🏷️ Metadata": page_features,
    "✂️ Segment & Split": page_split,
    "🎛️ Generate": page_generate,
    "🎤 Lyrics": page_lyrics,
    "🪄 Prompt builder": page_promptbuilder,
    "📏 Check BPM": page_check,
    "🎬 Visualize": page_visualize,
    "🏷️ Labels": page_labels,
    "📊 Compare": page_compare,
    "🧹 Hygiene": page_hygiene,
    "🏆 Leaderboard": page_leaderboard,
    "📈 Training": page_training,
    "🔬 Analytics": page_analytics,
    "🧮 Metrics Lab": page_metricslab,
    "🎯 Eval": page_eval,
    "🎧 Listening": page_listening,
    "✂️ Annotate": page_annotate,
    "🧪 Campaign": page_campaign,
    "📦 Model Ops": page_modelops,
    "📡 Ops & Alerts": page_ops,
    "🪵 Logs": page_logs,
    "⚙️ Settings": page_settings,
}


# feature 51 — collapsible nav groups
_NAV_GROUPS = [
    ("📁 Data", ["📋 Inventory", "🔧 Normalize", "🏷️ Metadata", "✂️ Segment & Split"]),
    ("🎛️ Generate", ["🎛️ Generate", "🎤 Lyrics", "🪄 Prompt builder", "📏 Check BPM", "🎬 Visualize"]),
    ("🏷️ Curate", ["🏷️ Labels", "🧹 Hygiene", "🎧 Listening", "✂️ Annotate", "🧪 Campaign"]),
    ("📊 Evaluate", ["📊 Compare", "🏆 Leaderboard", "🧮 Metrics Lab", "🎯 Eval"]),
    ("🔬 Model", ["📈 Training", "🔬 Analytics", "📦 Model Ops"]),
    ("🪵 System", ["📡 Ops & Alerts", "🪵 Logs", "⚙️ Settings"]),
]


def _sidebar_minimap(cfg: Config) -> None:
    """Feature 60: tiny section x BPM coverage heatmap in the sidebar."""
    import altair as alt

    ev = ROOT / "metadata" / "eval_results.jsonl"
    if not ev.exists():
        return
    rows = []
    for line in ev.open():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        sec = r.get("section")
        bpm = r.get("bpm_target")
        if sec and bpm:
            rows.append({"section": sec, "bpm": float(bpm)})
    if len(rows) < 2:
        return
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(x=alt.X("bpm:Q", bin=alt.Bin(maxbins=12), title=None),
                y=alt.Y("section:N", title=None, sort=None),
                color=alt.Color("count()", scale=alt.Scale(scheme="blues"), legend=None))
        .properties(height=90)
    )
    st.altair_chart(chart, width="stretch")
    st.caption("🧭 prompt coverage (section × BPM)")


def _inspector() -> None:
    """Feature 58: compact inspector drawer of session state."""
    with st.expander("ℹ️ Inspector"):
        st.caption(f"page: `{st.session_state.get('nav', '—')}`")
        st.caption(f"theme: `{st.session_state.get('mt_theme_mode', 'dark')}`")
        st.caption(f"history: {len(st.session_state.get('mt_history', []))} pages")
        st.caption(f"pinned: {len(st.session_state.get('mt_pinned', []))}")


# feature 70 — i18n-ready string table (extend with more locales as needed)
_I18N = {
    "nav_title": {"en": "MusicTrain", "es": "MusicTrain"},
    "pinned": {"en": "Pinned", "es": "Fijadas"},
    "history": {"en": "History", "es": "Historial"},
    "focus": {"en": "Focus mode", "es": "Modo enfoque"},
    "theme": {"en": "Theme", "es": "Tema"},
}


def t(key: str) -> str:
    lang = st.session_state.get("mt_lang", "en")
    return _I18N.get(key, {}).get(lang, key)


def _pipeline_checklist(cfg: Config) -> None:
    """Feature 65: highlight completed pipeline steps."""
    steps = [
        ("inventory", (ROOT / "metadata" / "audio_inventory.json").exists()),
        ("features", (ROOT / "metadata" / "manifest.jsonl").exists()),
        ("labels", (ROOT / "metadata" / "labels.csv").exists()),
        ("split", (ROOT / "data" / "train").exists() or (ROOT / "data" / "val").exists()),
        ("eval", (ROOT / "metadata" / "eval_results.jsonl").exists()),
        ("leaderboard", (ROOT / "metadata" / "leaderboard.json").exists()),
    ]
    st.caption("  ".join(f"{'✅' if done else '⬜'} {name}" for name, done in steps))


def _print_button() -> None:
    """Feature 68: print / save-as-PDF via the browser."""
    import streamlit.components.v1 as components

    js = """
    <button onclick="window.print()"
      style="width:100%;border-radius:9px;border:1px solid rgba(255,255,255,.2);
      background:rgba(255,255,255,.06);color:#eef1fb;padding:6px 12px;cursor:pointer;font-size:.8rem">
      🖨 Print / Save as PDF
    </button>
    """
    components.html(js, height=34)


def _nav_ui() -> str:
    """Features 51/53/56: collapsible grouped nav, pinned pages, history dropdown."""
    first = list(PAGES.keys())[0]
    choice = st.session_state.get("nav", first)

    pinned = st.multiselect("⭐ Pinned", list(PAGES.keys()), key="mt_pinned")
    for page in pinned:
        if st.button(f"📌 {page}", key=f"pin_{page}", width="stretch"):
            st.session_state["nav"] = page
            st.rerun()

    history = st.session_state.get("mt_history", [])
    if history:
        back = st.selectbox("🕘 History", ["—"] + list(reversed(history)), key="hist_sel")
        if back != "—":
            st.session_state["nav"] = back
            st.session_state["hist_sel"] = "—"
            st.rerun()

    st.markdown("---")
    for group, pages in _NAV_GROUPS:
        with st.expander(group):
            for page in pages:
                label = f"▸ {page}" if page == choice else page
                if st.button(label, key=f"grp_{page}", width="stretch"):
                    st.session_state["nav"] = page
                    st.rerun()
    return choice


def main() -> None:
    # Optional sign-in gate (#17) — no-op unless MUSICTRAIN_PASSWORD/USERS/OAUTH set.
    from musictrain.auth import streamlit_gate

    if not streamlit_gate():
        st.stop()

    _session_resume()  # feature 49 — restore last page/theme/pins
    _command_palette()

    history = st.session_state.setdefault("mt_history", [])
    with st.sidebar:
        st.markdown("### 🎵 MusicTrain")
        st.caption(f"Project: `{ROOT.name}`")
        # sidebar theme controls share keys with the Settings page, so they
        # are hidden here to avoid duplicate widget keys (Settings has its own)
        if st.session_state.get("nav", list(PAGES.keys())[0]) != "⚙️ Settings":
            _toggle_theme()
        st.toggle("🎯 Focus mode", value=st.session_state.get("mt_focus", False), key="mt_focus_toggle")
        focus = st.session_state["mt_focus_toggle"]
        if not focus:
            _quicknav()
            _global_search()
            st.markdown("---")
            _sidebar_stats(load_cfg())
            _pipeline_checklist(load_cfg())
            _sidebar_minimap(load_cfg())
            _last_job_ui()  # feature 39 — replay the last job
            _print_button()
        _inspector()
        with st.expander("⌨️ Shortcuts"):
            st.caption("Ctrl/⌘+K — command palette")
            st.caption("g · l · c · h · i · n · m · b · t · a · v · e · s — jump to pages")
            st.caption("Esc — close palette")
        with st.expander("🛠 Crash report"):
            _crash_panel()  # feature 50
        st.markdown("---")
        choice = _nav_ui()

    # theme CSS is injected after the sidebar resolves the theme, so the
    # toggle applies on a single click instead of requiring a second rerun.
    st.markdown(_theme_css(), unsafe_allow_html=True)

    # breadcrumb history — feature 4
    if not history or history[-1] != choice:
        history.append(choice)
        st.session_state["mt_history"] = history[-12:]
    _crumbs(history)

    # feature 49 — persist the session (nav/theme/pins) for resume on reload
    try:
        sfile = ROOT / "metadata" / "session.json"
        sfile.parent.mkdir(parents=True, exist_ok=True)
        sfile.write_text(json.dumps({
            "nav": choice,
            "mt_theme_mode": st.session_state.get("mt_theme_mode"),
            "mt_accent_label": st.session_state.get("mt_accent_label"),
            "mt_font_name": st.session_state.get("mt_font_name"),
            "mt_focus": st.session_state.get("mt_focus", False),
            "mt_pinned": st.session_state.get("mt_pinned", []),
            "mt_lang": st.session_state.get("mt_lang", "en"),
        }, indent=2))
    except Exception:  # noqa: BLE001 - session persistence is best-effort
        pass

    PAGES[choice]()


main()
