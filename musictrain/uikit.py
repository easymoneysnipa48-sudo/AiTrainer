"""Reusable UI primitives for the dashboard (batch: more visuals + skeletons).

Dependency-light building blocks rendered as HTML/CSS (with a graceful
Streamlit fallback) so pages stay tidy and consistent:

* ``animated_skeleton``  — shimmer loading placeholders (animated).
* ``progress_skeleton``  — skeleton progress rail with animated fill.
* ``preview_box``        — audio preview card (player + meta chips).
* ``review_box``         — human rating + note capture for a clip.
* ``gauge``              — SVG radial gauge (0..1).
* ``sparkline``          — SVG inline sparkline.
* ``metric_tile``        — HTML metric tile with delta.

Every component is safe to call headless (AppTest): it degrades to plain
markdown/text when ``components.html`` is unavailable.
"""
from __future__ import annotations

import html
from typing import Optional, Sequence

from .logging import get_logger

log = get_logger("uikit")


def _emit(js: str, height: int = 0) -> None:
    try:
        import streamlit.components.v1 as components

        components.html(js, height=height)
    except Exception as exc:  # noqa: BLE001
        log.warning("components.html unavailable: %s", exc)


def _esc(s: str) -> str:
    return html.escape(str(s))


# --------------------------------------------------------------------------- #
# skeletons
# --------------------------------------------------------------------------- #
_SHIMMER_CSS = """
<style>
  .mt-skel { position: relative; overflow: hidden; border-radius: 10px;
    background: rgba(255,255,255,0.06); margin: 6px 0; }
  .mt-skel::after { content: ""; position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    animation: mt-shimmer 1.4s infinite; transform: translateX(-100%); }
  @keyframes mt-shimmer { 100% { transform: translateX(100%); } }
  .mt-skel .bar { display: block; border-radius: 6px; background: rgba(255,255,255,0.08);
    margin: 8px 10px; }
</style>
"""


def animated_skeleton(n: int = 3, height: int = 60) -> None:
    """Render `n` shimmer placeholder blocks (animated loading skeleton)."""
    bars = "".join(
        f'<div class="mt-skel" style="height:{max(20, height - 16)}px">'
        f'<span class="bar" style="width:{80 - (i % 3) * 18}%;height:{min(16, height // 3)}px"></span>'
        f"</div>"
        for i in range(n)
    )
    _emit(_SHIMMER_CSS + f'<div>{bars}</div>', height=n * height)


def progress_skeleton(label: str = "loading", frac: float = 0.0, steps: int = 5) -> None:
    """Animated skeleton progress rail: blocky shimmer bars that fill to `frac`."""
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * steps))
    cells = []
    for i in range(steps):
        cls = "on" if i < filled else "off"
        cells.append(f'<span class="mt-ps {cls}"></span>')
    html_block = f"""
    <style>
      .mt-ps-rail {{ display:flex; gap:4px; margin:4px 0 10px; }}
      .mt-ps {{ flex:1; height:10px; border-radius:5px; }}
      .mt-ps.on {{ background: linear-gradient(90deg,#5b8cff,#7c5cff); }}
      .mt-ps.off {{ background: rgba(255,255,255,0.08); position:relative; overflow:hidden; }}
      .mt-ps.off::after {{ content:""; position:absolute; inset:0;
        background: linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent);
        animation: mt-shimmer 1.2s infinite; transform:translateX(-100%); }}
      @keyframes mt-shimmer {{ 100% {{ transform:translateX(100%); }} }}
    </style>
    <div style="font-size:.72rem;color:#9aa3c0">{_esc(label)} — {int(frac * 100)}%</div>
    <div class="mt-ps-rail">{''.join(cells)}</div>
    """
    _emit(html_block, height=40)


# --------------------------------------------------------------------------- #
# preview / review boxes
# --------------------------------------------------------------------------- #
def preview_box(path: str, label: str = "", meta: Optional[dict] = None,
                height: int = 210) -> None:
    """Audio preview card: native player + meta chips (BPM/CLAP/deviation)."""
    import streamlit as st

    with st.container(border=True):
        if label:
            st.markdown(f"**🎧 {_esc(label)}**")
        chips = []
        if meta:
            for k, v in meta.items():
                if v is not None:
                    chips.append(f"<span class='mt-chip'>{_esc(k)}: <b>{_esc(v)}</b></span>")
        if chips:
            st.markdown("<div>" + "".join(chips) + "</div>", unsafe_allow_html=True)
        try:
            st.audio(path)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"(audio unavailable: {exc})")


def review_box(path: str, key: str, label: str = "", default: int = 3) -> None:
    """Human rating + note capture for a clip (seeds human_ratings.jsonl)."""
    import streamlit as st

    with st.container(border=True):
        st.markdown(f"**🧪 Review** {f'— {_esc(label)}' if label else ''}")
        rating = st.slider("Rating", 1, 5, default, key=f"{key}_rating")
        note = st.text_input("Note", key=f"{key}_note")
        if st.button("Save rating", key=f"{key}_save"):
            st.session_state.setdefault("mt_ratings", []).append(
                {"path": path, "rating": rating, "note": note}
            )
            st.toast(f"Saved {rating}★ for {path.split('/')[-1]}")


# --------------------------------------------------------------------------- #
# gauges / sparklines / tiles
# --------------------------------------------------------------------------- #
def gauge(value: float, label: str = "", maximum: float = 1.0, key: str = "") -> None:
    """SVG radial gauge (0..maximum)."""
    value = max(0.0, min(maximum, value))
    pct = value / maximum if maximum else 0.0
    r = 40
    c = 2 * 3.14159 * r
    filled = c * pct
    color = "#7ee2a8" if pct >= 0.5 else "#ffd479" if pct >= 0.25 else "#ff7b7b"
    svg = f"""
    <div style="text-align:center">
      <svg width="110" height="90" viewBox="0 0 110 90">
        <circle cx="55" cy="45" r="{r}" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="10"/>
        <circle cx="55" cy="45" r="{r}" fill="none" stroke="{color}" stroke-width="10"
          stroke-linecap="round" stroke-dasharray="{c}" stroke-dashoffset="{c - filled}"
          transform="rotate(-90 55 45)"/>
        <text x="55" y="50" text-anchor="middle" fill="#eef1fb" font-size="16" font-weight="700">
          {int(round(pct * 100))}%</text>
      </svg>
      <div style="font-size:.72rem;color:#9aa3c0">{_esc(label)}</div>
    </div>
    """
    _emit(svg, height=110)


def sparkline(values: Sequence[float], label: str = "", width: int = 220,
              height: int = 48) -> None:
    """Inline SVG sparkline for a numeric series."""
    vals = [float(v) for v in values]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = 4 + i * (width - 8) / max(1, n - 1)
        y = height - 6 - (v - lo) / span * (height - 12)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    svg = f"""
    <div>
      <svg width="{width}" height="{height}" style="background:rgba(255,255,255,.03);border-radius:8px">
        <polyline points="{poly}" fill="none" stroke="#6ea8ff" stroke-width="2"
          stroke-linejoin="round" stroke-linecap="round"/>
      </svg>
      <div style="font-size:.7rem;color:#9aa3c0">{_esc(label)}</div>
    </div>
    """
    _emit(svg, height=height + 18)


def metric_tile(label: str, value, delta: str = "") -> None:
    """HTML metric tile (theme-styled, matches st.metric look)."""
    d = f'<div class="mt-tile-d">{_esc(delta)}</div>' if delta else ""
    _emit(
        f'<div class="mt-tile"><div class="mt-tile-l">{_esc(label)}</div>'
        f'<div class="mt-tile-v">{_esc(value)}</div>{d}</div>',
        height=86,
    )
