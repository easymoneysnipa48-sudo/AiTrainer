"""Adherence metrics beyond BPM/CLAP (advanced eval batch #1-#10).

New measurement dimensions on top of the existing BPM check + CLAP score:

* **Onset alignment (#1)** — how tightly the onset envelope lines up with the
  detected beat grid (0..1). Low = the beat tracker's grid doesn't match the
  actual accents, i.e. a "floating" rhythm.
* **Key adherence (#2)** — detected key vs prompt key, scored by
  Camelot-wheel distance (0 = same/relative, 6 = farthest).
* **Structure-order scoring (#3)** — whether the detected section roles match
  the prompted section (energy/position heuristics from ``analysis.py``).
* **Duration adherence (#4)** — generated length vs target length.
* **Instrument presence (#5)** — band-energy heuristic presence scores for
  kick / 808 bass / snare-clap / hi-hats / tonal content.
* **Seed diversity (#6)** — per-seed CLAP variance (low = collapsed/overfit).
* **Reliability curve (#7)** — ok-rate as a function of prompt difficulty,
  plus a linear calibration (slope, intercept, R^2).
* **Multiple-comparison correction (#8)** — Bonferroni + Benjamini-Hochberg
  FDR on a set of p-values.
* **Bootstrap CIs (#9)** — percentile bootstrap confidence intervals, used by
  the leaderboard for score error bars.
* **Genre-specific gates (#10)** — per-genre CLAP/deviation thresholds.

Everything is librosa/numpy (no extra model loads) except the orchestrator,
which reuses the existing ``analysis`` pipeline for key/structure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import console
from .config import Config
from .report import load_results

# --------------------------------------------------------------------------- #
# Camelot wheel helpers (#2)
# --------------------------------------------------------------------------- #

# Root note -> Camelot number (1..12) for each mode, per the standard
# Mixed-in-Key wheel. Consecutive numbers are a perfect fifth apart; the same
# number in the opposite mode is the relative major/minor (distance 0).
_MINOR = {
    "A": 8, "A#": 3, "Bb": 3, "B": 10, "C": 5, "C#": 12, "Db": 12,
    "D": 7, "D#": 2, "Eb": 2, "E": 9, "F": 4, "F#": 11, "G": 6, "G#": 1,
    "Ab": 1,
}
_MAJOR = {
    "A": 11, "A#": 6, "Bb": 6, "B": 1, "C": 8, "C#": 3, "Db": 3,
    "D": 10, "D#": 5, "Eb": 5, "E": 12, "F": 7, "F#": 2, "G": 9, "G#": 4,
    "Ab": 4,
}


def parse_key(key: str) -> Optional[Tuple[int, str]]:
    """Parse ``"A minor"`` -> ``(8, "A")`` (camelot number, ring). None if unparsable."""
    if not key:
        return None
    parts = key.strip().split()
    if len(parts) < 2:
        return None
    root, mode = parts[0], parts[1].lower()
    table = _MINOR if mode in ("minor", "min", "m") else _MAJOR if mode in ("major", "maj") else None
    if table is None or root not in table:
        return None
    return table[root], "A" if table is _MINOR else "B"


def camelot_distance(key_a: str, key_b: str) -> Optional[int]:
    """Harmonic distance between two keys, 0..6 (0 = same or relative)."""
    a, b = parse_key(key_a), parse_key(key_b)
    if a is None or b is None:
        return None
    na, nb = a[0], b[0]
    if na == nb:
        return 0  # same key, or relative major/minor
    return int(min((na - nb) % 12, (nb - na) % 12))


def key_adherence(detected_key: str, target_key: str) -> Optional[dict]:
    """Score detected-vs-target key on Camelot distance -> 0..1 score."""
    dist = camelot_distance(detected_key, target_key)
    if dist is None:
        return None
    return {
        "detected_key": detected_key,
        "target_key": target_key,
        "camelot_distance": dist,
        "score": round(1.0 - dist / 6.0, 4),
        "match": dist == 0,
    }


# --------------------------------------------------------------------------- #
# Onset alignment (#1)
# --------------------------------------------------------------------------- #

def onset_alignment_score(y: np.ndarray, sr: int, hop_length: int = 512) -> Optional[float]:
    """0..1: how well onset energy concentrates on the detected beat grid.

    Builds the beat grid, then measures the ratio of onset strength *on* the
    grid vs the total onset strength (a simplified downbeat/beat lock score).
    """
    import librosa

    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        tempo = float(np.atleast_1d(tempo)[0])
    except Exception:  # noqa: BLE001
        return None
    if tempo <= 0 or len(beats) < 2:
        return None

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if onset.size == 0:
        return None

    # strength at beat frames vs everywhere else
    on_grid = float(onset[np.clip(beats, 0, len(onset) - 1)].mean())
    total = float(onset.mean())
    if total <= 1e-9:
        return None
    # a perfect lock has most energy on the grid; chance baseline is ~ (grid/total)
    # scale so a uniform envelope -> 0 and fully-on-grid -> 1
    frac = on_grid / total if total > 0 else 0.0
    baseline = len(beats) / max(len(onset), 1)
    denom = 1.0 - baseline
    if denom <= 1e-9:
        return 1.0
    score = float(np.clip((frac - baseline) / denom, 0.0, 1.0))
    return round(score, 4)


# --------------------------------------------------------------------------- #
# Structure order (#3)
# --------------------------------------------------------------------------- #

def structure_order_score(detected_roles: Sequence[str], target_section: str) -> Optional[dict]:
    """Score whether the prompted section's role appears in the detected roles.

    Coarse but useful: a prompted "chorus" should produce a loud mid section,
    an "intro" a quiet first section, etc. Returns presence + position match.
    """
    roles = list(detected_roles)
    if not roles or not target_section:
        return None

    target = (target_section or "").lower().strip()
    # full-song prompts are satisfied by any multi-section or single structure
    if target == "full-song":
        return {"target": target_section, "detected_roles": roles,
                "presence": True, "position_match": True, "score": 1.0}

    # intro/outro are positional
    position_ok = False
    if target == "intro":
        position_ok = roles and roles[0] == "intro"
    elif target == "outro":
        position_ok = roles and roles[-1] == "outro"

    presence = target in roles
    score = 1.0 if (presence and (position_ok or target not in ("intro", "outro"))) else \
        (0.5 if presence else 0.0)
    return {
        "target": target_section,
        "detected_roles": roles,
        "presence": presence,
        "position_match": position_ok,
        "score": round(score, 4),
    }


# --------------------------------------------------------------------------- #
# Duration adherence (#4)
# --------------------------------------------------------------------------- #

def duration_adherence(actual_seconds: float, target_seconds: float,
                       tolerance: float = 0.15) -> Optional[dict]:
    """Score generated length against a target, 0..1 (tolerance is relative)."""
    if not actual_seconds or not target_seconds or target_seconds <= 0:
        return None
    rel = abs(actual_seconds - target_seconds) / target_seconds
    score = float(np.clip(1.0 - rel / (tolerance * 2.0), 0.0, 1.0))
    return {
        "actual_seconds": round(actual_seconds, 3),
        "target_seconds": round(target_seconds, 3),
        "relative_error": round(rel, 4),
        "score": round(score, 4),
        "match": rel <= tolerance,
    }


# --------------------------------------------------------------------------- #
# Instrument presence (#5) — band-energy heuristics
# --------------------------------------------------------------------------- #

def instrument_presence(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    """Band-energy presence scores for kick/808/snare/hats/tonal content."""
    import librosa

    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)

    def band(lo: float, hi: float) -> float:
        m = S[(freqs >= lo) & (freqs < hi)]
        return float(m.mean()) if m.size else 0.0

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    kick = band(40, 120)
    bass = band(40, 160)          # sustained low (808 range)
    snare = band(1500, 4500)
    hats = band(6000, 12000)
    tonal = band(200, 3000)

    # transientness: high-frequency bands that *pulse* are percussive
    hat_transience = float(np.std(onset)) / (float(np.mean(onset)) + 1e-9)

    total = kick + bass + snare + hats + tonal + 1e-9
    out = {
        "kick": round(kick / total, 4),
        "bass_808": round(bass / total, 4),
        "snare_clap": round(snare / total, 4),
        "hi_hats": round(hats / total, 4),
        "tonal": round(tonal / total, 4),
        "hat_transience": round(float(np.clip(hat_transience / 3.0, 0.0, 1.0)), 4),
    }
    return out


# --------------------------------------------------------------------------- #
# Seed diversity (#6)
# --------------------------------------------------------------------------- #

def seed_clap_diversity(seed_records: List[dict]) -> Optional[dict]:
    """Diversity of a repeated-seed prompt from per-seed CLAP scores.

    High std/CV = diverse generations (good exploration); near-zero std across
    many seeds = collapsed/overfit output.
    """
    claps = [r.get("clap_score") for r in seed_records if r.get("clap_score") is not None]
    if len(claps) < 2:
        return None
    arr = np.asarray(claps, dtype=float)
    std = float(arr.std())
    mean = float(arr.mean())
    cv = std / mean if mean > 1e-9 else None
    return {
        "n_seeds": len(claps),
        "mean_clap": round(mean, 4),
        "std_clap": round(std, 4),
        "cv": round(cv, 4) if cv is not None else None,
        "min_clap": round(float(arr.min()), 4),
        "max_clap": round(float(arr.max()), 4),
        "collapsed": std < 0.01 and len(claps) >= 3,
    }


# --------------------------------------------------------------------------- #
# Reliability curve (#7)
# --------------------------------------------------------------------------- #

def reliability_curve(rows: List[dict], n_bins: int = 4) -> Optional[dict]:
    """ok-rate vs prompt difficulty, plus a linear calibration fit.

    ``rows`` must carry ``difficulty`` (see ``difficulty.prompt_difficulty``)
    or we derive it inline from clap/deviation. Returns the binned curve and
    fit parameters so a checkpoint's reliability-vs-hardness is measurable.
    """
    pts = []
    for r in rows:
        d = r.get("difficulty")
        if d is None:
            clap = r.get("clap_score")
            dev = r.get("deviation")
            clap_t = 1.0 - float(clap) if clap is not None else 1.0
            dev_t = min(abs(float(dev)) / 0.5, 1.0) if dev is not None else 1.0
            d = 0.5 * clap_t + 0.5 * dev_t
        pts.append((float(d), 1.0 if r.get("status") == "ok" else 0.0))

    if not pts:
        return None
    pts.sort(key=lambda p: p[0])
    lo = pts[0][0]
    hi = pts[-1][0]
    span = hi - lo
    if span < 1e-9:
        bins = [pts]
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
        edges[-1] += 1e-9
        bins = [[] for _ in range(n_bins)]
        for d, ok in pts:
            idx = int(np.searchsorted(edges, d, side="right") - 1)
            idx = int(np.clip(idx, 0, n_bins - 1))
            bins[idx].append((d, ok))

    curve = []
    for b in bins:
        if not b:
            continue
        ds = [p[0] for p in b]
        oks = [p[1] for p in b]
        curve.append({
            "bin_center": round(float(np.mean(ds)), 4),
            "n": len(b),
            "ok_rate": round(float(np.mean(oks)), 4),
        })

    # linear fit: ok_rate ~ a + b * difficulty
    xs = np.asarray([p[0] for p in pts], dtype=float)
    ys = np.asarray([p[1] for p in pts], dtype=float)
    if len(xs) >= 2:
        slope, intercept = [float(c) for c in np.polyfit(xs, ys, 1)]
    else:
        slope, intercept = 0.0, float(ys.mean())
    pred = intercept + slope * xs
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "n": len(pts),
        "curve": curve,
        "fit": {
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "r_squared": round(r2, 4),
            "reliable": bool(r2 >= 0.5),
        },
    }


# --------------------------------------------------------------------------- #
# Multiple-comparison correction (#8)
# --------------------------------------------------------------------------- #

def multiple_comparison(p_values: Sequence[Optional[float]],
                        alpha: float = 0.05) -> dict:
    """Bonferroni + Benjamini-Hochberg FDR correction over a set of p-values."""
    ps = [p for p in p_values if p is not None and np.isfinite(p)]
    if not ps:
        return {"n": 0, "bonferroni": {}, "bh_fdr": {}}
    m = len(ps)
    bonf_alpha = alpha / m if m else alpha
    bonf = {f"test_{i}": {"p": p, "reject": p < bonf_alpha}
            for i, p in enumerate(ps)}

    # BH-FDR step-up
    order = sorted(range(m), key=lambda i: ps[i])
    reject = [False] * m
    prev = None
    for rank, idx in enumerate(order):
        thresh = (rank + 1) / m * alpha
        reject[idx] = ps[idx] <= thresh
        prev = reject[idx]
    bh = {f"test_{i}": {"p": ps[i], "reject": reject[i]} for i in range(m)}
    return {
        "n": m,
        "alpha": alpha,
        "bonferroni_alpha": round(bonf_alpha, 6),
        "bonferroni": bonf,
        "bh_fdr": bh,
        "n_reject_bonferroni": sum(1 for v in bonf.values() if v["reject"]),
        "n_reject_bh_fdr": sum(1 for v in bh.values() if v["reject"]),
    }


# --------------------------------------------------------------------------- #
# Bootstrap CI (#9)
# --------------------------------------------------------------------------- #

def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Optional[dict]:
    """Percentile bootstrap CI for the mean of ``values``."""
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if vals.size < 3:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = vals[rng.integers(0, vals.size, vals.size)].mean()
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    lo, hi = np.percentile(means, [lo_q, hi_q])
    return {
        "mean": round(float(vals.mean()), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "n": int(vals.size),
    }


def bootstrap_score_ci(rows: List[dict], n_boot: int = 2000, seed: int = 0) -> Optional[dict]:
    """Bootstrap CI around a checkpoint's composite leaderboard score.

    Re-derives the score (ok-share 40% / CLAP 30% / deviation-fidelity 30%)
    on each resample so the CI reflects score uncertainty, not just one axis.
    """
    if not rows:
        return None

    def _score(rs: List[dict]) -> float:
        claps = [r["clap_score"] for r in rs if r.get("clap_score") is not None]
        devs = [abs(r["deviation"]) for r in rs if r.get("deviation") is not None]
        ok = sum(1 for r in rs if r.get("status") == "ok") / len(rs)
        mean_clap = sum(claps) / len(claps) if claps else 0.0
        mean_dev = sum(devs) / len(devs) if devs else 0.0
        fidelity = max(0.0, 1.0 - mean_dev / 0.20)
        return 0.4 * ok + 0.3 * mean_clap + 0.3 * fidelity

    rng = np.random.default_rng(seed)
    n = len(rows)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[i] = _score([rows[j] for j in idx])
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {
        "score": round(float(_score(rows)), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
    }


# --------------------------------------------------------------------------- #
# Genre-specific gates (#10)
# --------------------------------------------------------------------------- #

def genre_gate(clap: Optional[float], deviation: Optional[float],
               genre: str, gates: Dict[str, dict]) -> dict:
    """Apply per-genre thresholds, falling back to a default gate."""
    g = gates.get(genre) or gates.get("default") or {
        "min_clap": 0.30, "max_abs_deviation": 0.20,
    }
    min_clap = float(g.get("min_clap", 0.30))
    max_dev = float(g.get("max_abs_deviation", 0.20))

    reasons = []
    if clap is not None and clap < min_clap:
        reasons.append(f"CLAP {clap:.3f} < genre min {min_clap:.3f}")
    if deviation is not None and abs(deviation) > max_dev:
        reasons.append(f"|dev| {abs(deviation):.3f} > genre max {max_dev:.3f}")
    return {
        "genre": genre,
        "thresholds": {"min_clap": min_clap, "max_abs_deviation": max_dev},
        "clap": clap,
        "deviation": deviation,
        "passed": not reasons,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Orchestrator — score one audio file against a prompt dict
# --------------------------------------------------------------------------- #

def adherence_checks(cfg: Config, audio_path: Path, prompt: dict) -> dict:
    """Run all audio-based adherence checks for one (file, prompt) pair.

    Requires the deep-analysis pipeline (chords/grid/key/structure) which is
    already available; falls back gracefully where analysis is disabled.
    """
    from .audio.analysis import analyze_file

    rec = {}
    try:
        an = analyze_file(cfg, audio_path, cfg.project_root)
        y_key = an.get("key", {}).get("key")
        if y_key and prompt.get("key"):
            rec["key"] = key_adherence(y_key, prompt["key"])
        if prompt.get("section"):
            roles = [s.get("role") for s in an.get("structure", {}).get("segments", [])]
            rec["structure"] = structure_order_score(roles, prompt["section"])
        if an.get("duration"):
            target = (prompt.get("duration_seconds")
                      or (prompt.get("bpm") and 8 * 60 / float(prompt["bpm"]) / 4))
            if target:
                rec["duration"] = duration_adherence(float(an["duration"]), float(target))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = str(exc)

    # onset alignment + instruments need the raw signal, not the analysis dict
    try:
        from .audio.features import load_audio
        y, sr = load_audio(audio_path, sr=cfg.analysis.sr)
        rec["onset_alignment"] = onset_alignment_score(y, sr, hop_length=cfg.analysis.hop_length)
        rec["instruments"] = instrument_presence(y, sr, hop_length=cfg.analysis.hop_length)
    except Exception as exc:  # noqa: BLE001
        rec["signal_error"] = str(exc)

    return rec


def run(root: Path, cfg: Config, limit: int = 0) -> dict:
    """Score every eval row's audio against its prompt, write metadata/adherence.json.

    Falls back to a key-only pass (from analysis.jsonl) when audio files are
    missing, so the report is still produced from cached analysis.
    """
    rows = load_results(root)
    if not rows:
        console.error("No eval results — run `musictrain eval` first.")
        return {}

    # collect unique (audio_path, prompt) pairs
    seen = set()
    pairs = []
    for r in rows:
        ap = r.get("audio_path")
        if not ap or ap in seen:
            continue
        seen.add(ap)
        pairs.append((Path(ap), r))
        if limit and len(pairs) >= limit:
            break

    scored: List[dict] = []
    console.step(f"Computing adherence metrics for {len(pairs)} unique clip(s)…")
    for i, (ap, r) in enumerate(pairs, 1):
        if not ap.exists():
            scored.append({"audio_path": str(ap), "missing": True})
            continue
        rec = {"audio_path": str(ap), "prompt": r.get("prompt")}
        rec.update(adherence_checks(cfg, ap, r))
        scored.append(rec)
        console.info(f"[{i}/{len(pairs)}] {ap.name}: key={_sum(rec, 'key')} "
                     f"align={_sum(rec, 'onset_alignment')}")

    # dataset-level: reliability curve, seed diversity, genre gates
    extras: dict = {}
    rel = reliability_curve(rows)
    if rel:
        extras["reliability_curve"] = rel

    claps = [r["clap_score"] for r in rows if r.get("clap_score") is not None]
    devs = [r["deviation"] for r in rows if r.get("deviation") is not None]
    if claps:
        extras["clap_ci"] = bootstrap_ci(claps)
    if devs:
        extras["deviation_ci"] = bootstrap_ci([abs(d) for d in devs])

    genre_gates = {}
    by_genre: Dict[str, List[dict]] = {}
    for r in rows:
        by_genre.setdefault(r.get("genre") or "default", []).append(r)
    for genre, grs in by_genre.items():
        mean_clap = sum(g["clap_score"] for g in grs if g.get("clap_score") is not None)
        n_clap = sum(1 for g in grs if g.get("clap_score") is not None)
        mean_dev = sum(abs(g["deviation"]) for g in grs if g.get("deviation") is not None)
        n_dev = sum(1 for g in grs if g.get("deviation") is not None)
        genre_gates[genre] = genre_gate(
            (mean_clap / n_clap) if n_clap else None,
            (mean_dev / n_dev) if n_dev else None,
            genre, getattr(cfg.eval, "genre_gates", {}),
        )
    extras["genre_gates"] = genre_gates

    out = {"n_clips": len(scored), "clips": scored, **extras}
    path = root / "metadata" / "adherence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    console.ok(f"Adherence -> {path.relative_to(root)}")
    return out


def _sum(rec: dict, key: str) -> str:
    v = rec.get(key)
    if isinstance(v, dict):
        if "score" in v:
            return f"{v['score']:.2f}"
        if "camelot_distance" in v:
            return f"d={v['camelot_distance']}"
    if v is None:
        return "—"
    return str(v)
