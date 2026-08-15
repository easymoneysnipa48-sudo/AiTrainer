"""Extended evaluation & measurement (gap #8-#13).

Pure-logic functions (testable without audio) plus a dispatcher that reuses the
existing ``metrics``/``similarity``/``significance`` plumbing:

* ``fad_gate``        — gate a FAD score against a threshold (wire into CI).
* ``mos_proxy``       — fold CLAP + quality heuristics into one perceptual score.
* ``listening_ab``    — binomial sign test for human A/B ratings (with p-value).
* ``embedding_leakage``— nearest-neighbor overlap between train/val embeddings.
* ``robustness_prompts`` — typo/phonetic perturbations of existing prompts.
* ``per_genre_gate``  — enforce per-genre CLAP/deviation thresholds.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from . import console
from .config import Config
from .logging import get_logger

log = get_logger("evalx")


# --------------------------------------------------------------------------- #
# Pure logic
# --------------------------------------------------------------------------- #

def mos_proxy(clap: Optional[float], clipping: float = 0.0,
              silence: float = 0.0, snr_db: Optional[float] = None) -> Optional[float]:
    """Fold CLAP + audio-quality heuristics into a single 0..1 perceptual proxy.

    Weights: CLAP 0.6, artifacts 0.25, silence 0.15. All inputs are clipped to
    sane ranges so a single bad value can't zero the score. Returns None when
    CLAP is missing (the proxy is only meaningful with an adherence signal).
    """
    if clap is None:
        return None
    clap = max(0.0, min(1.0, clap))
    artifact = 1.0 - max(0.0, min(1.0, clipping * 10.0))          # 1 = clean
    silence_penalty = 1.0 - max(0.0, min(1.0, silence * 2.0))     # 1 = not silent
    snr = 1.0
    if snr_db is not None:
        snr = max(0.0, min(1.0, (snr_db + 20.0) / 40.0))          # -20..20 dB -> 0..1
    score = 0.6 * clap + 0.25 * artifact * snr + 0.15 * silence_penalty
    return round(max(0.0, min(1.0, score)), 4)


def listening_ab(a_wins: int, b_wins: int, ties: int = 0) -> dict:
    """Two-sided binomial sign test for human A/B listening ratings.

    Null hypothesis: A and B are equally likely to win (p = 0.5), ties excluded.
    Returns win-rate plus the exact binomial p-value (and the normal approx for
    large n).
    """
    n = a_wins + b_wins
    if n == 0:
        return {"a_wins": a_wins, "b_wins": b_wins, "ties": ties, "n": 0,
                "win_rate_a": None, "p_value": None, "verdict": "no_data"}
    win_rate = a_wins / n
    # exact two-sided binomial p-value
    p = 0.0
    import statistics
    for k in range(0, n + 1):
        pk = math.comb(n, k) * (0.5 ** n)
        if abs(k - n / 2) >= abs(a_wins - n / 2):
            p += pk
    p = min(1.0, p)
    verdict = "a_better" if p < 0.05 and win_rate > 0.5 else \
              "b_better" if p < 0.05 and win_rate < 0.5 else "no_significant_difference"
    return {"a_wins": a_wins, "b_wins": b_wins, "ties": ties, "n": n,
            "win_rate_a": round(win_rate, 4), "p_value": round(p, 6),
            "verdict": verdict}


def embedding_leakage(ref_embs: Sequence[Sequence[float]],
                      gen_embs: Sequence[Sequence[float]],
                      threshold: float = 0.95) -> dict:
    """Detect train/val leakage via nearest-neighbor embedding overlap.

    For each generated embedding, find its nearest reference neighbor; a hit
    within ``threshold`` (cosine similarity) flags a potential duplicate/leak.
    """
    ref = [tuple(r) for r in ref_embs]
    gen = [tuple(g) for g in gen_embs]
    if not ref or not gen:
        return {"n_ref": len(ref), "n_gen": len(gen), "n_leaks": 0, "leak_rate": 0.0}
    leaks = 0
    nearest: List[float] = []
    for g in gen:
        best = max(_cos(g, r) for r in ref)
        nearest.append(best)
        if best >= threshold:
            leaks += 1
    return {
        "n_ref": len(ref), "n_gen": len(gen), "n_leaks": leaks,
        "leak_rate": round(leaks / len(gen), 4),
        "mean_nn_sim": round(sum(nearest) / len(nearest), 4),
        "max_nn_sim": round(max(nearest), 4),
    }


def _cos(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def robustness_prompts(prompts: Sequence[str], n: int = 0,
                       seed: int = 0) -> List[dict]:
    """Perturb prompts (typos, phonetic swaps) to test robustness to phrasing."""
    _TYPO = {"chorus": "chorous", "trap": "trp", "synth": "synthh",
             "melodic": "melodicc", "808": "8o8", "piano": "piano"}
    rng = random.Random(seed)
    pool = list(prompts)
    if n and n < len(pool):
        pool = sorted(pool, key=lambda _: rng.random())[:n]
    out: List[dict] = []
    for p in pool:
        perturbed = p
        for word, typo in _TYPO.items():
            if word in perturbed and rng.random() < 0.5:
                perturbed = perturbed.replace(word, typo, 1)
        out.append({"original": p, "perturbed": perturbed})
    return out


def per_genre_gate(rows: Sequence[dict], gates: Dict[str, dict]) -> dict:
    """Enforce per-genre CLAP/deviation thresholds across eval rows."""
    by_genre: Dict[str, List[dict]] = {}
    for r in rows:
        g = (r.get("genre") or "default").strip() or "default"
        by_genre.setdefault(g, []).append(r)
    report: Dict[str, dict] = {}
    all_pass = True
    for genre, group in by_genre.items():
        gate = gates.get(genre) or gates.get("default") or {}
        min_clap = gate.get("min_clap", 0.0)
        max_dev = gate.get("max_abs_deviation", 0.2)
        claps = [r["clap_score"] for r in group if r.get("clap_score") is not None]
        devs = [abs(r["deviation"]) for r in group if r.get("deviation") is not None]
        mean_clap = sum(claps) / len(claps) if claps else 0.0
        mean_dev = sum(devs) / len(devs) if devs else float("inf")
        ok = mean_clap >= min_clap and mean_dev <= max_dev
        all_pass = all_pass and ok
        report[genre] = {
            "n": len(group), "mean_clap": round(mean_clap, 4),
            "mean_abs_deviation": round(mean_dev, 4),
            "min_clap": min_clap, "max_dev": max_dev, "passed": ok,
        }
    return {"passed": all_pass, "genres": report}


def fad_gate(fad: Optional[float], threshold: float = 10.0) -> dict:
    """Gate a FAD score (lower = better) against a threshold."""
    if fad is None:
        return {"passed": None, "fad": None, "threshold": threshold,
                "reason": "FAD unavailable"}
    return {"passed": fad <= threshold, "fad": round(fad, 4),
            "threshold": threshold,
            "reason": "ok" if fad <= threshold else "FAD above threshold"}


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def run(root, cfg: Config, task: str, **kwargs) -> dict:
    """Dispatch for the `musictrain evalx --task ...` command."""
    if task == "mos":
        out = mos_proxy(kwargs.get("clap"), kwargs.get("clipping", 0.0),
                        kwargs.get("silence", 0.0), kwargs.get("snr_db"))
        console.ok(f"MOS proxy: {out}" if out is not None else "MOS proxy needs a CLAP score")
        return {"task": task, "mos_proxy": out}

    if task == "ab":
        out = listening_ab(kwargs.get("a_wins", 0), kwargs.get("b_wins", 0),
                           kwargs.get("ties", 0))
        console.ok(f"A/B: {out['verdict']} (p={out['p_value']}, n={out['n']})")
        return {"task": task, **out}

    if task == "leakage":
        ref = kwargs.get("ref_embs") or []
        gen = kwargs.get("gen_embs") or []
        ref_dir = kwargs.get("ref_dir")
        gen_dir = kwargs.get("gen_dir")
        if (not ref or not gen) and ref_dir and gen_dir:
            from .embeddings import embed_dir

            ref_map = embed_dir(root, cfg, which=str(ref_dir), limit=kwargs.get("limit", 0))
            gen_map = embed_dir(root, cfg, which=str(gen_dir), limit=kwargs.get("limit", 0))
            ref = list(ref_map.values())
            gen = list(gen_map.values())
        threshold = kwargs.get("threshold")
        if threshold is None:
            threshold = 0.95
        out = embedding_leakage(ref, gen, threshold)
        console.ok(f"Leakage: {out['n_leaks']}/{out['n_gen']} ({out['leak_rate']:.0%})")
        return {"task": task, **out}

    if task == "robust":
        prompts = kwargs.get("prompts") or []
        if not prompts:
            from .evalset import load

            prompts = [p.get("description", "") for p in load(root) if p.get("description")]
        out = robustness_prompts(prompts, kwargs.get("n", 0), kwargs.get("seed", 0))
        console.ok(f"Robustness prompts: {len(out)} perturbation(s)")
        return {"task": task, "n": len(out), "prompts": out}

    if task == "fad-gate":
        from .metrics import compute

        ref = kwargs.get("ref_dir")
        gen = kwargs.get("gen_dir")
        if not ref or not gen:
            console.error("fad-gate needs --ref and --gen dirs")
            return {"task": task, "error": "missing dirs"}
        rec = compute(cfg, ref, gen, limit=kwargs.get("limit", 0))
        threshold = kwargs.get("threshold")
        if threshold is None:
            threshold = 10.0
        out = fad_gate(rec.get("fad_clap"), threshold)
        console.ok(f"FAD gate: {out['reason']} (fad={out['fad']}, threshold={out['threshold']})")
        return {"task": task, **out, "metrics": rec}

    if task == "genre-gate":
        from .report import load_results

        rows = load_results(root)
        gates = cfg.eval.genre_gates or {}
        out = per_genre_gate(rows, gates)
        console.ok("Per-genre gate: " + ("PASSED" if out["passed"] else "FAILED"))
        for g, r in out["genres"].items():
            console.info(f"  {g}: clap={r['mean_clap']} dev={r['mean_abs_deviation']} "
                         f"{'ok' if r['passed'] else 'FAIL'}")
        return {"task": task, **out}

    console.error(f"Unknown evalx task {task!r}")
    return {"task": task, "error": f"unknown task {task}"}
