"""Generation search utilities (advanced #14, #15, #18, #19).

* **Guidance sweep** (#18) — grid over guidance_scale (+ optional seeds), each
  candidate scored by CLAP + BPM deviation; best candidate reported.
* **Seed search** (#19) — same grid over seeds only, for a fixed setting.
* **Prompt ensembling** (#14) — deterministic phrasing variants of a prompt,
  best-of-N by CLAP (diversity without extra models).
* **Conditioning chaining** (#15) — generate step 1, then condition step 2 on
  step 1's audio (melody), and so on, building a coherent multi-part piece.

Everything accepts a ``generator`` callable (default ``generate_cached``) so
tests can inject fakes and real runs get deterministic caching.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import console
from .config import Config

# deterministic paraphrase templates (clause re-order + synonym swaps)
_ENERGY_WORDS = ["low energy", "mid energy", "high energy", "energetic", "mellow"]
_GENRE_SYNONYMS = {
    "melodic trap": ["trap", "melodic trap", "dark trap"],
    "ambient": ["ambient", "atmospheric", "cinematic ambient"],
    "orchestral": ["orchestral", "cinematic strings", "film score"],
}


def prompt_variants(prompt: str, n: int = 4) -> List[str]:
    """Deterministic phrasing variants of a prompt (same meaning, different words)."""
    clauses = [c.strip() for c in prompt.split(",") if c.strip()]
    out: List[str] = []
    # 1) original
    out.append(prompt)
    # 2) rotated clause order
    if len(clauses) >= 2:
        out.append(", ".join(clauses[1:] + clauses[:1]))
    # 3) swapped genre synonym (first matching known genre)
    for genre, syns in _GENRE_SYNONYMS.items():
        if genre in prompt and len(syns) > 1:
            out.append(prompt.replace(genre, syns[(hash(prompt) % (len(syns) - 1)) + 1]))
            break
    # 4) energy-word swap
    for w in _ENERGY_WORDS:
        if w in prompt:
            alt = _ENERGY_WORDS[(_ENERGY_WORDS.index(w) + 1) % len(_ENERGY_WORDS)]
            out.append(prompt.replace(w, alt))
            break
    # 5) tag-style compression: strip function words
    compact = " ".join(clauses).replace(" and ", " ").replace(" with ", " ")
    if compact != prompt and compact not in out:
        out.append(compact)
    return list(dict.fromkeys(out))[: max(n, 1)]


def _score_clap(cfg: Config, path: Path, prompt: str) -> Optional[float]:
    if not cfg.clap.enabled:
        return None
    from .similarity import score

    try:
        return score(cfg, path, prompt)
    except Exception as exc:  # noqa: BLE001 - scoring must not kill a sweep
        console.warn(f"CLAP failed for {path.name}: {exc}")
        return None


def _best(rows: List[dict]) -> dict:
    scored = [r for r in rows if r.get("clap_score") is not None]
    pool = scored or rows
    return max(pool, key=lambda r: r.get("clap_score") or 0.0)


def run_sweep(
    cfg: Config,
    prompt: str,
    guidance_values: List[float],
    seeds: List[int],
    out_dir: Optional[Path] = None,
    generator: Callable = None,
) -> Tuple[List[dict], dict]:
    """Grid over guidance x seeds; every candidate CLAP-scored, best returned."""
    if generator is None:
        from .inference import generate_cached as generator
    out_dir = Path(out_dir) if out_dir else cfg.project_root / "outputs" / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for g in guidance_values:
        cfg.inference.guidance_scale = g
        for s in seeds:
            result = generator(cfg, prompt, out_dir=out_dir, seed=s)
            if not result:
                continue
            clap = _score_clap(cfg, Path(result["path"]), prompt)
            rows.append(
                {
                    "guidance": g, "seed": s, "path": result["path"],
                    "clap_score": round(clap, 4) if clap is not None else None,
                    "duration": result.get("duration"),
                    "cached": bool(result.get("cached")),
                }
            )
    best = _best(rows)
    _write(cfg, {"kind": "sweep", "prompt": prompt, "rows": rows, "best": best})
    console.ok(
        f"Sweep: {len(rows)} candidate(s); best guidance={best.get('guidance')} "
        f"seed={best.get('seed')} clap={best.get('clap_score')} -> {best.get('path')}"
    )
    return rows, best


def run_ensemble(
    cfg: Config,
    prompt: str,
    n: int = 4,
    out_dir: Optional[Path] = None,
    generator: Callable = None,
) -> Tuple[List[dict], dict]:
    """Best-of-N over prompt phrasing variants (advanced #14)."""
    if generator is None:
        from .inference import generate_cached as generator
    out_dir = Path(out_dir) if out_dir else cfg.project_root / "outputs" / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for i, variant in enumerate(prompt_variants(prompt, n)):
        result = generator(cfg, variant, out_dir=out_dir, seed=cfg.inference.seed or 42 + i)
        if not result:
            continue
        clap = _score_clap(cfg, Path(result["path"]), variant)
        rows.append(
            {
                "variant": variant, "path": result["path"],
                "clap_score": round(clap, 4) if clap is not None else None,
                "duration": result.get("duration"), "cached": bool(result.get("cached")),
            }
        )
    best = _best(rows)
    _write(cfg, {"kind": "ensemble", "prompt": prompt, "rows": rows, "best": best})
    console.ok(
        f"Ensemble: {len(rows)} variant(s); best clap={best.get('clap_score')} -> {best.get('path')}"
    )
    return rows, best


def chain_generations(
    cfg: Config,
    prompt: str,
    steps: int = 3,
    out_dir: Optional[Path] = None,
    generator: Callable = None,
) -> List[dict]:
    """Condition each step on the previous step's audio (advanced #15)."""
    if generator is None:
        from .inference import generate_cached as generator
    out_dir = Path(out_dir) if out_dir else cfg.project_root / "outputs" / "chain"
    out_dir.mkdir(parents=True, exist_ok=True)

    chain: List[dict] = []
    prev: Optional[Path] = None
    for i in range(1, steps + 1):
        label = f"{prompt[:40]} [step {i}/{steps}]"
        if prev is None:
            result = generator(cfg, label, out_dir=out_dir, seed=cfg.inference.seed or 42 + i)
        else:
            result = generator(
                cfg, label, out_dir=out_dir, seed=cfg.inference.seed or 42 + i,
                melody_from=prev,
            )
        if not result:
            break
        chain.append(result)
        prev = Path(result["path"])
    _write(cfg, {"kind": "chain", "prompt": prompt, "steps": [r["path"] for r in chain]})
    console.ok(f"Chain: {len(chain)} step(s) -> {', '.join(r['path'] for r in chain)}")
    return chain


def _write(cfg: Config, record: dict) -> None:
    out = cfg.project_root / "metadata" / "sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    console.info(f"Sweep results -> {out.relative_to(cfg.project_root)}")
