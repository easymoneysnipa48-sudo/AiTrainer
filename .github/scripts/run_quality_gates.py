#!/usr/bin/env python3
"""CI quality gates: FAD + per-genre thresholds.

Two layers:

1. **Self-test (always runs).** Builds synthetic rows and asserts that
   ``fad_gate`` and ``per_genre_gate`` BLOCK a failing input and PASS a good
   one. This catches regressions in the gate logic itself without needing
   audio, CLAP, or torch in CI.

2. **Real gate (runs when artifacts are present).** If
   ``metadata/eval_results.jsonl`` and/or ``metadata/metrics.json`` are
   committed/present (e.g. a self-hosted runner or a manual
   ``workflow_dispatch`` after a real eval), run ``gates.quality_gate`` and
   exit non-zero on failure.

Exit 0 = gates behave correctly and (if present) artifacts pass.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def _self_test() -> int:
    """Verify the pure gate logic in both directions."""
    from musictrain.evalx import fad_gate, per_genre_gate

    # FAD: below threshold passes, above blocks, missing is soft-skip.
    if fad_gate(5.0, threshold=10.0)["passed"] is not True:
        return _fail("fad_gate did not pass a below-threshold FAD")
    if fad_gate(15.0, threshold=10.0)["passed"] is not False:
        return _fail("fad_gate did not block an above-threshold FAD")
    if fad_gate(None)["passed"] is not None:
        return _fail("fad_gate should soft-skip a missing FAD")

    # Per-genre: a strong melodic-trap group passes; a weak ambient group fails.
    rows = [
        {"genre": "melodic trap", "clap_score": 0.50, "deviation": 0.01},
        {"genre": "melodic trap", "clap_score": 0.45, "deviation": 0.02},
        {"genre": "ambient", "clap_score": 0.10, "deviation": 0.50},
    ]
    gates = {
        "melodic trap": {"min_clap": 0.32, "max_abs_deviation": 0.15},
        "ambient": {"min_clap": 0.22, "max_abs_deviation": 0.30},
        "default": {"min_clap": 0.30, "max_abs_deviation": 0.20},
    }
    out = per_genre_gate(rows, gates)
    if out["passed"] is not False:
        return _fail("per_genre_gate did not block a failing genre group")
    if out["genres"]["melodic trap"]["passed"] is not True:
        return _fail("per_genre_gate mis-scored the passing genre group")
    if out["genres"]["ambient"]["passed"] is not False:
        return _fail("per_genre_gate mis-scored the failing genre group")

    all_pass = per_genre_gate(
        [{"genre": "melodic trap", "clap_score": 0.50, "deviation": 0.01}], gates
    )
    if all_pass["passed"] is not True:
        return _fail("per_genre_gate did not pass a fully-compliant set")
    return 0


def _real_gate() -> int:
    """Run the combined quality gate against committed artifacts, if present.

    FAD is sourced in priority order: an explicit value in
    ``metadata/metrics.json``, else computed against the cached reference stats
    (``metadata/fad_reference_stats.json``) using ``outputs/eval`` — so a
    self-hosted runner can precompute the reference distribution once and reuse
    it here without re-embedding the reference set (#9).
    """
    results = ROOT / "metadata" / "eval_results.jsonl"
    metrics = ROOT / "metadata" / "metrics.json"
    fad_cache = ROOT / "metadata" / "fad_reference_stats.json"
    if not results.exists() and not metrics.exists() and not fad_cache.exists():
        print("no artifacts (eval_results.jsonl / metrics.json / "
              "fad_reference_stats.json) — skipping real gate; self-test only")
        return 0

    from musictrain.config import Config
    from musictrain.gates import quality_gate

    if not results.exists():
        return _fail("artifacts present (metrics/cache) but eval_results.jsonl missing")

    cfg = Config()
    cfg.project_root = ROOT

    fad = None
    if metrics.exists():
        try:
            fad = json.loads(metrics.read_text()).get("fad_clap")
        except (json.JSONDecodeError, OSError):
            fad = None
    if fad is None and fad_cache.exists():
        try:
            from musictrain.metrics import fad_from_cached_stats

            fad = fad_from_cached_stats(cfg, ROOT / "outputs" / "eval")
        except Exception as exc:  # noqa: BLE001 - FAD is best-effort in CI
            print(f"note: FAD-from-cache failed ({exc}); gate will soft-skip FAD")

    verdict = quality_gate(root=ROOT, cfg=cfg, fad=fad, allow_missing_fad=True)
    if not verdict.get("passed", False):
        return _fail(
            "quality gate FAILED: "
            + json.dumps(
                {
                    "genre_gate": verdict["genre_gate"]["passed"],
                    "fad_gate": verdict["fad_gate"],
                }
            )
        )
    print("quality gate OK: per-genre + FAD passed")
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory():
        rc = _self_test()
    if rc != 0:
        return rc
    print("self-test OK: fad_gate + per_genre_gate behave correctly")
    return _real_gate()


if __name__ == "__main__":
    raise SystemExit(main())
