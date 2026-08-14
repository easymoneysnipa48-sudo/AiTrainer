#!/usr/bin/env python3
"""CI eval-regression gate (Advanced #50).

Builds a throwaway project from fixture eval results and asserts that
``gates.eval_gate``:

1. BLOCKS a candidate that regresses (CLAP drop beyond tolerance), and
2. PASSES a candidate within tolerance.

Exits non-zero if the gate misbehaves, so CI fails on real regressions.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rows(checkpoint: str, claps, devs, statuses=None) -> list:
    statuses = statuses or ["ok"] * len(claps)
    rows = []
    for i, (c, d, s) in enumerate(zip(claps, devs, statuses)):
        rows.append(
            {
                "checkpoint": checkpoint,
                "prompt": f"prompt {i} at 96 BPM",
                "bpm_target": 96,
                "clap_score": c,
                "deviation": d,
                "status": s,
                "section": "chorus",
            }
        )
    return rows


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        meta = root / "metadata"
        meta.mkdir()

        baseline = _rows("base", [0.50] * 4, [0.10] * 4)
        regression = _rows("bad", [0.30] * 4, [0.25] * 4)
        within_tol = _rows("cand", [0.49] * 4, [0.11] * 4)
        (meta / "eval_results.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in baseline + regression + within_tol)
        )

        from musictrain.config import Config
        from musictrain.gates import eval_gate

        cfg = Config()
        cfg.project_root = root

        blocked = eval_gate(root, cfg, "base", "bad",
                            max_clap_drop=0.02, max_deviation_increase=0.05)
        if blocked.get("passed", True):
            print("FAIL: regression was not blocked by the gate")
            return 1

        passed = eval_gate(root, cfg, "base", "cand",
                           max_clap_drop=0.02, max_deviation_increase=0.05)
        if not passed.get("passed", False):
            print("FAIL: within-tolerance candidate was blocked")
            return 1

        print("gate OK: regression blocked, within-tolerance candidate passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
