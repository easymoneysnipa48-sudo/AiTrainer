"""Fast CI smoke test for the non-ML pipeline stages.

Exercises config, eval-set generation, label validation, BPM checking,
inventory, and report export — no torch/transformers/models needed.

Run with: python tests/smoke.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from musictrain.audio.inventory import inventory
from musictrain.config import Config
from musictrain.evaluate import check as bpm_check
from musictrain.evalset import build, load
from musictrain.labels import check as labels_check
from musictrain.report import export

ROOT = Path(__file__).resolve().parent.parent


def _make_click(path: Path, bpm: float = 120.0, secs: float = 8.0, sr: int = 32000) -> None:
    t = np.arange(int(sr * secs)) / sr
    y = 0.05 * np.sin(2 * np.pi * 220 * t)
    beat = 60.0 / bpm
    y = y + 0.9 * ((t % beat) < 0.012).astype(float)
    sf.write(path, y, sr)


def main() -> None:
    cfg = Config.load(ROOT / "configs" / "default.yaml")
    cfg.project_root = ROOT

    # 1. config loads and coerces nested sections
    assert cfg.normalize.sample_rate == 32000
    assert cfg.mlflow.enabled is True

    # 2. eval prompt set generation
    build(ROOT, force=True)
    prompts = load(ROOT)
    assert len(prompts) >= 40, f"expected >=40 prompts, got {len(prompts)}"

    # 3. label validation against the controlled vocabulary
    issues = labels_check(ROOT / "metadata" / "labels.csv")
    assert not issues, f"label issues: {issues}"

    # 4. synthetic click track -> BPM detection
    clean = ROOT / "data" / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    wav = clean / "smoke_click.wav"
    _make_click(wav, bpm=120.0)
    report = bpm_check(cfg, wav, target_bpm=120.0)
    assert report["status"] == "ok", f"BPM check failed: {report}"

    # 5. audio inventory
    inv = inventory(ROOT, which="clean")
    assert any(r.get("valid") for r in inv), "no valid audio in inventory"

    # 6. report export from a results fixture
    fixture = [
        {
            "experiment_id": "ci", "checkpoint": "facebook/musicgen-small",
            "prompt": "intro, 72 BPM, A minor, dark piano loop",
            "seed": 42, "n_seeds": 1, "ok_seeds": 1, "section": "intro",
            "genre": "melodic trap", "key": "A minor", "bpm_target": 72,
            "detected_bpm": 72.1, "deviation": 0.0014, "clap_score": 0.4,
            "human_rating": None, "status": "ok", "notes": "",
            "audio_path": str(wav),
        }
    ]
    (ROOT / "metadata" / "eval_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in fixture) + "\n"
    )
    out = export(cfg)
    assert Path(out["csv"]).exists() and Path(out["html"]).exists()

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
