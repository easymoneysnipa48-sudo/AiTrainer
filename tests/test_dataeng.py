"""Unit tests for advanced data batch #31-#40 (musictrain.dataeng)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from musictrain import dataeng as de
from musictrain.config import Config


# --------------------------------------------------------------------------- #
# #33 corpus-wide embedding dedup
# --------------------------------------------------------------------------- #
def test_corpus_dedup_finds_duplicates():
    emb = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    out = de.corpus_dedup(emb, ["a", "b", "c"], threshold=0.99)
    assert out["n_duplicates"] == 1
    assert out["clusters"][0]["representative"] == "a"
    assert out["clusters"][0]["duplicates"] == ["b"]


def test_corpus_dedup_no_duplicates():
    emb = np.eye(4)
    out = de.corpus_dedup(emb, ["a", "b", "c", "d"], threshold=0.9)
    assert out["n_duplicates"] == 0


def test_corpus_dedup_single():
    out = de.corpus_dedup(np.array([[1.0, 0.0]]), ["a"])
    assert out["n"] == 1
    assert out["n_duplicates"] == 0


# --------------------------------------------------------------------------- #
# #34 SNR + sample quality
# --------------------------------------------------------------------------- #
def test_snr_estimate_positive_for_signal_vs_noise():
    sr = 32000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = np.zeros(sr, dtype=np.float32)
    rng = np.random.default_rng(0)
    y[:sr // 2] = (0.001 * rng.standard_normal(sr // 2)).astype(np.float32)
    y[sr // 2:] = (0.5 * np.sin(2 * np.pi * 440 * t[sr // 2:])).astype(np.float32)
    snr = de.snr_estimate(y, sr)
    assert snr is not None
    assert snr > 10.0


def test_snr_estimate_too_short_none():
    assert de.snr_estimate(np.zeros(100, dtype=np.float32), 32000) is None


# --------------------------------------------------------------------------- #
# #35 snapshots
# --------------------------------------------------------------------------- #
def test_snapshot_hashes_files(tmp_path):
    cfg = Config()
    cfg.project_root = tmp_path
    import soundfile as sf

    seg = tmp_path / "data" / "segments"
    seg.mkdir(parents=True)
    sf.write(seg / "clip.wav", np.zeros(3200, dtype=np.float32), 32000)

    manifest = de.snapshot(tmp_path, cfg, label="test", which="segments")
    assert manifest["n_files"] == 1
    assert len(manifest["files"][0]["sha256"]) == 64
    assert (tmp_path / "metadata" / "snapshots" / "test" / "manifest.json").exists()


# --------------------------------------------------------------------------- #
# #36 prompt expansion
# --------------------------------------------------------------------------- #
def test_expand_prompts():
    prompts = de.expand_prompts(7, seed=0)
    assert len(prompts) == 7
    assert all(p["synthetic"] for p in prompts)
    assert all(p["description"] and p["bpm"] and p["key"] for p in prompts)


# --------------------------------------------------------------------------- #
# #37 tag co-occurrence
# --------------------------------------------------------------------------- #
def test_tag_cooccurrence():
    rows = [
        {"genre": "trap", "mood": "dark", "section": "chorus"},
        {"genre": "trap", "mood": "dark", "section": "chorus"},
        {"genre": "lofi", "mood": "calm", "section": "intro"},
    ]
    out = de.tag_cooccurrence(rows)
    assert out["n_combos"] == 2
    top = out["top"][0]
    assert top["count"] == 2
    assert top["combo"]["genre"] == "trap"
    assert out["underrepresented"][0]["count"] == 1


# --------------------------------------------------------------------------- #
# #38 balanced sampling
# --------------------------------------------------------------------------- #
def test_balanced_sample_balances_classes():
    items = list(range(100))
    labels = ["a"] * 90 + ["b"] * 10
    idx = de.balanced_sample(items, labels, 10, seed=0)
    picked = [labels[i] for i in idx]
    assert "a" in picked and "b" in picked
    # minority class is over-sampled relative to its 10% share
    assert picked.count("b") >= 4


def test_balanced_sample_empty():
    assert de.balanced_sample([], [], 5) == []


# --------------------------------------------------------------------------- #
# #39 provenance
# --------------------------------------------------------------------------- #
def test_annotate_provenance_fills_defaults():
    rows = [{"a": 1}, {}]
    out = de.annotate_provenance(rows, source_url="http://x", license_name="CC0", origin="buy")
    assert all(r["source_url"] == "http://x" for r in out)
    assert all(r["license"] == "CC0" for r in out)
    assert all(r["origin"] == "buy" for r in out)


# --------------------------------------------------------------------------- #
# #40 pre-annotation
# --------------------------------------------------------------------------- #
def test_pre_annotate_writes_csv(tmp_path, monkeypatch):
    cfg = Config()
    cfg.project_root = tmp_path
    import soundfile as sf

    clean = tmp_path / "data" / "clean"
    clean.mkdir(parents=True)
    sf.write(clean / "song.wav", np.zeros(3200, dtype=np.float32), 32000)

    def fake_analyze(cfg, path, root):
        return {
            "duration": 0.1,
            "key": {"key": "A minor"},
            "beat_grid": {"tempo": 120.0},
            "structure": {"segments": [{"role": "intro"}]},
        }

    import musictrain.audio.analysis as analysis

    monkeypatch.setattr(analysis, "analyze_file", fake_analyze)
    rows = de.pre_annotate(tmp_path, cfg, which="clean", limit=1)
    assert len(rows) == 1
    assert rows[0]["key"] == "A minor"
    assert rows[0]["bpm"] == 120.0
    assert rows[0]["auto"] is True
    assert (tmp_path / "metadata" / "preannotated_labels.csv").exists()
