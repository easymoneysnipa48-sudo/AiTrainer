"""Unit tests for the Phase 1 dataset-hygiene modules.

Covers: audio quality, loudness normalization, duplicate detection, embedding
index helpers, auto-label scoring, corpus statistics, OOD curation, and stem
separation guards.

Heavy dependencies (CLAP, Demucs) are *not* loaded — these tests exercise the
pure logic, synthetic-audio signal paths, and lazy-import guards so they run in
seconds without model downloads.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from musictrain import corpus as corpus_mod
from musictrain import dedup
from musictrain import embeddings
from musictrain import ood
from musictrain import stems
from musictrain.audio import quality as quality_mod
from musictrain.autolabel import _cos
from musictrain.config import Config
from musictrain.corpus import _bpm_histogram, _counts
from musictrain.loudnorm import loudnorm


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


def make_wav(path: Path, waveform: str = "sine", sr: int = 32000, seconds: float = 1.0) -> Path:
    """Write a small deterministic synthetic WAV (no ML deps)."""
    t = np.arange(int(sr * seconds)) / sr
    if waveform == "sine":
        y = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    elif waveform == "click":
        bpm = 120.0
        y = 0.3 * np.sin(2 * np.pi * 440.0 * t)
        beat = 60.0 / bpm
        y = y + 0.9 * ((t % beat) < 0.012).astype(float)
    elif waveform == "silence":
        y = np.zeros_like(t)
    elif waveform == "clipped":
        y = np.clip(3.0 * np.sin(2 * np.pi * 440.0 * t), -1.0, 1.0)
    elif waveform == "dc":
        y = np.full_like(t, 0.5)
    else:
        raise ValueError(waveform)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr)
    return path


def clean_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "clean"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(root: Path, rows: list) -> Path:
    p = root / "metadata" / "manifest.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


# --------------------------------------------------------------------------- #
# audio/quality.py (#4)
# --------------------------------------------------------------------------- #


def test_quality_keys_and_bounded_score(tmp_path):
    p = make_wav(tmp_path / "a.wav")
    rec = quality_mod.analyze_file(p, Config().quality)
    for key in ("path", "duration", "sample_rate", "quality_score", "grade", "flags"):
        assert key in rec, rec
    assert 0.0 <= rec["quality_score"] <= 100.0
    assert rec["grade"] in ("A", "B", "C", "F")


def test_quality_detects_clipping(tmp_path):
    p = make_wav(tmp_path / "a.wav", "clipped")
    rec = quality_mod.analyze_file(p, Config().quality)
    assert any("clipping" in f for f in rec["flags"]), rec["flags"]


def test_quality_detects_silence(tmp_path):
    p = make_wav(tmp_path / "a.wav", "silence")
    rec = quality_mod.analyze_file(p, Config().quality)
    assert any("silence" in f for f in rec["flags"]), rec["flags"]


def test_quality_detects_dc_offset(tmp_path):
    p = make_wav(tmp_path / "a.wav", "dc")
    rec = quality_mod.analyze_file(p, Config().quality)
    assert any("DC offset" in f for f in rec["flags"]), rec["flags"]


def test_quality_full_sweep_writes_report(tmp_path):
    d = clean_dir(tmp_path)
    make_wav(d / "a.wav")
    cfg = Config()
    results = quality_mod.quality(tmp_path, cfg, which="clean")
    assert len(results) == 1
    assert (tmp_path / "metadata" / "quality_report.json").exists()


# --------------------------------------------------------------------------- #
# loudnorm.py (#3)
# --------------------------------------------------------------------------- #


def test_loudnorm_dry_run_invokes_no_ffmpeg(tmp_path, monkeypatch):
    d = clean_dir(tmp_path)
    make_wav(d / "a.wav")
    called = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: called.append(a) or SimpleNamespace(returncode=0, stderr=""),
    )
    conv, skip, fail = loudnorm(tmp_path, Config(), which="clean", dry_run=True)
    assert (conv, skip, fail) == (0, 0, 0)
    assert not called


def test_loudnorm_force_writes(tmp_path, monkeypatch):
    d = clean_dir(tmp_path)
    make_wav(d / "a.wav")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stderr=""),
    )
    conv, skip, fail = loudnorm(tmp_path, Config(), which="clean", force=True)
    assert conv == 1 and skip == 0 and fail == 0


def test_loudnorm_ffmpeg_failure_counts_failed(tmp_path, monkeypatch):
    d = clean_dir(tmp_path)
    make_wav(d / "a.wav")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="boom"),
    )
    conv, skip, fail = loudnorm(tmp_path, Config(), which="clean", force=True)
    assert conv == 0 and fail == 1


# --------------------------------------------------------------------------- #
# dedup.py (#1)
# --------------------------------------------------------------------------- #


def test_pitch_invariant_sim_identity_and_shift():
    a = np.zeros(12)
    a[0] = 1.0
    shifted = np.roll(a, 3)
    assert abs(dedup._pitch_invariant_sim(a, a) - 1.0) < 1e-9
    assert dedup._pitch_invariant_sim(a, shifted) > 0.999


def test_chroma_fingerprint_shape(tmp_path):
    p = make_wav(tmp_path / "a.wav")
    fp = dedup.chroma_fingerprint(p)
    assert fp.shape == (12,)
    assert fp.dtype == np.float32


def test_find_duplicates_exact(tmp_path):
    d = clean_dir(tmp_path)
    p1 = make_wav(d / "a.wav")
    p2 = d / "b.wav"
    p2.write_bytes(p1.read_bytes())  # byte-identical copy
    cfg = Config()
    cfg.dedup.exact_only = True
    report = dedup.find_duplicates(tmp_path, cfg, which="clean")
    assert report["duplicate_groups"] == 1
    assert report["duplicate_files"] == 1
    assert report["groups"][0]["kind"] == "exact"


# --------------------------------------------------------------------------- #
# embeddings.py (#9) — non-CLAP helpers only
# --------------------------------------------------------------------------- #


def test_embeddings_scan_audio_only(tmp_path):
    d = clean_dir(tmp_path)
    (d / "a.wav").write_bytes(b"x")
    (d / "b.mp3").write_bytes(b"x")
    (d / "c.txt").write_bytes(b"x")
    found = embeddings._scan(d)
    assert len(found) == 2


def test_embeddings_cache_path(tmp_path):
    assert embeddings._cache_path(tmp_path) == tmp_path / "metadata" / "audio_embeddings.json"


# --------------------------------------------------------------------------- #
# autolabel.py (#6) — cosine helper
# --------------------------------------------------------------------------- #


def test_cos_identical_and_orthogonal():
    assert abs(_cos(np.array([1.0, 0.0]), np.array([1.0, 0.0])) - 1.0) < 1e-9
    assert abs(_cos(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-9


# --------------------------------------------------------------------------- #
# corpus.py (#7)
# --------------------------------------------------------------------------- #


def test_counts_flat_and_nested():
    assert _counts([1, 2, 2, None]) == {"1": 1, "2": 2}
    assert _counts([[1, 2], [1]]) == {"1": 2, "2": 1}


def test_bpm_histogram_bins():
    h = _bpm_histogram([72.0, 96.0, 140.0])
    assert h["70-80"] == 1
    assert h["90-100"] == 1
    assert h["140-150"] == 1


def test_corpus_end_to_end(tmp_path):
    write_manifest(
        tmp_path,
        [
            {"path": "a.wav", "duration": 10, "bpm": 72, "key": "A minor",
             "genre": "melodic trap", "mood": ["dark"], "instruments": ["piano"], "section": "verse"},
            {"path": "b.wav", "duration": 20, "bpm": 96, "key": "C minor",
             "genre": "trap", "mood": ["energetic"], "instruments": ["drums"], "section": "chorus"},
        ],
    )
    stats = corpus_mod.corpus(tmp_path, Config())
    assert stats["n_tracks"] == 2
    assert stats["total_duration_s"] == 30.0
    assert stats["bpm"]["mean"] == 84.0
    assert stats["genre"]["melodic trap"] == 1
    assert stats["key"]["A minor"] == 1
    assert (tmp_path / "metadata" / "corpus_stats.json").exists()


# --------------------------------------------------------------------------- #
# ood.py (#8)
# --------------------------------------------------------------------------- #


def test_ood_flags_bpm_and_tag_outliers(tmp_path):
    write_manifest(
        tmp_path,
        [
            {"path": "a.wav", "bpm": 200, "genre": "trap", "mood": [], "instruments": []},
            {"path": "b.wav", "bpm": 100, "genre": "ambient", "mood": [], "instruments": []},
            {"path": "c.wav", "bpm": 100, "genre": "trap", "mood": [], "instruments": []},
        ],
    )
    cfg = Config()
    cfg.ood.bpm_range = [70.0, 160.0]
    cfg.ood.tag_exclude = ["ambient", "orchestral"]
    flagged = ood.curate_ood(tmp_path, cfg)
    paths = {r["path"] for r in flagged}
    assert paths == {"a.wav", "b.wav"}
    assert all(r.get("ood_reasons") for r in flagged)
    assert (tmp_path / "metadata" / "ood_tracks.json").exists()


# --------------------------------------------------------------------------- #
# stems.py (#5) — lazy-import guard + scan
# --------------------------------------------------------------------------- #


def test_stems_load_demucs_returns_none_when_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "demucs" or name.startswith("demucs."):
            raise ImportError("demucs not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert stems._load_demucs() is None


def test_stems_separate_returns_empty_when_demucs_missing(tmp_path, monkeypatch):
    clean_dir(tmp_path)
    monkeypatch.setattr(stems, "_load_demucs", lambda: None)
    assert stems.separate_stems(tmp_path, Config()) == []


def test_stems_scan(tmp_path):
    d = clean_dir(tmp_path)
    (d / "a.wav").write_bytes(b"x")
    (d / "b.flac").write_bytes(b"x")
    (d / "c.txt").write_bytes(b"x")
    assert len(stems._scan(d)) == 2
