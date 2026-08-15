"""Unit tests for musictrain.viz pure helpers (scan_audio, _load_env)."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from musictrain.viz import _load_env, scan_audio


def test_scan_audio_collects_and_dedups(tmp_path):
    d = tmp_path / "data" / "clean"
    d.mkdir(parents=True)
    (d / "a.wav").write_bytes(b"x")
    (d / "b.wav").write_bytes(b"x")
    nested = d / "sub"
    nested.mkdir()
    (nested / "c.wav").write_bytes(b"x")  # non-recursive glob must skip this
    files = scan_audio(tmp_path, ["data/clean"])
    assert {p.name for p in files} == {"a.wav", "b.wav"}


def test_scan_audio_missing_dir_is_safe(tmp_path):
    assert scan_audio(tmp_path, ["nope"]) == []


def test_load_env_returns_frame_for_wav(tmp_path):
    sr = 8000
    t = np.arange(sr) / sr
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wav = tmp_path / "tone.wav"
    sf.write(wav, y, sr)
    df = _load_env(str(wav))
    assert df is not None and list(df.columns) == ["t", "amp"]


def test_load_env_returns_none_for_missing(tmp_path):
    assert _load_env(str(tmp_path / "missing.wav")) is None
