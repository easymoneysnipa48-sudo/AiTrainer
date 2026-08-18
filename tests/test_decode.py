"""Tests for universal audio decoding (m4a/AAC via ffmpeg fallback) and the
inventory ffprobe fallback."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest
import soundfile as sf

from musictrain.audio import decode
from musictrain.audio.features import load_audio
from musictrain.audio.inventory import _probe

_FFMPEG = decode.ffmpeg_available()


def _make_wav(path, sr=22050, seconds=1.0):
    rng = np.random.default_rng(0)
    sf.write(path, rng.standard_normal(int(sr * seconds)).astype("float32") * 0.1, sr)


def _make_m4a(wav_path, m4a_path):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(wav_path), "-c:a", "aac", str(m4a_path)],
        check=True, capture_output=True,
    )


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not available")
def test_load_audio_reads_m4a(tmp_path):
    wav = tmp_path / "src.wav"
    m4a = tmp_path / "clip.m4a"
    _make_wav(wav)
    _make_m4a(wav, m4a)
    assert m4a.exists()

    # soundfile alone cannot read m4a
    with pytest.raises(Exception):
        sf.info(m4a)

    y, sr = load_audio(m4a, sr=16000)
    assert sr == 16000
    assert y.ndim == 1
    assert len(y) > 0


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not available")
def test_load_any_falls_back_and_cleans_tmp(tmp_path):
    wav = tmp_path / "src.wav"
    m4a = tmp_path / "clip.m4a"
    _make_wav(wav)
    _make_m4a(wav, m4a)

    before = set(p.name for p in tmp_path.glob("mt_decode_*"))
    y, sr = decode.load_any(m4a, sr=22050)
    after = set(p.name for p in tmp_path.glob("mt_decode_*"))
    assert y is not None and sr == 22050
    assert before == after  # temp wav cleaned up


def test_load_audio_reads_wav(tmp_path):
    wav = tmp_path / "plain.wav"
    _make_wav(wav)
    y, sr = load_audio(wav, sr=8000)
    assert sr == 8000
    assert len(y) > 0


@pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not available")
def test_inventory_probe_reads_m4a(tmp_path):
    wav = tmp_path / "src.wav"
    m4a = tmp_path / "clip.m4a"
    _make_wav(wav, sr=44100)
    _make_m4a(wav, m4a)
    info = _probe(m4a)
    assert info.get("sample_rate") == 44100
    assert info.get("channels") == 1
    assert info.get("duration", 0) > 0.5
    assert info.get("format") == "aac"


def test_inventory_probe_missing_file(tmp_path):
    assert _probe(tmp_path / "nope.wav") == {}
