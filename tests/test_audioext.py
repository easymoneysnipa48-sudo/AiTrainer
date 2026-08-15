import numpy as np
import pytest

from musictrain import audioext
from musictrain.config import Config


def _tone(hz, sr=8000, seconds=1.0, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _silence(sr=8000, seconds=0.5):
    return np.zeros(int(sr * seconds), dtype=np.float32)


def test_transposed_key_major():
    assert audioext.transposed_key("C major", 2) == "D major"
    assert audioext.transposed_key("A major", -1) == "G# major"


def test_transposed_key_minor():
    assert audioext.transposed_key("A minor", 3) == "C minor"


def test_tempo_key_augment_folds_ratio():
    out = audioext.tempo_key_augment(120.0, "C major", semitones=2, tempo_ratio=1.5)
    assert out["new_bpm"] == 180.0  # 1.5x is in the fold set
    assert out["new_key"] == "D major"
    assert out["in_fold_set"] is True


def test_tempo_key_augment_snaps_non_fold():
    out = audioext.tempo_key_augment(120.0, "C major", tempo_ratio=1.4)
    # 1.4 is not in the fold set; snaps to nearest allowed (3/2 = 1.5)
    assert out["tempo_ratio"] == 1.5
    assert out["in_fold_set"] is True


def test_diarize_silence_is_empty():
    y = _silence()
    assert audioext.diarize(y, 8000) == []


def test_diarize_single_voice():
    y = _tone(220, seconds=2.0)
    segs = audioext.diarize(y, 8000, energy_threshold=0.02)
    assert len(segs) >= 1
    assert segs[0]["end"] > segs[0]["start"]


def test_transcribe_midi_detects_note():
    y = _tone(440, seconds=1.0)  # A4 -> MIDI 69
    notes = audioext.transcribe_midi(y, 8000)
    assert notes, "expected at least one note"
    # tolerance: autocorrelation on 8 kHz sine should land near A4
    assert 60 <= notes[0]["note"] <= 76


def test_transcribe_midi_silence_is_empty():
    notes = audioext.transcribe_midi(_silence(), 8000)
    assert notes == []


def test_bundle_and_verify(tmp_path):
    (tmp_path / "data" / "clean").mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    (tmp_path / "data" / "clean" / "a.wav").write_bytes(b"\x00" * 100)
    (tmp_path / "data" / "clean" / "b.wav").write_bytes(b"\x01" * 100)
    cfg = Config(project_root=tmp_path)
    out = audioext.bundle(tmp_path, cfg, which="clean")
    assert out["n_files"] == 2
    assert audioext.verify_bundle(__import__("pathlib").Path(out["archive"]))["ok"] is True


def test_bundle_missing_dir(tmp_path):
    cfg = Config(project_root=tmp_path)
    out = audioext.bundle(tmp_path, cfg, which="nope")
    assert out["error"] == "no_dir"


def test_verify_bundle_missing():
    assert audioext.verify_bundle(__import__("pathlib").Path("/nonexistent.zip"))["error"] == "not_found"
