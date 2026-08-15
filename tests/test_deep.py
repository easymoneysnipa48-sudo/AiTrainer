"""Unit tests for advanced eval #11-#20.

Covers negative-control + paraphrase prompt generation and robustness
(evalset), and the deep signal analysis module (tempo drift, groove,
loudness, stereo, artifacts, spectral profile, onset density, masking).
"""
from __future__ import annotations

import numpy as np
import pytest

from musictrain import evalset
from musictrain.audio import deep as dp


def _tone(freq: float, seconds: float = 1.0, sr: int = 32000,
          amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _clicks(bpm: float, seconds: float, sr: int = 32000) -> np.ndarray:
    """Metronomic impulse train at `bpm` (120 BPM -> 0.5s period)."""
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    period = int(sr * 60 / bpm)
    for t0 in range(0, len(y) - 100, period):
        y[t0:t0 + 100] = 0.8
    return y


# --------------------------------------------------------------------------- #
# #11 negative controls / #12 paraphrases (evalset)
# --------------------------------------------------------------------------- #
def test_negative_controls_flagged():
    neg = evalset.negative_controls(n=3)
    assert len(neg) == 3
    assert all(p["negative_control"] for p in neg)
    assert all(p["description"] for p in neg)


def test_paraphrase_prompts_share_intent():
    par = evalset.paraphrase_prompts(n_groups=2)
    intents = {p["intent"] for p in par}
    assert len(intents) == 2
    for intent in intents:
        group = [p for p in par if p["intent"] == intent]
        assert len(group) == 3
        assert all(p["paraphrase"] for p in group)


def test_paraphrase_robustness_spread():
    rows = [
        {"intent": "g0", "paraphrase": True, "clap_score": 0.5},
        {"intent": "g0", "paraphrase": True, "clap_score": 0.52},
        {"intent": "g0", "paraphrase": True, "clap_score": 0.51},
    ]
    out = evalset.paraphrase_robustness(rows)
    assert out["n_groups"] == 1
    assert out["robust"] is True
    assert out["mean_spread"] < 0.1


def test_paraphrase_robustness_high_spread():
    rows = [
        {"intent": "g0", "paraphrase": True, "clap_score": 0.2},
        {"intent": "g0", "paraphrase": True, "clap_score": 0.8},
    ]
    out = evalset.paraphrase_robustness(rows)
    assert out["robust"] is False


def test_paraphrase_robustness_no_groups():
    assert evalset.paraphrase_robustness([{"clap_score": 0.5}])["n_groups"] == 0


# --------------------------------------------------------------------------- #
# #13 tempo drift
# --------------------------------------------------------------------------- #
def test_tempo_drift_steady_click_track():
    y = _clicks(120, seconds=10.0)
    out = dp.tempo_drift(y, 32000, window_seconds=4.0)
    assert out is not None
    assert 100 <= out["mean_bpm"] <= 150
    assert out["verdict"] in ("steady", "rushing", "dragging")


def test_tempo_drift_short_audio_none():
    y = _tone(440, seconds=1.0)
    assert dp.tempo_drift(y, 32000) is None


# --------------------------------------------------------------------------- #
# #14 groove
# --------------------------------------------------------------------------- #
def test_groove_returns_feel():
    y = _clicks(120, seconds=4.0)
    out = dp.groove(y, 32000)
    assert out is not None
    assert out["feel"] in ("straight", "moderate", "swung", "unknown")
    assert out["tempo"] > 0


def test_groove_short_audio_none():
    assert dp.groove(_tone(440, seconds=0.2), 32000) is None


# --------------------------------------------------------------------------- #
# #15 loudness profile
# --------------------------------------------------------------------------- #
def test_loudness_profile():
    out = dp.loudness_profile(_tone(440, seconds=1.0), 32000)
    assert len(out["envelope"]) > 0
    assert out["dynamic_range_db"] is not None


# --------------------------------------------------------------------------- #
# #16 stereo profile
# --------------------------------------------------------------------------- #
def _write_stereo(tmp_path, left, right, sr=32000):
    import soundfile as sf

    p = tmp_path / "stereo.wav"
    sf.write(p, np.stack([left, right], axis=1), sr)
    return p


def test_stereo_mono_correlated(tmp_path):
    y = _tone(440, seconds=0.5)
    p = _write_stereo(tmp_path, y, y)
    out = dp.stereo_profile(p)
    assert out["stereo_width"] == pytest.approx(0.0, abs=0.01)
    assert out["phase_correlation"] == pytest.approx(1.0, abs=0.01)
    assert out["mono_compatible"] is True


def test_stereo_out_of_phase(tmp_path):
    y = _tone(440, seconds=0.5)
    p = _write_stereo(tmp_path, y, -y)
    out = dp.stereo_profile(p)
    assert out["phase_correlation"] == pytest.approx(-1.0, abs=0.01)
    assert out["mono_compatible"] is False


def test_stereo_mono_source(tmp_path):
    import soundfile as sf

    p = tmp_path / "mono.wav"
    sf.write(p, _tone(440, seconds=0.5), 32000)
    out = dp.stereo_profile(p)
    assert out["channels"] == 1


# --------------------------------------------------------------------------- #
# #17 artifacts
# --------------------------------------------------------------------------- #
def test_artifacts_clean_tone():
    out = dp.detect_artifacts(_tone(440, seconds=1.0), 32000)
    assert out["clean"] is True
    assert out["n_clicks"] == 0


def test_artifacts_detects_click():
    y = _tone(440, seconds=1.0)
    y[16000] = 1.0  # single impulse
    out = dp.detect_artifacts(y, 32000)
    assert out["n_clicks"] >= 1


def test_artifacts_detects_dc_offset():
    y = _tone(440, seconds=1.0) + 0.1
    out = dp.detect_artifacts(y, 32000)
    assert out["dc_offset"] > 0.05


# --------------------------------------------------------------------------- #
# #18 spectral profile
# --------------------------------------------------------------------------- #
def test_spectral_profile_centroid_near_tone():
    out = dp.spectral_profile(_tone(440, seconds=1.0), 32000)
    assert 300 <= out["centroid_hz"] <= 600
    assert out["crest_factor"] > 1.0


# --------------------------------------------------------------------------- #
# #19 onset density map
# --------------------------------------------------------------------------- #
def test_onset_density_map():
    y = _clicks(120, seconds=4.0)
    out = dp.onset_density_map(y, 32000, bin_seconds=2.0)
    assert out["overall_onset_density"] > 0
    assert len(out["map"]) == 2


# --------------------------------------------------------------------------- #
# #20 frequency masking
# --------------------------------------------------------------------------- #
def test_frequency_masking_collision():
    # 440 vs 520 Hz: within ~1/3 octave, resolvable at n_fft=2048
    y = _tone(440, seconds=1.0, amp=0.9) + _tone(520, seconds=1.0, amp=0.05)
    out = dp.frequency_masking(y, 32000)
    assert out["masking_risk"] is True
    assert out["n_collisions"] >= 1


def test_frequency_masking_no_collision():
    y = _tone(220, seconds=1.0) + _tone(4000, seconds=1.0)
    out = dp.frequency_masking(y, 32000)
    assert out["n_collisions"] == 0
