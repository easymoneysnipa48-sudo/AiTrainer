"""Tests for advanced batch 3 (#21-#30): resynth, invert, active, augment,
dedup segments, auto sections, drift, curation, embed refresh, labelprop."""
import json
from pathlib import Path

import numpy as np
import pytest

import soundfile as sf


def _write_wav(path: Path, seconds: float = 1.0, freq: float = 220.0, sr: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    return path


# --------------------------------------------------------------------------- #
# 21 — resynth
# --------------------------------------------------------------------------- #
def test_mix_applies_gains():
    from musictrain.resynth import _mix

    a = np.ones(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32)
    out = _mix({"a": a, "b": b}, {"a": 2.0, "b": 0.5}, 100)
    np.testing.assert_allclose(out, 2.5)


def test_resynth_writes_mix(tmp_path):
    from musictrain.config import Config
    from musictrain.resynth import resynth

    track = tmp_path / "data" / "clean" / "song.wav"
    _write_wav(track)
    stem_dir = tmp_path / "data" / "stems" / "song"
    stem_dir.mkdir(parents=True)
    _write_wav(stem_dir / "vocals.wav", freq=440.0)
    _write_wav(stem_dir / "drums.wav", freq=110.0)
    (tmp_path / "metadata").mkdir(exist_ok=True)
    (tmp_path / "metadata" / "stems.json").write_text(
        json.dumps(
            [
                {
                    "track": "data/clean/song.wav",
                    "stems": {
                        "vocals": "data/stems/song/vocals.wav",
                        "drums": "data/stems/song/drums.wav",
                    },
                }
            ]
        )
    )

    cfg = Config()
    cfg.project_root = tmp_path
    results = resynth(tmp_path, cfg, gains={"vocals": 0.0, "drums": 1.0})
    assert len(results) == 1
    out = tmp_path / results[0]["output"]
    assert out.exists()
    assert (tmp_path / "metadata" / "resynth.json").exists()
    assert results[0]["gains"]["vocals"] == 0.0


def test_resynth_instrumental_drops_vocals(tmp_path):
    from musictrain.config import Config
    from musictrain.resynth import rebuild_instrumental

    track = tmp_path / "data" / "clean" / "song.wav"
    _write_wav(track)
    stem_dir = tmp_path / "data" / "stems" / "song"
    stem_dir.mkdir(parents=True)
    _write_wav(stem_dir / "vocals.wav", freq=440.0)
    _write_wav(stem_dir / "drums.wav", freq=110.0)
    (tmp_path / "metadata").mkdir(exist_ok=True)
    (tmp_path / "metadata" / "stems.json").write_text(
        json.dumps(
            [{"track": "data/clean/song.wav",
              "stems": {"vocals": "data/stems/song/vocals.wav",
                        "drums": "data/stems/song/drums.wav"}}]
        )
    )
    cfg = Config()
    cfg.project_root = tmp_path
    results = rebuild_instrumental(tmp_path, cfg)
    assert results[0]["gains"]["vocals"] == 0.0
    assert (tmp_path / "data" / "resynth_instrumental" / "song_resynth.wav").exists()


# --------------------------------------------------------------------------- #
# 22 — invert
# --------------------------------------------------------------------------- #
def test_template_prompt_uses_features():
    from musictrain.invert import template_prompt

    feats = {
        "beat_grid": {"tempo": 140.0},
        "key": {"key": "A minor"},
        "swing": {"feel": "swung"},
        "vocal": {"verdict": "vocal"},
        "onsets": {"onset_density": 8.0},
    }
    p = template_prompt(feats)
    assert "fast" in p or "uptempo" in p
    assert "A minor" in p
    assert "swung" in p
    assert "vocals" in p

    assert "clean, balanced" in template_prompt({})


def test_invert_writes_report(tmp_path, monkeypatch):
    from musictrain.config import Config
    from musictrain.invert import invert

    wav = _write_wav(tmp_path / "ref.wav")
    cfg = Config()
    cfg.project_root = tmp_path
    cfg.clap.enabled = False

    # avoid real analysis cost
    import musictrain.invert as invert_mod
    monkeypatch.setattr(invert_mod, "_analyze_features", lambda cfg, p: {})
    report = invert(tmp_path, cfg, wav)
    assert report["template_prompt"] == "a clean, balanced music track"
    assert (tmp_path / "metadata" / "inverted_prompts.json").exists()


# --------------------------------------------------------------------------- #
# 23 — active learning
# --------------------------------------------------------------------------- #
def test_zscore_normalizes():
    from musictrain.active import _zscore

    z = _zscore([0.0, 1.0, 2.0])
    assert abs(float(z.std())) - 1.0 < 1e-6
    assert np.allclose(_zscore([5.0, 5.0, 5.0]), 0.0)


# --------------------------------------------------------------------------- #
# 24 — augment
# --------------------------------------------------------------------------- #
def test_augment_writes_variants(tmp_path):
    from musictrain.config import Config
    from musictrain.augment import augment

    clean = tmp_path / "data" / "clean"
    _write_wav(clean / "song.wav")
    cfg = Config()
    cfg.project_root = tmp_path
    results = augment(tmp_path, cfg, which="clean", ops=["quiet", "noise"])
    assert len(results) == 1
    assert len(results[0]["variants"]) == 2
    for v in results[0]["variants"]:
        assert (tmp_path / v["path"]).exists()
    assert (tmp_path / "metadata" / "augmented.json").exists()


def test_augment_ops_change_signal(tmp_path):
    from musictrain.augment import add_noise, spectral_tilt

    x = np.zeros(4000, dtype=np.float32)
    assert np.abs(add_noise(x, level=0.01)).max() > 0
    tilt = spectral_tilt(np.random.default_rng(1).standard_normal(4000).astype(np.float32), 16000, 3.0)
    assert tilt.shape == (4000,)


# --------------------------------------------------------------------------- #
# 25 — post-segment dedup
# --------------------------------------------------------------------------- #
def test_dedup_segments_finds_copies(tmp_path):
    from musictrain.config import Config
    from musictrain.dedup import dedup_segments

    seg = tmp_path / "data" / "segments"
    a = _write_wav(seg / "a_seg000.wav", freq=220.0)
    _write_wav(seg / "b_seg000.wav", freq=220.0)  # near-identical
    _write_wav(seg / "c_seg000.wav", freq=880.0)  # distinct

    cfg = Config()
    cfg.project_root = tmp_path
    cfg.dedup.exact_only = False
    report = dedup_segments(tmp_path, cfg)
    assert report["scanned"] == 3
    assert report["duplicate_files"] >= 1
    assert (tmp_path / "metadata" / "segment_duplicates.json").exists()


# --------------------------------------------------------------------------- #
# 26 — auto-section labeling
# --------------------------------------------------------------------------- #
def test_role_at():
    from musictrain.sections import _role_at

    structure = {
        "segments": [
            {"start": 0.0, "end": 10.0, "role": "intro"},
            {"start": 10.0, "end": 30.0, "role": "chorus"},
        ]
    }
    assert _role_at(structure, 5.0) == "intro"
    assert _role_at(structure, 20.0) == "chorus"
    assert _role_at(structure, 99.0) is None


def test_auto_sections_patches_labels(tmp_path):
    from musictrain.config import Config
    from musictrain.sections import auto_sections

    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    (meta / "analysis.jsonl").write_text(
        json.dumps(
            {
                "path": "data/clean/song.wav",
                "structure": {
                    "segments": [
                        {"start": 0.0, "end": 10.0, "role": "intro"},
                        {"start": 10.0, "end": 30.0, "role": "chorus"},
                    ]
                },
            }
        )
        + "\n"
    )
    (meta / "segments.json").write_text(
        json.dumps(
            [
                {"path": "data/segments/song_seg000.wav", "song_id": "song",
                 "source": "data/clean/song.wav", "start_time": 2.0},
                {"path": "data/segments/song_seg001.wav", "song_id": "song",
                 "source": "data/clean/song.wav", "start_time": 12.0},
            ]
        )
    )
    (meta / "labels.csv").write_text(
        "source_id,genre\nsong_seg000,trap\nsong_seg001,trap\n"
    )

    cfg = Config()
    cfg.project_root = tmp_path
    report = auto_sections(tmp_path, cfg)
    assert report["labeled"] == 2
    csv_text = (meta / "labels.csv").read_text()
    assert "section_type" in csv_text
    assert "intro" in csv_text and "chorus" in csv_text


# --------------------------------------------------------------------------- #
# 27 — drift
# --------------------------------------------------------------------------- #
def test_drift_psi_and_ks():
    from musictrain.drift import _psi, _cat_shift

    r = np.linspace(80, 100, 100)
    c = np.linspace(80, 100, 100)
    assert _psi(r, c) < 0.01  # identical distributions
    assert _cat_shift(["a", "a", "b"], ["a", "a", "b"]) < 1e-9
    assert _cat_shift(["a", "a", "a"], ["b", "b", "b"]) > 0.5


def test_drift_report_identical(tmp_path):
    from musictrain.config import Config
    from musictrain.drift import drift_report

    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    rows = [
        {"path": "data/train/s1.wav", "bpm": 96.0, "loudness": -14.0,
         "duration": 30.0, "key_confidence": 0.8, "key": "A minor"},
        {"path": "data/train/s2.wav", "bpm": 100.0, "loudness": -13.0,
         "duration": 28.0, "key_confidence": 0.7, "key": "A minor"},
    ]
    (meta / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    cfg = Config()
    cfg.project_root = tmp_path
    report = drift_report(tmp_path, cfg, reference="clean", current="train")
    assert report["current_n"] == 2
    assert report["drifted_features"] == []
    assert (tmp_path / "metadata" / "drift.json").exists()


# --------------------------------------------------------------------------- #
# 28 — curation score
# --------------------------------------------------------------------------- #
def test_curation_score_ranks(tmp_path):
    from musictrain.config import Config
    from musictrain.curation import curation_score

    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    rows = [
        {"path": "data/clean/good.wav", "bpm": 96.0},
        {"path": "data/clean/meh.wav", "bpm": 120.0},
    ]
    (meta / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (meta / "quality.json").write_text(
        json.dumps({"tracks": {"good.wav": {"clap_score": 0.8, "loudness": -14.0}}})
    )
    (meta / "audio_embeddings.json").write_text(
        json.dumps(
            {
                "data/clean/good.wav": {"vec": [1.0, 0.0], "size": 1, "mtime": 1},
                "data/clean/meh.wav": {"vec": [0.0, 1.0], "size": 1, "mtime": 1},
            }
        )
    )

    cfg = Config()
    cfg.project_root = tmp_path
    scored = curation_score(tmp_path, cfg)
    assert len(scored) == 2
    assert all(0 <= r["score"] <= 100 for r in scored)
    # good track has CLAP quality boost -> should outrank meh
    assert scored[0]["path"].endswith("good.wav")
    assert (tmp_path / "metadata" / "curation_scores.json").exists()


# --------------------------------------------------------------------------- #
# 29 — embedding refresh
# --------------------------------------------------------------------------- #
def test_refresh_prunes_and_embeds(tmp_path, monkeypatch):
    from musictrain.config import Config
    from musictrain.embeddings import refresh

    clean = tmp_path / "data" / "clean"
    _write_wav(clean / "keep.wav")
    _write_wav(clean / "new.wav")
    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    # stale entry for a file that no longer exists + a stale entry for keep.wav
    (meta / "audio_embeddings.json").write_text(
        json.dumps(
            {
                "data/clean/keep.wav": {"vec": [1.0, 0.0], "size": 12345, "mtime": 0},
                "data/clean/gone.wav": {"vec": [0.0, 1.0], "size": 1, "mtime": 1},
            }
        )
    )

    import musictrain.embeddings as emb
    monkeypatch.setattr(emb, "embed_audio", lambda cfg, p: np.array([0.5, 0.5], dtype=np.float32))

    cfg = Config()
    cfg.project_root = tmp_path
    report = refresh(tmp_path, cfg)
    assert report["pruned"] == 1  # gone.wav dropped
    assert report["changed"] == 2  # keep.wav (mtime mismatch) + new.wav
    assert report["live"] == 2
    cached = json.loads((meta / "audio_embeddings.json").read_text())
    assert "data/clean/gone.wav" not in cached


# --------------------------------------------------------------------------- #
# 30 — label propagation + leakage check
# --------------------------------------------------------------------------- #
def test_labeled_rows(tmp_path):
    from musictrain.labelprop import _labeled_rows

    meta = tmp_path / "metadata"
    meta.mkdir(exist_ok=True)
    p = meta / "labels.csv"
    p.write_text("source_id,genre,mood\na,trap,dark\nb,trap,dark\n")
    rows = _labeled_rows(tmp_path)
    assert rows["a"]["genre"] == "trap"


def test_leakage_check_finds_cross_split(tmp_path):
    from musictrain.config import Config
    from musictrain.labelprop import leakage_check

    _write_wav(tmp_path / "data" / "train" / "t1.wav", freq=220.0)
    _write_wav(tmp_path / "data" / "val" / "v1.wav", freq=220.0)  # near-dup of t1
    _write_wav(tmp_path / "data" / "val" / "v2.wav", freq=880.0)

    cfg = Config()
    cfg.project_root = tmp_path
    cfg.dedup.exact_only = False
    report = leakage_check(tmp_path, cfg)
    assert report["files_checked"] == 3
    assert report["cross_split_duplicates"] >= 1
    assert (tmp_path / "metadata" / "leakage.json").exists()
