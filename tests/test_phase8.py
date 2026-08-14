"""Tests for advanced batch 2 (#11-#20): sweep, cache, merge, finetune, mining."""
import json
from pathlib import Path

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# sweep — prompt variants, sweeps, ensemble, chain (with fake generators)
# --------------------------------------------------------------------------- #
def test_prompt_variants_returns_distinct():
    from musictrain.sweep import prompt_variants

    v = prompt_variants("melodic trap chorus, 96 BPM, A minor, dark piano, high energy", n=4)
    assert len(v) >= 2
    assert len(set(v)) == len(v)  # deduped
    assert "96 BPM" in v[0]


def _fake_generator(results):
    def gen(cfg, prompt, out_dir=None, seed=None, **kwargs):
        return results.pop(0)
    return gen


def test_run_sweep_picks_best_by_clap(tmp_path):
    from musictrain.config import Config
    from musictrain.sweep import run_sweep

    cfg = Config()
    cfg.project_root = tmp_path
    cfg.clap.enabled = False
    results = [
        {"path": "/tmp/a.wav", "duration": 5.0, "clap": 0.3},
        {"path": "/tmp/b.wav", "duration": 5.0, "clap": 0.7},
    ]

    def gen(cfg, prompt, out_dir=None, seed=None, **kwargs):
        r = results.pop(0)
        return {"path": r["path"], "duration": r["duration"], "cached": False}

    def fake_score(cfg, path, prompt):
        return 0.3 if str(path) == "/tmp/a.wav" else 0.7

    import musictrain.sweep as sweep
    sweep._score_clap = fake_score
    rows, best = run_sweep(cfg, "p", [2.0, 3.0], [1], generator=gen)
    assert len(rows) == 2
    assert best["path"] == "/tmp/b.wav"
    assert (tmp_path / "metadata" / "sweep.json").exists()


def test_run_ensemble_and_chain(tmp_path):
    from musictrain.config import Config
    from musictrain.sweep import chain_generations, run_ensemble

    cfg = Config()
    cfg.project_root = tmp_path
    cfg.clap.enabled = False

    calls = []

    def gen(cfg, prompt, out_dir=None, seed=None, **kwargs):
        calls.append((prompt, seed, kwargs.get("melody_from")))
        return {"path": f"/tmp/o{len(calls)}.wav", "duration": 5.0, "cached": False}

    import musictrain.sweep as sweep
    sweep._score_clap = lambda cfg, path, prompt: 0.5
    rows, best = run_ensemble(cfg, "chorus, 96 BPM, A minor", n=3, generator=gen)
    assert len(rows) >= 2
    assert best["path"]

    chain = chain_generations(cfg, "chain start", steps=3, generator=gen)
    assert len(chain) == 3
    # step 2 and 3 must be melody-conditioned on the previous output
    assert calls[-2][2] is not None and calls[-1][2] is not None


# --------------------------------------------------------------------------- #
# inference — deterministic cache
# --------------------------------------------------------------------------- #
def test_cache_key_deterministic():
    from musictrain.config import Config
    from musictrain.inference import _cache_key

    cfg = Config()
    assert _cache_key(cfg, "prompt x", 42) == _cache_key(cfg, "prompt x", 42)
    assert _cache_key(cfg, "prompt x", 42) != _cache_key(cfg, "prompt x", 43)
    assert _cache_key(cfg, "prompt x", 42) != _cache_key(cfg, "prompt y", 42)


def test_generate_cached_hit_and_miss(tmp_path):
    from musictrain.config import Config
    from musictrain.inference import generate_cached

    cfg = Config()
    cfg.project_root = tmp_path
    wav = tmp_path / "out.wav"
    import soundfile as sf
    import numpy as np

    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 32000)

    generated = {"calls": 0}

    def fake_generate(cfg, prompt, out_dir=None, seed=None, **kwargs):
        generated["calls"] += 1
        return {"path": str(wav), "duration": 5.0, "max_new_tokens": 256}

    import musictrain.inference as inf
    inf.generate = fake_generate

    r1 = generate_cached(cfg, "p", seed=1)
    assert r1["path"] == str(wav) and generated["calls"] == 1
    r2 = generate_cached(cfg, "p", seed=1)  # cache hit — no second call
    assert r2.get("cached") is True and generated["calls"] == 1
    r3 = generate_cached(cfg, "p", seed=2)  # different seed -> miss
    assert generated["calls"] == 2


# --------------------------------------------------------------------------- #
# merge — dummy weight files
# --------------------------------------------------------------------------- #
def test_merge_averages_weights(tmp_path):
    from musictrain.merge import merge

    m1 = tmp_path / "m1"
    m2 = tmp_path / "m2"
    m1.mkdir()
    m2.mkdir()
    # legacy torch .bin format for the dummy test
    import torch

    torch.save({"w": torch.tensor([1.0, 3.0])}, m1 / "pytorch_model.bin")
    torch.save({"w": torch.tensor([3.0, 5.0])}, m2 / "pytorch_model.bin")
    (m1 / "config.json").write_text(json.dumps({"model_type": "musicgen"}))

    out = merge([m1, m2], tmp_path / "merged")
    assert (out / "config.json").exists()
    merged = torch.load(str(out / "pytorch_model.bin"), map_location="cpu")
    np.testing.assert_allclose(merged["w"].numpy(), [2.0, 4.0])


def test_merge_needs_two():
    from musictrain.merge import merge

    with pytest.raises(ValueError):
        merge([Path("/tmp/x")], Path("/tmp/y"))


# --------------------------------------------------------------------------- #
# finetune — data prep + dry run
# --------------------------------------------------------------------------- #
def test_finetune_pairs_and_dry_run(tmp_path):
    from musictrain.config import Config
    from musictrain.finetune import _pairs, train

    clean = tmp_path / "data" / "clean"
    clean.mkdir(parents=True)
    (clean / "song.wav").write_bytes(b"RIFF")
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "manifest.jsonl").write_text(
        json.dumps({"path": "data/clean/song.wav", "description": "dark piano verse"}) + "\n"
    )

    pairs = _pairs(tmp_path)
    assert len(pairs) == 1
    assert "dark piano" in pairs[0][1]

    cfg = Config()
    cfg.project_root = tmp_path
    result = train(cfg, steps=0)
    assert result.get("dry_run") is True


# --------------------------------------------------------------------------- #
# difficulty — negative mining
# --------------------------------------------------------------------------- #
def test_mine_negatives_lowest_clap_first():
    from musictrain.difficulty import mine_negatives

    rows = [
        {"prompt": f"p{i}", "section": "chorus", "clap_score": 0.1 * i,
         "audio_path": f"/tmp/p{i}.wav"}
        for i in range(1, 6)
    ]
    mined = mine_negatives(rows, k=3)
    assert mined
    assert mined[0]["clap_score"] == min(r["clap_score"] for r in rows)
    assert all(m["candidate_for"] == "negative set" for m in mined)
