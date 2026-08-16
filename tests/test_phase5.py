"""Unit tests for Phase 5 — generation (#33-#39).

Exercises the pure plumbing with fake model/processor objects so no MusicGen,
CLAP, or torch model weights are loaded: sampling presets, target-seconds ->
max_new_tokens, continuation/melody conditioning kwargs, the negative-prompt
retry loop (via a monkeypatched CLAP scorer), JSONL batch items, and repro
manifest capture/diff.
"""
from __future__ import annotations

import json
import types

import pytest

from musictrain import inference as inf
from musictrain import reproduce as repro
from musictrain.config import Config, InferenceCfg


def _cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.project_root = tmp_path
    return cfg


class _FakeInputs(dict):
    def to(self, device):
        for k, v in self.items():
            self[k] = v.to(device)
        return self


def _fake_model():
    torch = pytest.importorskip("torch")

    m = types.SimpleNamespace()
    m.config = types.SimpleNamespace(
        audio_encoder=types.SimpleNamespace(sampling_rate=32000)
    )
    m.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return torch.zeros(1, 1, 32000)  # 1 second of silence

    m.generate = generate.__get__(m)
    return m


def _fake_processor():
    torch = pytest.importorskip("torch")

    class P:
        def __call__(self, text, padding=True, return_tensors="pt"):
            n = len(text)
            return _FakeInputs(
                input_ids=torch.zeros(n, 4, dtype=torch.long),
                attention_mask=torch.ones(n, 4, dtype=torch.long),
            )

    return P()


def _cond_wav(tmp_path, name="cond.wav"):
    import numpy as np
    import soundfile as sf

    p = tmp_path / name
    sf.write(p, np.zeros(32000, dtype=np.float32), 32000)
    return p


# --------------------------------------------------------------------------- #
# #37 sampling presets
# --------------------------------------------------------------------------- #


def test_sampling_kwargs_preset():
    icfg = InferenceCfg()
    kw = inf._sampling_kwargs(icfg, "creative")
    assert kw["temperature"] == 1.2
    assert kw["top_k"] == 400
    assert kw["top_p"] == 0.98
    assert kw["guidance_scale"] == 2.0
    assert kw["do_sample"] is True


def test_sampling_kwargs_default():
    icfg = InferenceCfg()
    kw = inf._sampling_kwargs(icfg, None)
    assert kw["temperature"] == icfg.temperature
    assert kw["top_k"] == icfg.top_k
    assert kw["top_p"] == icfg.top_p
    assert kw["guidance_scale"] == icfg.guidance_scale


def test_sampling_kwargs_unknown_preset_raises():
    with pytest.raises(KeyError):
        inf._sampling_kwargs(InferenceCfg(), "bogus")


# --------------------------------------------------------------------------- #
# #39 target seconds
# --------------------------------------------------------------------------- #


def test_generate_target_seconds(tmp_path):
    cfg = _cfg(tmp_path)
    model, proc = _fake_model(), _fake_processor()
    res = inf.generate(
        cfg, "test", out_dir=tmp_path, name="t1", seed=1,
        processor=proc, model=model, device="cpu",
        target_seconds=2.0, manifest=False,
    )
    assert res["max_new_tokens"] == 100  # 2 s * 50 tokens/s
    assert res["target_seconds"] == 2.0
    assert res["attempts"] == 1
    assert (tmp_path / f"{res['path'].split('/')[-1]}").exists()


def test_target_seconds_minimum_clamp():
    cfg = Config()
    model, proc = _fake_model(), _fake_processor()
    res = inf.generate(
        cfg, "t", out_dir=cfg.project_root / "outputs", name="t2", seed=1,
        processor=proc, model=model, device="cpu",
        target_seconds=0.05, manifest=False,
    )
    assert res["max_new_tokens"] == 8  # clamped to the minimum


# --------------------------------------------------------------------------- #
# #35 continuation / #36 melody
# --------------------------------------------------------------------------- #


def test_generate_continuation_plumbs_input_values(tmp_path):
    cfg = _cfg(tmp_path)
    cond = _cond_wav(tmp_path)
    model, proc = _fake_model(), _fake_processor()
    res = inf.generate(
        cfg, "keep going", out_dir=tmp_path, name="cont", seed=1,
        processor=proc, model=model, device="cpu",
        continue_from=cond, manifest=False,
    )
    assert res["conditioning_kind"] == "continuation"
    assert res["conditioned_on"].endswith("cond.wav")
    iv = model.last_kwargs["input_values"]
    assert tuple(iv.shape) == (1, 1, 32000)  # [batch, channels, samples] at model rate


def test_generate_melody_kind(tmp_path):
    cfg = _cfg(tmp_path)
    cond = _cond_wav(tmp_path)
    model, proc = _fake_model(), _fake_processor()
    res = inf.generate(
        cfg, "follow this", out_dir=tmp_path, name="mel", seed=1,
        processor=proc, model=model, device="cpu",
        melody_from=cond, manifest=False,
    )
    assert res["conditioning_kind"] == "melody"
    assert "input_values" in model.last_kwargs


def test_generate_both_conditioning_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    cond = _cond_wav(tmp_path)
    res = inf.generate(
        cfg, "x", out_dir=tmp_path, name="x", seed=1,
        processor=_fake_processor(), model=_fake_model(), device="cpu",
        continue_from=cond, melody_from=cond, manifest=False,
    )
    assert res == {}


# --------------------------------------------------------------------------- #
# #33 negative prompting with retries
# --------------------------------------------------------------------------- #


def _gen(tmp_path, **kw):
    return inf.generate(
        Config() if "cfg" not in kw else kw.pop("cfg"),
        "test prompt", out_dir=tmp_path, name="neg", seed=1,
        processor=_fake_processor(), model=_fake_model(), device="cpu",
        manifest=False, **kw,
    )


def test_negative_clean_first_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(inf, "_clap_score", lambda cfg, path, text: 0.05)
    res = _gen(tmp_path, negative_prompt="vocals", negative_retries=1)
    assert res["attempts"] == 1
    assert res["negative_violation"] is False
    assert res["negative_clap"] == 0.05


def test_negative_retries_until_clean(tmp_path, monkeypatch):
    scores = iter([0.9, 0.1])
    monkeypatch.setattr(inf, "_clap_score", lambda cfg, path, text: next(scores))
    res = _gen(tmp_path, negative_prompt="vocals", negative_retries=1)
    assert res["attempts"] == 2
    assert res["negative_violation"] is False
    assert res["negative_clap"] == 0.1


def test_negative_no_retries_keeps_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(inf, "_clap_score", lambda cfg, path, text: 0.9)
    res = _gen(tmp_path, negative_prompt="vocals", negative_retries=0)
    assert res["attempts"] == 1
    assert res["negative_violation"] is True


def test_negative_disabled_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(inf, "_clap_score", lambda cfg, path, text: 0.99)
    res = _gen(tmp_path, negative_prompt=None)
    assert res["attempts"] == 1
    assert res["negative_violation"] is None


# --------------------------------------------------------------------------- #
# #34 batch JSONL items
# --------------------------------------------------------------------------- #


def test_generate_batch_dict_items(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(inf, "load_model", lambda icfg: (_fake_processor(), _fake_model(), "cpu"))
    items = [
        {"prompt": "first", "seed": 7, "target_seconds": 1.0},
        {"prompt": "second"},
    ]
    results = inf.generate_batch(cfg, items, out_dir=tmp_path, seed=100)
    assert len(results) == 2
    assert results[0]["seed"] == 7
    assert results[0]["max_new_tokens"] == 50
    assert results[1]["seed"] == 101  # seed + index fallback
    assert results[1]["max_new_tokens"] == cfg.inference.max_new_tokens


def test_generate_batch_plain_lines(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(inf, "load_model", lambda icfg: (_fake_processor(), _fake_model(), "cpu"))
    results = inf.generate_batch(cfg, ["a", "b"], out_dir=tmp_path, seed=5)
    assert [r["seed"] for r in results] == [5, 6]


# --------------------------------------------------------------------------- #
# #38 repro manifest
# --------------------------------------------------------------------------- #


def test_reproduce_capture_and_load(tmp_path):
    cfg = _cfg(tmp_path)
    entry = repro.capture_run(cfg, "inference", extra={"prompt": "hello", "seed": 3})
    assert entry["kind"] == "inference"
    assert entry["prompt"] == "hello"
    assert "git_commit" in entry and "vocab_version" in entry
    entries = repro.load_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["prompt"] == "hello"


def test_reproduce_diff(tmp_path):
    cfg = _cfg(tmp_path)
    a = repro.capture_run(cfg, "inference", extra={"prompt": "x", "seed": 1})
    b = repro.capture_run(cfg, "inference", extra={"prompt": "y", "seed": 2})
    lines = repro.diff(a, b)
    assert any("prompt:" in l for l in lines)
    assert any("seed:" in l for l in lines)
    assert repro.diff(a, a) == ["(no differences)"]


def test_generate_writes_manifest(tmp_path):
    cfg = _cfg(tmp_path)
    model, proc = _fake_model(), _fake_processor()
    inf.generate(
        cfg, "manifest test", out_dir=tmp_path, name="mf", seed=1,
        processor=proc, model=model, device="cpu", manifest=True,
    )
    entries = repro.load_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["kind"] == "inference"
    assert entries[0]["model"] == cfg.inference.model_name
