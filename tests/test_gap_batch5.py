"""Tests for gap batch 5 (#16 retry, #17 fallback, #18 secrets, #19 migration)."""
from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# #16 retry/backoff
# --------------------------------------------------------------------------- #
def test_retry_succeeds_after_transient_failures():
    from musictrain import retry

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("boom")
        return "ok"

    assert retry.retry(flaky, retries=3, base_delay=0) == "ok"
    assert calls["n"] == 3


def test_retry_raises_after_exhausted():
    from musictrain import retry

    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        retry.retry(always_fail, retries=2, base_delay=0)
    assert calls["n"] == 3  # initial + 2 retries


def test_retry_non_transient_not_retried():
    from musictrain import retry

    calls = {"n": 0}

    def fail():
        calls["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        retry.retry(fail, retries=5, retryable=retry.is_transient, base_delay=0)
    assert calls["n"] == 1


def test_is_transient():
    from musictrain import retry

    assert retry.is_transient(TimeoutError("connection timed out"))
    assert retry.is_transient(ConnectionError("connection refused"))
    assert not retry.is_transient(ValueError("bad value"))


# --------------------------------------------------------------------------- #
# #17 GPU fallback chain
# --------------------------------------------------------------------------- #
def test_device_chain_ends_in_cpu():
    from musictrain.inference import device_chain

    assert device_chain("mps")[-1] == "cpu"
    assert device_chain("cuda")[-1] == "cpu"
    assert "mps" in device_chain("mps")


def test_resolve_device_falls_back_to_cpu():
    pytest.importorskip("torch")  # resolve_device probes torch device availability
    from musictrain.inference import resolve_device

    # "bogus" is never available -> must land on cpu (never crash)
    assert resolve_device("bogus") == "cpu"


# --------------------------------------------------------------------------- #
# #18 secrets
# --------------------------------------------------------------------------- #
def test_redact():
    from musictrain.secrets import redact

    assert redact("hf_abc123secretxyz") == "hf_a…txyz"
    assert redact("short") == "*****"


def test_redact_mapping():
    from musictrain.secrets import redact_mapping

    data = {"hf_token": "hf_secret_value_123", "model": "musicgen", "n": 1}
    out = redact_mapping(data)
    assert out["hf_token"] != "hf_secret_value_123"
    assert "…" in out["hf_token"]
    assert out["model"] == "musicgen"
    assert out["n"] == 1


def test_validate_hf_token():
    from musictrain.secrets import validate_hf_token

    assert validate_hf_token("hf_" + "a" * 30)["valid"] is True
    assert validate_hf_token("")["valid"] is False
    assert validate_hf_token("not-an-hf-token")["valid"] is False


def test_validate_webhook():
    from musictrain.secrets import validate_webhook

    assert validate_webhook("https://hooks.slack.com/services/abc")["valid"] is True
    assert validate_webhook("ftp://nope")["valid"] is False
    assert validate_webhook("")["valid"] is False


# --------------------------------------------------------------------------- #
# #19 config schema migration
# --------------------------------------------------------------------------- #
def test_migrate_config_renames_and_backfills(tmp_path):
    from musictrain.config import Config, migrate_config

    cfg_path = tmp_path / "configs" / "default.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text(
        "inference:\n  max_tokens: 128\n  model: facebook/musicgen-small\n"
        "normalize:\n  sr: 22050\n"
    )
    out = migrate_config(cfg_path, backup=True)
    assert "inference.max_tokens -> inference.max_new_tokens" in out["changes"]
    assert "inference.model -> inference.model_name" in out["changes"]
    assert "normalize.sr -> normalize.sample_rate" in out["changes"]
    assert (tmp_path / "configs" / "default.yaml.bak").exists()

    cfg = Config.load(cfg_path)
    assert cfg.inference.max_new_tokens == 128
    assert cfg.inference.model_name == "facebook/musicgen-small"
    assert cfg.normalize.sample_rate == 22050


def test_migrate_config_missing_file_creates_defaults(tmp_path):
    from musictrain.config import Config, migrate_config

    cfg_path = tmp_path / "configs" / "default.yaml"
    out = migrate_config(cfg_path, backup=False)
    assert out["changes"] == []
    assert cfg_path.exists()
    cfg = Config.load(cfg_path)
    assert cfg.inference.max_new_tokens == 256  # default backfilled
