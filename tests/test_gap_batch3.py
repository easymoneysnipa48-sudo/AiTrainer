"""Tests for gap batch 3 (#11 persistent job queue, #13 cache warm-up)."""
from __future__ import annotations

import json
import time

from musictrain.config import Config


# --------------------------------------------------------------------------- #
# #11 persistent job queue
# --------------------------------------------------------------------------- #
def test_job_queue_persists_and_restores(tmp_path):
    from musictrain.server import JobQueue

    q = JobQueue(tmp_path)

    def job(progress=None, cancel=None):
        progress(1, 1)
        return {"ok": True}

    jid = q.submit(job)
    for _ in range(100):
        if q.status(jid) == "done":
            break
        time.sleep(0.02)
    assert q.status(jid) == "done"
    assert (tmp_path / "metadata" / "jobs" / f"{jid}.json").exists()

    # a fresh queue (simulating a restart) restores the finished job
    q2 = JobQueue(tmp_path)
    restored = q2.get(jid)
    assert restored is not None
    assert restored["status"] == "done"
    assert restored["result"] == {"ok": True}


def test_job_queue_marks_inflight_interrupted(tmp_path):
    from musictrain.server import JobQueue

    jobs = tmp_path / "metadata" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "abc123.json").write_text(
        json.dumps({"id": "abc123", "status": "running", "progress": 0.5,
                    "message": "", "result": None, "error": None})
    )
    q = JobQueue(tmp_path)
    j = q.get("abc123")
    assert j is not None and j["status"] == "interrupted"
    assert "restarted" in j["error"]


def test_job_queue_inmemory_default():
    from musictrain.server import JobQueue

    q = JobQueue()
    assert q._dir is None  # no persistence by default (back-compat)


# --------------------------------------------------------------------------- #
# #13 HF cache warm-up
# --------------------------------------------------------------------------- #
def test_warm_cache_degrades_without_hf(monkeypatch):
    import builtins

    import musictrain.cache_warm as cw

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("no huggingface_hub in test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = Config()
    out = cw.warm(cfg)
    assert out["warmed"] is False
    assert "huggingface_hub" in out["note"]


def test_cached_bytes_none_without_hf(monkeypatch):
    import builtins

    import musictrain.cache_warm as cw

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("no huggingface_hub in test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert cw._cached_bytes("facebook/musicgen-small") is None
