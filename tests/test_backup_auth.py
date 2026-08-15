import json
import os
import tempfile
from pathlib import Path

import pytest

from musictrain import auth, backup
from musictrain.config import Config


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "eval_results.jsonl").write_text('{"a": 1}\n')
    (tmp_path / "metadata" / "leaderboard.json").write_text("[]\n")
    (tmp_path / "config.yaml").write_text("project_root: .\n")
    return Config(project_root=tmp_path)


def test_snapshot_and_list(cfg):
    out = backup.snapshot(cfg, label="test", include_mlflow=False)
    assert out["n_files"] >= 3
    assert (cfg.project_root / "backups").exists()

    rows = backup.list_backups(cfg)
    assert len(rows) == 1
    assert rows[0]["name"].endswith(".tar.gz")


def test_restore_roundtrip(cfg):
    out = backup.snapshot(cfg, label="rt", include_mlflow=False)
    # wipe a metadata file, then restore it back
    (cfg.project_root / "metadata" / "leaderboard.json").unlink()
    res = backup.restore(cfg, out["archive"], force=True)
    assert res["restored"] is True
    assert (cfg.project_root / "metadata" / "leaderboard.json").exists()


def test_restore_missing_archive(cfg):
    res = backup.restore(cfg, "nope.tar.gz")
    assert res["error"] == "not_found"


def test_restore_refuses_unsafe_member(tmp_path):
    # archive containing a path that escapes the project root
    import tarfile

    cfg = Config(project_root=tmp_path)
    (tmp_path / "metadata").mkdir()
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        import io
        data = b"x"
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    res = backup.restore(cfg, str(evil))
    assert res["error"] == "unsafe_member"


def test_backup_run_unknown_task(cfg):
    res = backup.run(cfg, "bogus")
    assert res["error"]


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #
def test_hash_and_verify():
    stored = auth.hash_password("secret")
    assert "$" in stored
    assert auth.verify_password("secret", stored)
    assert not auth.verify_password("wrong", stored)


def test_verify_malformed():
    assert not auth.verify_password("x", "nodelimiter")


def test_auth_not_configured_by_default(monkeypatch):
    monkeypatch.delenv("MUSICTRAIN_PASSWORD", raising=False)
    monkeypatch.delenv("MUSICTRAIN_USERS", raising=False)
    monkeypatch.delenv("MUSICTRAIN_OAUTH_TOKENS", raising=False)
    assert auth.is_configured() is False


def test_auth_password_env(monkeypatch):
    monkeypatch.setenv("MUSICTRAIN_PASSWORD", "hunter2")
    assert auth.is_configured() is True
    assert auth.authenticate("default", "hunter2")
    assert not auth.authenticate("default", "nope")


def test_auth_oauth_token(monkeypatch):
    monkeypatch.setenv("MUSICTRAIN_OAUTH_TOKENS", json.dumps(["tok123"]))
    assert auth.authenticate("", "", token="tok123")
    assert not auth.authenticate("", "", token="bad")
