"""Unit tests for the Settings page backing modules (templates + transfer + config)."""
import pytest

from musictrain import transfer
from musictrain import templates
from musictrain.config import Config, SettingsCfg


# --------------------------------------------------------------------------- #
# templates.py
# --------------------------------------------------------------------------- #
def test_model_catalog_is_unique_and_populated():
    assert len(templates.MODELS) >= 3
    ids = [m.model_id for m in templates.MODELS]
    assert len(ids) == len(set(ids))
    assert all(m.model_id.startswith("facebook/") for m in templates.MODELS)


def test_model_lookup():
    m = templates.get_model("facebook/musicgen-small")
    assert m is not None and m.model_id == "facebook/musicgen-small"
    assert templates.find_model_by_name(m.name) is m
    assert templates.get_model("does/not-exist") is None


def test_prompt_templates():
    assert len(templates.PROMPT_TEMPLATES) == 8
    names = templates.prompt_template_names()
    assert "Full-song demo" in names
    t = templates.get_prompt_template("Outro fade")
    assert t.section == "outro" and t.energy == "low"


# --------------------------------------------------------------------------- #
# transfer.py
# --------------------------------------------------------------------------- #
def test_resolve_dir_default(tmp_path):
    p = transfer.resolve_dir("", tmp_path, "data/raw")
    assert p == (tmp_path / "data/raw")


def test_resolve_dir_relative_inside_root(tmp_path):
    p = transfer.resolve_dir("sub/dir", tmp_path, "data/raw")
    assert p == (tmp_path / "sub" / "dir").resolve()


def test_resolve_dir_rejects_external_by_default(tmp_path):
    outside = tmp_path.parent / "elsewhere"
    with pytest.raises(PermissionError):
        transfer.resolve_dir(str(outside), tmp_path, "data/raw", allow_external=False)


def test_resolve_dir_allows_external_when_permitted(tmp_path):
    outside = tmp_path.parent / "elsewhere"
    p = transfer.resolve_dir(str(outside), tmp_path, "data/raw", allow_external=True)
    assert p == outside.resolve()


def test_resolve_dir_expands_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = transfer.resolve_dir("~/music", tmp_path, "data/raw", allow_external=False)
    # ~/music is inside tmp_path (the fake HOME), so it is permitted
    assert p == (tmp_path / "music")


def test_safe_name_blocks_traversal():
    assert transfer.safe_name("../../etc/passwd") == "passwd"
    assert transfer.safe_name("a/b\\c.wav") == "c.wav"
    assert transfer.safe_name("../../etc/") == "etc"


def test_save_upload_and_copy_download(tmp_path):
    dest = transfer.save_upload(b"hello", tmp_path / "up", "../x.txt")
    assert dest.read_bytes() == b"hello"
    assert dest.parent == (tmp_path / "up")

    copied, overwrote = transfer.copy_download(dest, tmp_path / "down")
    assert copied.read_bytes() == b"hello"
    assert not overwrote
    _, overwrote = transfer.copy_download(dest, tmp_path / "down")
    assert overwrote


def test_list_artifacts(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "a.wav").write_bytes(b"x")
    (tmp_path / "outputs" / "nested").mkdir(parents=True)
    (tmp_path / "outputs" / "nested" / "b.wav").write_bytes(b"y")
    arts = transfer.list_artifacts(tmp_path, ["outputs"])
    assert len(arts) == 2


# --------------------------------------------------------------------------- #
# config.py SettingsCfg
# --------------------------------------------------------------------------- #
def test_settings_cfg_defaults():
    s = SettingsCfg()
    assert s.theme == "dark"
    assert s.default_model == "facebook/musicgen-small"
    assert s.allow_external_paths is False


def test_settings_roundtrip(tmp_path):
    cfg = Config()
    cfg.settings.upload_dir = "my/uploads"
    cfg.settings.allow_external_paths = True
    cfg.settings.default_model = "facebook/musicgen-melody"

    p = tmp_path / "configs" / "default.yaml"
    cfg.save(p)
    loaded = Config.load(p)
    assert loaded.settings.upload_dir == "my/uploads"
    assert loaded.settings.allow_external_paths is True
    assert loaded.settings.default_model == "facebook/musicgen-melody"
