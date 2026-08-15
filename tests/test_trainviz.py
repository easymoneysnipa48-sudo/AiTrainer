"""Unit tests for musictrain.trainviz pure helpers."""
from __future__ import annotations

from musictrain.trainviz import _read_json, _read_jsonl


def test_read_jsonl_parses_and_skips_bad_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\nnot json\n{"b": 2}\n')
    assert _read_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_read_jsonl_missing(tmp_path):
    assert _read_jsonl(tmp_path / "nope.jsonl") == []


def test_read_json_missing(tmp_path):
    assert _read_json(tmp_path / "nope.json") is None


def test_read_json_bad(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{bad")
    assert _read_json(p) is None
