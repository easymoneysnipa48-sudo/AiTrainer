"""Unit tests for musictrain.analyteviz pure helpers."""
from __future__ import annotations

import numpy as np

from musictrain.analyteviz import _pca2d, _read_json, _read_jsonl


def test_pca2d_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 16))
    coords = _pca2d(X)
    assert coords.shape == (6, 2)


def test_read_jsonl_and_json(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n')
    assert _read_jsonl(p) == [{"a": 1}, {"b": 2}]
    assert _read_json(tmp_path / "x.json") is None
