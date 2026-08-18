"""Tests for cleanup (transient artifact removal) and the pipeline orchestrator
(one-command retrain plan + augment-to-finetune pair wiring)."""
from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from musictrain import augment, cleanup, pipeline
from musictrain.config import Config
from musictrain.finetune import _pairs


def _seed_root(tmp_path) -> None:
    import os
    import time as _time

    (tmp_path / "metadata").mkdir()
    (tmp_path / "outputs" / "lyrics").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "metadata" / "session.json").write_text("{}")
    (tmp_path / "outputs" / "lyrics" / "draft.tmp").write_text("x")
    (tmp_path / "logs" / "musictrain.log.2026-01-01").write_text("old")
    (tmp_path / "metadata" / "eval_results.jsonl").write_text("")
    # distinct mtimes so "keep N newest" is deterministic
    base = _time.time() - 1000
    for i in range(1, 7):
        p = tmp_path / "metadata" / f"eval_results.jsonl.{i}"
        p.write_text("")
        os.utime(p, (base + i * 10, base + i * 10))
    (tmp_path / "metadata" / "user-kept.csv").write_text("keep me")


def test_cleanup_plan_lists_safe_items(tmp_path):
    _seed_root(tmp_path)
    items = cleanup.plan(tmp_path, keep_eval=3)
    paths = [str(p.relative_to(tmp_path)) for p, _ in items]
    assert "metadata/session.json" in paths
    assert "outputs/lyrics/draft.tmp" in paths
    # keep_eval=3 keeps the 3 NEWEST backups (.6/.5/.4); the oldest are removed
    assert "metadata/eval_results.jsonl.1" in paths
    assert "metadata/eval_results.jsonl.2" in paths
    assert "metadata/eval_results.jsonl.3" in paths
    assert "metadata/eval_results.jsonl.4" not in paths
    assert "metadata/eval_results.jsonl.5" not in paths
    assert "metadata/eval_results.jsonl.6" not in paths
    # user data is never touched
    assert "metadata/user-kept.csv" not in paths


def test_cleanup_run_removes_only_targets(tmp_path):
    _seed_root(tmp_path)
    summary = cleanup.run(tmp_path, dry_run=False, keep_eval=3)
    assert summary["deleted"] > 0
    assert not (tmp_path / "metadata" / "session.json").exists()
    assert not (tmp_path / "outputs" / "lyrics" / "draft.tmp").exists()
    assert (tmp_path / "metadata" / "user-kept.csv").exists()
    # the 3 oldest backups are removed, the 3 newest (.4/.5/.6) are kept
    assert not (tmp_path / "metadata" / "eval_results.jsonl.3").exists()
    assert (tmp_path / "metadata" / "eval_results.jsonl.4").exists()


def test_cleanup_dry_run_deletes_nothing(tmp_path):
    _seed_root(tmp_path)
    cleanup.run(tmp_path, dry_run=True)
    assert (tmp_path / "metadata" / "session.json").exists()


def test_pipeline_dry_run_plans_without_executing(tmp_path):
    cfg = Config(project_root=tmp_path)
    out = pipeline.run(tmp_path, cfg, steps=["audio", "lyrics"], dry_run=True)
    assert out == {"dry_run": True}
    assert not (tmp_path / "metadata").exists()


def test_augment_feeds_finetune_pairs(tmp_path):
    (tmp_path / "data" / "segments").mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    sr = 22050
    rng = np.random.default_rng(1)
    sf.write(tmp_path / "data" / "segments" / "song_seg000.wav",
             rng.standard_normal(sr).astype("float32") * 0.1, sr)
    (tmp_path / "metadata" / "labels.csv").write_text(
        'source_id,description\nsong_seg000,"verse, 140 BPM, dark trap"\n')

    cfg = Config(project_root=tmp_path)
    aug = augment.augment(tmp_path, cfg, which="segments", variants=2, seed=0)
    n_variants = sum(len(r["variants"]) for r in aug)
    assert n_variants == 12  # 6 ops x 2 variants

    pairs = _pairs(tmp_path)
    assert len(pairs) == 13  # 1 segment + 12 augmented
    aug_pairs = [p for p in pairs if "__" in p[0].name]
    assert all("140 BPM" in d for _, d in aug_pairs)
    assert all("semitones" in d or "rate=" in d or "level=" in d or "gain=" in d or "tilt_db=" in d
               for _, d in aug_pairs)


def test_append_records_merge_shape(tmp_path):
    from musictrain import lyricdataset as LD

    records = [
        {"artist": "Future", "title": "One", "lines": ["a", "b", "c", "d"],
         "mood": "dark", "topic": "struggle", "source": "scrape"},
    ]
    LD.append_records(tmp_path, records)
    row = json.loads(next(open(tmp_path / "lyrics" / "future" / "songs.jsonl")))
    assert row["artist_id"] == "future"
    assert row["n_lines"] == 4
