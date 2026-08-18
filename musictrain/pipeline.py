"""One-command retrain pipeline.

Ties the audio chain (features -> segment -> split -> labels -> augment ->
fine-tune) and the lyric chain (split -> train-files -> optional train-lyrics)
into a single deterministic run, plus an optional eval pass. Every step is a
thin wrapper over the existing module functions, so behavior matches the CLI.

Idempotent by default: segments that already exist are skipped (unless
``force_segment``), and ``augment`` re-writes the same variant filenames.
Use ``dry_run`` to print the exact plan without executing.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from . import console
from .config import Config


def _step(name: str) -> None:
    console.title(f"▶ {name}")
    console.line()


def run(
    root: Path,
    cfg: Config,
    steps: Sequence[str] = ("audio", "lyrics"),
    force_segment: bool = False,
    augment_enabled: bool = True,
    finetune_epochs: int = 2,
    lyrics_steps: int = 0,
    limit: int = 0,
    dry_run: bool = False,
) -> dict:
    """Run the requested pipeline stages. Returns a per-stage result dict."""
    steps = [s.strip().lower() for s in steps if s and s.strip()]
    if not steps:
        console.error("pipeline needs at least one step from: audio, lyrics, eval")
        return {}

    t0 = time.time()
    out: dict = {}

    if dry_run:
        console.step("Pipeline plan (dry run):")
        if "audio" in steps:
            console.info("  audio: extract features -> segment (skip existing) -> split -> beatlabels --force -> augment segments x3 -> finetune")
        if "lyrics" in steps:
            console.info(f"  lyrics: split -> train-files -> train-lyrics ({lyrics_steps} steps)" if lyrics_steps
                         else "  lyrics: split -> train-files (training skipped; pass --lyrics-steps N)")
        if "eval" in steps:
            console.info("  eval: run eval suite (CLAP + BPM checks)")
        return {"dry_run": True}

    # ------------------------------------------------------------------ audio
    if "audio" in steps:
        _step("Audio chain: features")
        from .metadata import extract

        records = extract(root, cfg, which="clean", limit=limit)
        out["features"] = len(records)

        _step("Audio chain: segment")
        from .audio.segment import segment

        segment(root, cfg, force=force_segment, dry_run=False)

        _step("Audio chain: split")
        from .split import split

        assign = split(root, cfg)
        out["split"] = {k: len(v) for k, v in assign.items()} if assign else {}

        _step("Audio chain: labels")
        from .beatlabels import generate_labels

        n = generate_labels(root, force=True)
        out["labels"] = n or 0

        if augment_enabled:
            _step("Audio chain: augment (segments x3 variants)")
            from .augment import augment

            aug = augment(root, cfg, which="segments", variants=3, seed=0, limit=limit)
            out["augmented"] = sum(len(r["variants"]) for r in aug)

        _step("Audio chain: fine-tune")
        from .finetune import train

        train(
            cfg,
            steps=finetune_epochs,
            lr=1e-4,
            batch_size=1,
            limit=limit,
            out_dir=root / "adapters",
        )
        out["finetune_epochs"] = finetune_epochs

    # ----------------------------------------------------------------- lyrics
    if "lyrics" in steps:
        _step("Lyrics chain: split + instruction files")
        from . import lyricdataset

        try:
            counts = lyricdataset.split_dataset(root, seed=42)
            out["lyrics_split"] = counts
        except FileNotFoundError:
            console.warn("No lyric dataset yet — run `musictrain lyricscrape` or `synthcorpus` first.")
        try:
            out["lyrics_instructions"] = lyricdataset.write_training_files(root)
        except FileNotFoundError:
            console.warn("No lyric training files — skipping train-files.")

        if lyrics_steps > 0:
            _step(f"Lyrics chain: train-lyrics ({lyrics_steps} steps)")
            from .trainlyrics import train as train_lyrics

            res = train_lyrics(root, steps=lyrics_steps, limit=limit)
            if res:
                out["lyrics_run"] = res.get("run_dir", "")
                console.ok(f"Lyrics adapter -> {res.get('run_dir', '?')} "
                           f"(set MUSICTRAIN_LLM_MODEL_PATH to use it)")

    # ------------------------------------------------------------------- eval
    if "eval" in steps:
        _step("Eval suite")
        from .evalset import run_eval

        results = run_eval(cfg, limit=limit)
        out["eval_runs"] = len(results) if results else 0

    out["seconds"] = round(time.time() - t0, 1)
    console.ok(f"Pipeline finished in {out['seconds']}s")
    return out
