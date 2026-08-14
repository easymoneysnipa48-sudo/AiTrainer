"""Command-line interface for musictrain."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, console
from .config import Config
from .paths import ensure_layout


def _build_config(args) -> Config:
    cfg_path = Path(args.config) if getattr(args, "config", None) else None
    cfg = Config.load(cfg_path) if cfg_path else Config()
    cfg.project_root = Path(args.root).resolve()
    return cfg


def cmd_init(args) -> int:
    root = Path(args.root).resolve()
    ensure_layout(root)
    cfg = Config(project_root=root)
    data = cfg.to_dict()
    data.pop("project_root", None)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    import yaml

    (root / "configs" / "default.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False)
    )

    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            ".venv/\n__pycache__/\n*.pyc\n.DS_Store\n"
            "data/raw/\ndata/clean/\ndata/segments/\ndata/train/\ndata/val/\ndata/test/\n"
            "checkpoints/\noutputs/*.wav\nlogs/\n.env\n"
        )
    console.ok(f"Initialized project at {root}")
    console.info("Next: add audio to data/raw, then run `musictrain normalize`.")
    return 0


def cmd_config(args) -> int:
    import yaml

    cfg = _build_config(args)
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return 0


def cmd_normalize(args) -> int:
    from .audio.normalize import normalize

    cfg = _build_config(args)
    ensure_layout(cfg.project_root)
    normalize(cfg.project_root, cfg, force=args.force, dry_run=args.dry_run, limit=args.limit)
    return 0


def cmd_inventory(args) -> int:
    from .audio.inventory import inventory

    cfg = _build_config(args)
    inventory(cfg.project_root, which=args.dir, sha256=args.sha256)
    return 0


def cmd_features(args) -> int:
    from .metadata import extract, validate

    cfg = _build_config(args)
    labels = Path(args.labels) if args.labels else None
    records = extract(
        cfg.project_root, cfg, which=args.dir, labels_path=labels, limit=args.limit
    )
    if records:
        from .experiments import log_dataset

        log_dataset(cfg, records)
    if args.validate and records:
        issues = validate(records)
        if issues:
            console.warn(f"{len(issues)} metadata issue(s):")
            for i in issues:
                console.warn("  - " + i)
        else:
            console.ok("Metadata validation passed.")
    return 0


def cmd_segment(args) -> int:
    from .audio.segment import segment

    cfg = _build_config(args)
    if args.downbeat:
        cfg.segment.downbeat_aligned = True
    if args.overlap is not None:
        cfg.segment.overlap_seconds = args.overlap
    if args.fade is not None:
        cfg.segment.fade_seconds = args.fade
    segment(cfg.project_root, cfg, force=args.force, dry_run=args.dry_run)
    return 0


def cmd_split(args) -> int:
    from .split import split

    cfg = _build_config(args)
    if args.stratify is not None:
        cfg.split.stratify = args.stratify
    if args.k_folds is not None:
        cfg.split.k_folds = args.k_folds
    split(cfg.project_root, cfg, dry_run=args.dry_run)
    return 0


def cmd_export(args) -> int:
    from .export import export

    cfg = _build_config(args)
    export(cfg.project_root, cfg, which=args.which or "", format_=args.format or "")
    return 0


def cmd_infer(args) -> int:
    from .inference import generate, generate_batch

    cfg = _build_config(args)
    icfg = cfg.inference
    if args.model:
        icfg.model_name = args.model
    if args.device:
        icfg.device = args.device
    if args.dtype:
        icfg.torch_dtype = args.dtype
    if args.guidance is not None:
        icfg.guidance_scale = args.guidance
    if args.max_new_tokens is not None:
        icfg.max_new_tokens = args.max_new_tokens

    out_dir = Path(args.out) if args.out else cfg.project_root / "outputs"

    if args.prompts_file:
        prompts = [
            ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()
        ]
        if not prompts:
            console.error("Prompts file is empty.")
            return 1
        results = generate_batch(cfg, prompts, out_dir=out_dir, seed=args.seed)
        from .experiments import log_inference

        for r in results:
            log_inference(cfg, r)
    else:
        if not args.prompt:
            console.error("Provide --prompt or --prompts-file.")
            return 1
        result = generate(cfg, args.prompt, out_dir=out_dir, seed=args.seed)
        from .experiments import log_inference

        log_inference(cfg, result)
    return 0


def cmd_labels(args) -> int:
    from .labels import check, hierarchy_notes, scaffold

    cfg = _build_config(args)
    path = Path(args.file) if args.file else cfg.project_root / "metadata" / "labels.csv"
    if args.check:
        issues = check(path)
        if issues:
            console.warn(f"{len(issues)} label issue(s):")
            for i in issues:
                console.warn("  - " + i)
            return 1
        console.ok("Labels CSV is valid.")
        return 0
    if args.notes:
        notes = hierarchy_notes(path)
        if notes:
            console.info(f"{len(notes)} hierarchy suggestion(s):")
            for n in notes:
                console.info("  - " + n)
        else:
            console.ok("No broad parent terms used — hierarchy is clean.")
        return 0
    scaffold(cfg.project_root)
    return 0


def cmd_vocab(args) -> int:
    from .labels import VOCAB_VERSION, VOCAB_VERSION_NOTES
    from .vocab import migrate, render_tree

    cfg = _build_config(args)
    if args.migrate:
        labels_path = (
            Path(args.labels)
            if args.labels
            else cfg.project_root / "metadata" / "labels.csv"
        )
        migrate(cfg.project_root, labels_path, Path(args.migrate), backup=not args.no_backup)
        return 0
    if args.version:
        console.info(f"vocabulary version: v{VOCAB_VERSION}")
        console.info(VOCAB_VERSION_NOTES)
        return 0
    print(render_tree())
    return 0


def cmd_agree(args) -> int:
    from .agreement import agreement

    cfg = _build_config(args)
    agreement(Path(args.a).resolve(), Path(args.b).resolve(), cfg.project_root)
    return 0


def cmd_suggest(args) -> int:
    from .suggest import suggest

    cfg = _build_config(args)
    report = suggest(
        cfg.project_root, cfg, Path(args.query).resolve(), top_k=args.top, which=args.dir
    )
    return 0 if report else 1


def cmd_prompt(args) -> int:
    from .promptbuilder import build_prompt

    prompt = build_prompt(
        section=args.section,
        genre=args.genre,
        mood=args.mood,
        instruments=args.instruments,
        bpm=args.bpm,
        key=args.key,
        energy=args.energy,
        role=args.role,
    )
    if not prompt:
        console.error(
            "Nothing to build — provide at least one of --section/--genre/"
            "--mood/--instruments/--bpm/--key."
        )
        return 1
    print(prompt)
    return 0


def cmd_check(args) -> int:
    from .evaluate import check, check_dir

    from .experiments import log_eval

    cfg = _build_config(args)
    path = Path(args.path).resolve()
    if path.is_dir():
        reports = check_dir(cfg, path, target_bpm=args.bpm, fix=args.fix)
        for r in reports:
            log_eval(cfg, r)
    else:
        report = check(cfg, path, target_bpm=args.bpm, fix=args.fix, out_dir=path.parent)
        log_eval(cfg, report)
    return 0


def cmd_package(args) -> int:
    from .package import package

    cfg = _build_config(args)
    package(
        cfg.project_root,
        host=args.host,
        which=args.dir,
        python=args.python,
        tarball=args.tarball,
    )
    return 0


def cmd_evalset(args) -> int:
    from .evalset import build

    cfg = _build_config(args)
    build(cfg.project_root, force=args.force)
    return 0


def cmd_eval(args) -> int:
    from .evalset import run_eval

    cfg = _build_config(args)
    if args.no_clap:
        cfg.clap.enabled = False
    run_eval(cfg, limit=args.limit, check_bpm=not args.no_check, section=args.section, seeds=args.seeds)
    return 0


def cmd_score(args) -> int:
    from .similarity import score

    cfg = _build_config(args)
    value = score(cfg, Path(args.path).resolve(), args.text)
    if value is None:
        console.error("CLAP scoring is disabled (clap.enabled=false).")
        return 1
    console.ok(f"CLAP similarity: {value:.4f}  <-  {args.path}")
    return 0


def cmd_report(args) -> int:
    from .report import export

    cfg = _build_config(args)
    export(cfg)
    return 0


def cmd_quality(args) -> int:
    from .audio.quality import quality

    cfg = _build_config(args)
    quality(cfg.project_root, cfg, which=args.dir, limit=args.limit)
    return 0


def cmd_loudnorm(args) -> int:
    from .loudnorm import loudnorm

    cfg = _build_config(args)
    loudnorm(
        cfg.project_root, cfg, which=args.dir,
        target_lufs=args.target, force=args.force, dry_run=args.dry_run,
    )
    return 0


def cmd_dedup(args) -> int:
    from .dedup import find_duplicates

    cfg = _build_config(args)
    if args.move:
        cfg.dedup.action = "move"
    find_duplicates(cfg.project_root, cfg, which=args.dir)
    return 0


def cmd_similar(args) -> int:
    from .embeddings import nearest

    cfg = _build_config(args)
    hits = nearest(
        cfg.project_root, cfg, Path(args.query).resolve(), which=args.dir, top_k=args.top
    )
    if not hits:
        console.error("No embeddings available — check data/<dir> or run `musictrain similar` again.")
        return 1
    for rel, sim in hits:
        console.ok(f"{sim:.4f}  {rel}")
    return 0


def cmd_autolabel(args) -> int:
    from .autolabel import autolabel

    cfg = _build_config(args)
    autolabel(cfg.project_root, cfg, which=args.dir, limit=args.limit)
    return 0


def cmd_corpus(args) -> int:
    from .corpus import corpus

    cfg = _build_config(args)
    corpus(cfg.project_root, cfg, which=args.dir)
    return 0


def cmd_ood(args) -> int:
    from .ood import curate_ood

    cfg = _build_config(args)
    if args.move:
        cfg.ood.action = "move"
    curate_ood(cfg.project_root, cfg, which=args.dir)
    return 0


def cmd_stems(args) -> int:
    from .stems import separate_stems

    cfg = _build_config(args)
    if args.model:
        cfg.stems.model = args.model
    if args.two_stems:
        cfg.stems.two_stems = True
    separate_stems(cfg.project_root, cfg, which=args.dir, limit=args.limit)
    return 0


def cmd_analyze(args) -> int:
    from .audio.analysis import analyze

    cfg = _build_config(args)
    path = Path(args.path).resolve() if args.path else None
    analyze(
        cfg.project_root, cfg, which=args.dir, limit=args.limit, path=path
    )
    return 0


def cmd_ui(args) -> int:
    from .experiments import launch_ui

    cfg = _build_config(args)
    return launch_ui(cfg, args.port)


def cmd_dashboard(args) -> int:
    import subprocess

    dash = Path(__file__).parent / "dashboard.py"
    root = Path(args.root).resolve()
    console.info(f"Launching dashboard at http://localhost:{args.port}")
    return subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", str(dash),
            "--server.port", str(args.port),
            "--browser.gatherUsageStats", "false",
        ],
        cwd=root,
    ).returncode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="musictrain",
        description="MusicGen fine-tuning control-plane toolkit for Apple Silicon.",
    )
    p.add_argument("--version", action="version", version=f"musictrain {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--root", default=".", help="Project root (default: cwd)")
        sp.add_argument("--config", default=None, help="Path to YAML config")

    # init
    sp = sub.add_parser("init", help="Create project layout + default config")
    sp.add_argument("--root", default=".")
    sp.set_defaults(func=cmd_init)

    # config
    sp = sub.add_parser("config", help="Print the effective configuration")
    add_common(sp)
    sp.set_defaults(func=cmd_config)

    # normalize
    sp = sub.add_parser("normalize", help="FFmpeg-normalize data/raw -> data/clean")
    add_common(sp)
    sp.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    sp.add_argument("--dry-run", action="store_true", help="Print actions without running")
    sp.add_argument("--limit", type=int, default=0, help="Only process first N files")
    sp.set_defaults(func=cmd_normalize)

    # inventory
    sp = sub.add_parser("inventory", help="Validate audio and write inventory JSON")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to scan")
    sp.add_argument("--sha256", action="store_true", help="Compute SHA-256 hashes")
    sp.set_defaults(func=cmd_inventory)

    # features
    sp = sub.add_parser("features", help="Extract BPM/key/loudness metadata")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to process")
    sp.add_argument("--labels", default=None, help="CSV/JSON of manual labels")
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--validate", action="store_true", help="Validate the resulting manifest")
    sp.set_defaults(func=cmd_features)

    # segment
    sp = sub.add_parser("segment", help="Split audio into 30s (bar-aligned) examples")
    add_common(sp)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--downbeat", action="store_true", help="Cut on detected downbeats (#21)")
    sp.add_argument("--overlap", type=float, default=None, help="Overlap seconds between segments (#24)")
    sp.add_argument("--fade", type=float, default=None, help="Fade in/out seconds at cut boundaries (#25)")
    sp.set_defaults(func=cmd_segment)

    # split
    sp = sub.add_parser("split", help="Train/val/test split by song")
    add_common(sp)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--stratify", default=None, help="Stratify by key|bpm|genre|mood (#23)")
    sp.add_argument("--k-folds", type=int, default=None, help="N-fold cross-validation instead of train/val/test (#22)")
    sp.set_defaults(func=cmd_split)

    # export
    sp = sub.add_parser("export", help="Export split corpus to HF datasets (arrow/jsonl/csv) (#26)")
    add_common(sp)
    sp.add_argument("--which", default="", choices=["train", "val", "test", "all"], help="Split to export (default: cfg)")
    sp.add_argument("--format", default="", choices=["arrow", "jsonl", "csv"], help="Output format (default: cfg)")
    sp.set_defaults(func=cmd_export)

    # infer
    sp = sub.add_parser("infer", help="Generate audio with MusicGen on MPS")
    add_common(sp)
    sp.add_argument("--prompt", default=None)
    sp.add_argument("--prompts-file", default=None)
    sp.add_argument("--model", default=None, help="e.g. facebook/musicgen-medium")
    sp.add_argument("--device", default=None, choices=["mps", "cpu", "cuda", "auto"])
    sp.add_argument("--dtype", default=None, choices=["float32", "float16"])
    sp.add_argument("--guidance", type=float, default=None)
    sp.add_argument("--max-new-tokens", type=int, default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--out", default=None, help="Output directory")
    sp.set_defaults(func=cmd_infer)

    # labels
    sp = sub.add_parser("labels", help="Scaffold or validate the labels CSV")
    add_common(sp)
    sp.add_argument("--check", action="store_true", help="Validate instead of scaffolding")
    sp.add_argument("--notes", action="store_true", help="Show broad-parent hierarchy suggestions")
    sp.add_argument("--file", default=None, help="Path to labels CSV (default metadata/labels.csv)")
    sp.set_defaults(func=cmd_labels)

    # vocab
    sp = sub.add_parser(
        "vocab", help="Show the hierarchical vocabulary tree or migrate terms (#27, #32)"
    )
    add_common(sp)
    sp.add_argument("--version", action="store_true", help="Show the vocabulary version")
    sp.add_argument(
        "--migrate", default=None, metavar="RENAME_MAP.json",
        help="Apply a term rename map to the labels CSV (#32)",
    )
    sp.add_argument(
        "--labels", default=None,
        help="Labels CSV to migrate (default metadata/labels.csv)",
    )
    sp.add_argument(
        "--no-backup", action="store_true",
        help="Skip the .bak backup before migrating",
    )
    sp.set_defaults(func=cmd_vocab)

    # agree
    sp = sub.add_parser(
        "agree", help="Inter-annotator agreement between two label files (#29)"
    )
    add_common(sp)
    sp.add_argument("--a", required=True, help="Annotator A labels CSV")
    sp.add_argument("--b", required=True, help="Annotator B labels CSV")
    sp.set_defaults(func=cmd_agree)

    # suggest
    sp = sub.add_parser(
        "suggest", help="Auto-suggest labels for a track via CLAP (#31)"
    )
    add_common(sp)
    sp.add_argument("--query", required=True, help="Query audio file")
    sp.add_argument("--dir", default="clean", help="data/<dir> for the neighbor index")
    sp.add_argument("--top", type=int, default=5, help="Number of neighbors")
    sp.set_defaults(func=cmd_suggest)

    # prompt
    sp = sub.add_parser(
        "prompt", help="Assemble a generation prompt from vocabulary selections (#30)"
    )
    add_common(sp)
    sp.add_argument("--section", default=None)
    sp.add_argument("--genre", default=None)
    sp.add_argument("--mood", action="append", default=None)
    sp.add_argument("--instruments", action="append", default=None)
    sp.add_argument("--bpm", type=float, default=None)
    sp.add_argument("--key", default=None)
    sp.add_argument("--energy", type=float, default=None)
    sp.add_argument("--role", default=None)
    sp.set_defaults(func=cmd_prompt)

    # check
    sp = sub.add_parser("check", help="Detect BPM drift of generated audio")
    add_common(sp)
    sp.add_argument("--path", required=True, help="WAV file or directory")
    sp.add_argument("--bpm", type=float, default=None, help="Target BPM")
    sp.add_argument("--fix", action="store_true", help="Time-stretch within tolerance")
    sp.set_defaults(func=cmd_check)

    # package
    sp = sub.add_parser("package", help="Prepare for Ubuntu transfer")
    add_common(sp)
    sp.add_argument("--host", default="", help="user@ubuntu-workstation")
    sp.add_argument("--dir", default="clean", help="data/<dir> to checksum")
    sp.add_argument("--python", default=".venv/bin/python")
    sp.add_argument("--tarball", action="store_true")
    sp.set_defaults(func=cmd_package)

    # evalset
    sp = sub.add_parser("evalset", help="Generate the fixed evaluation prompt set")
    add_common(sp)
    sp.add_argument("--force", action="store_true", help="Overwrite existing set")
    sp.set_defaults(func=cmd_evalset)

    # eval
    sp = sub.add_parser("eval", help="Run batch inference + BPM checks over the eval set")
    add_common(sp)
    sp.add_argument("--limit", type=int, default=0, help="Only run the first N prompts")
    sp.add_argument("--section", default=None, help="Only run prompts for this section (comma-separated for multiple)")
    sp.add_argument("--seeds", type=int, default=1, help="Seeds per prompt (e.g. 3) for majority verdicts")
    sp.add_argument("--no-check", action="store_true", help="Skip BPM post-check")
    sp.add_argument("--no-clap", action="store_true", help="Skip CLAP prompt-adherence scoring")
    sp.set_defaults(func=cmd_eval)

    # report
    sp = sub.add_parser("report", help="Export eval results to CSV + HTML")
    add_common(sp)
    sp.set_defaults(func=cmd_report)

    # quality
    sp = sub.add_parser("quality", help="Score audio quality (bitrate/clipping/silence)")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to score")
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_quality)

    # loudnorm
    sp = sub.add_parser("loudnorm", help="Normalize loudness to a target LUFS")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to normalize")
    sp.add_argument("--target", type=float, default=-14.0, help="Target integrated LUFS")
    sp.add_argument("--force", action="store_true", help="Overwrite files in place")
    sp.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    sp.set_defaults(func=cmd_loudnorm)

    # dedup
    sp = sub.add_parser("dedup", help="Find exact + near-duplicate audio")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to scan")
    sp.add_argument("--move", action="store_true", help="Move non-canonical copies to data/dupes/")
    sp.set_defaults(func=cmd_dedup)

    # similar
    sp = sub.add_parser("similar", help="Find tracks similar to a query file")
    add_common(sp)
    sp.add_argument("--query", required=True, help="Query audio file")
    sp.add_argument("--dir", default="clean", help="data/<dir> to search")
    sp.add_argument("--top", type=int, default=10, help="Number of results")
    sp.set_defaults(func=cmd_similar)

    # autolabel
    sp = sub.add_parser("autolabel", help="Suggest genre/mood/instrument tags via CLAP")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to label")
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_autolabel)

    # corpus
    sp = sub.add_parser("corpus", help="Report BPM/key/tag coverage statistics")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> (for messaging)")
    sp.set_defaults(func=cmd_corpus)

    # ood
    sp = sub.add_parser("ood", help="Flag off-distribution tracks")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> (for messaging)")
    sp.add_argument("--move", action="store_true", help="Move flagged tracks to data/ood/")
    sp.set_defaults(func=cmd_ood)

    # stems
    sp = sub.add_parser("stems", help="Separate stems with Demucs (vocals/drums/bass/other)")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to separate")
    sp.add_argument("--model", default=None, help="htdemucs | htdemucs_ft | htdemucs_6s")
    sp.add_argument("--two-stems", action="store_true", help="Vocals + accompaniment only")
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_stems)

    # analyze
    sp = sub.add_parser(
        "analyze",
        help="Deep audio analysis (chords, beat grid, key confidence, structure)",
    )
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to scan (ignored with --path)")
    sp.add_argument("--path", default=None, help="Analyze a single audio file")
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_analyze)

    # score
    sp = sub.add_parser("score", help="Score audio-text similarity with CLAP")
    add_common(sp)
    sp.add_argument("--path", required=True, help="WAV file to score")
    sp.add_argument("--text", required=True, help="Prompt text to compare against")
    sp.set_defaults(func=cmd_score)

    # ui (MLflow)
    sp = sub.add_parser("ui", help="Launch the MLflow tracking UI")
    add_common(sp)
    sp.add_argument("--port", type=int, default=5000)
    sp.set_defaults(func=cmd_ui)

    # dashboard
    sp = sub.add_parser("dashboard", help="Launch the Streamlit dashboard")
    add_common(sp)
    sp.add_argument("--port", type=int, default=8501)
    sp.set_defaults(func=cmd_dashboard)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.error("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
