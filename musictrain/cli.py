"""Command-line interface for musictrain."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, console
from .config import Config
from .logging import get_logger, setup as setup_logging
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
    if getattr(args, "migrate", False):
        from .config import migrate_config

        target = Path(args.config) if getattr(args, "config", None) else (
            cfg.project_root / "configs" / "default.yaml"
        )
        out = migrate_config(target, backup=not args.no_backup)
        if out.get("error"):
            console.error(out["error"])
            return 1
        console.ok(f"Migrated config -> {target}")
        for change in out.get("changes", []):
            console.info("  " + change)
        return 0
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
    if getattr(args, "check_leakage", False):
        from .split import check_split_leakage

        report = check_split_leakage(cfg.project_root)
        if report["clean"]:
            console.ok(f"Split leakage check: {report['checked']} — CLEAN")
        else:
            console.warn(
                f"Split leakage: {report['n_overlaps']} content-hash overlap(s) — "
                f"{report['overlapping_files'][:10]}"
            )
        return 0 if report["clean"] else 1
    split(cfg.project_root, cfg, dry_run=args.dry_run)
    return 0


def cmd_export(args) -> int:
    from .export import export

    cfg = _build_config(args)
    export(cfg.project_root, cfg, which=args.which or "", format_=args.format or "")
    return 0


def cmd_sweep(args) -> int:
    from .sweep import chain_generations, run_ensemble, run_sweep

    cfg = _build_config(args)
    if args.kind == "ensemble":
        generator = None
        if args.no_cache:
            from .inference import generate

            generator = generate  # bypass the deterministic cache
        _rows, best = run_ensemble(cfg, args.prompt, n=args.n, generator=generator)
        return 0 if best else 1
    if args.kind == "chain":
        chain = chain_generations(cfg, args.prompt, steps=args.n)
        return 0 if chain else 1
    guidance = [float(x) for x in args.guidance.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.kind == "seeds":
        guidance = [cfg.inference.guidance_scale]
    _rows, best = run_sweep(cfg, args.prompt, guidance, seeds)
    return 0 if best else 1


def cmd_finetune(args) -> int:
    from .finetune import train

    cfg = _build_config(args)
    result = train(
        cfg, steps=args.steps, lr=args.lr, limit=args.limit,
        out_dir=Path(args.out) if args.out else None, r=args.r,
        warmup_steps=args.warmup, lr_mode=args.lr_mode,
        gradient_checkpointing=args.grad_ckpt, bf16=args.bf16,
        stream=args.stream, curriculum=args.curriculum, ema=args.ema,
        ema_decay=args.ema_decay, ddp=args.ddp,
        cfg_base=args.cfg_base, cfg_sweep=args.cfg_sweep,
        accum=args.accum, val_split=args.val_split,
        full=args.full, resume=args.resume or "",
        check_leakage=not args.no_leakage,
    )
    return 0 if result else 1


def cmd_tuning(args) -> int:
    from .tuning import run

    cfg = _build_config(args)
    tokens = args.tokens if getattr(args, "tokens", None) else []
    result = run(
        cfg.project_root, cfg, args.task,
        adapters_dir=Path(args.adapters) if args.adapters else None,
        metric=args.metric,
        n_trials=args.trials,
        seed=args.seed,
        model_name=args.model,
        bits=args.bits,
        tokens=tokens,
        tokenizer_name=args.tokenizer or "",
        concept=args.concept or "",
        examples=[Path(p) for p in (args.examples or [])],
        model_bytes=args.model_bytes or 0,
        vram_bytes=args.vram_bytes or 0,
        dtype=args.dtype or "fp32",
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        n=args.n,
        losses=args.losses or [],
        lrs=args.lrs or [],
        per_sample_bytes=args.per_sample_bytes or 0,
        headroom=args.headroom,
    )
    if result.get("error"):
        return 1
    return 0


def cmd_evalx(args) -> int:
    from . import evalx

    cfg = _build_config(args)
    result = evalx.run(
        cfg.project_root, cfg, args.task,
        clap=args.clap,
        clipping=args.clipping,
        silence=args.silence,
        snr_db=args.snr_db,
        a_wins=args.a_wins,
        b_wins=args.b_wins,
        ties=args.ties,
        threshold=args.threshold,
        n=args.n,
        seed=args.seed,
        prompts=args.prompts,
        ref_dir=Path(args.ref) if args.ref else None,
        gen_dir=Path(args.gen) if args.gen else None,
        limit=args.limit,
    )
    if result.get("error"):
        return 1
    return 0


def cmd_merge(args) -> int:
    from .merge import merge

    try:
        out = merge([Path(m) for m in args.models], Path(args.out), weights=args.weights)
    except Exception as exc:  # noqa: BLE001
        console.error(f"Merge failed: {exc}")
        return 1
    console.ok(f"Merged model -> {out}")
    return 0


def cmd_infer(args) -> int:
    from .inference import generate, generate_batch, generate_cached

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
    if args.preset is not None:
        icfg.preset = args.preset
    if args.target_seconds is not None:
        icfg.target_seconds = args.target_seconds
    if args.negative is not None:
        icfg.negative_prompt = args.negative
    if args.negative_retries is not None:
        icfg.negative_retries = args.negative_retries
    if getattr(args, "adapter", None):
        icfg.adapter = args.adapter

    out_dir = Path(args.out) if args.out else cfg.project_root / "outputs"

    generator = generate_cached if args.cache else generate

    if args.prompts_file:
        items = []
        for ln in Path(args.prompts_file).read_text().splitlines():
            if not ln.strip():
                continue
            if ln.lstrip().startswith("{"):
                items.append(json.loads(ln))
            else:
                items.append(ln.strip())
        if not items:
            console.error("Prompts file is empty.")
            return 1
        results = generate_batch(cfg, items, out_dir=out_dir, seed=args.seed)
        from .experiments import log_inference

        for r in results:
            log_inference(cfg, r)
    else:
        if not args.prompt:
            console.error("Provide --prompt or --prompts-file.")
            return 1
        result = generator(
            cfg, args.prompt, out_dir=out_dir, seed=args.seed,
            continue_from=Path(args.continue_from) if args.continue_from else None,
            melody_from=Path(args.melody_from) if args.melody_from else None,
            preset=args.preset,
            target_seconds=args.target_seconds,
            negative_prompt=args.negative,
            negative_retries=args.negative_retries,
        )
        if not result:
            return 1
        from .experiments import log_inference

        log_inference(cfg, result)
    return 0


def cmd_presets(args) -> int:
    cfg = _build_config(args)
    for name, vals in cfg.inference.presets.items():
        console.info(
            f"{name:10s} temperature={vals.get('temperature')} top_k={vals.get('top_k')} "
            f"top_p={vals.get('top_p')} guidance={vals.get('guidance_scale')}"
        )
    console.info(f"active preset: {cfg.inference.preset or '(none — raw knobs)'}")
    return 0


def cmd_manifest(args) -> int:
    from .reproduce import diff, load_entries

    cfg = _build_config(args)
    entries = load_entries(cfg.project_root)
    if not entries:
        console.warn("No manifest entries yet — run `musictrain infer` or `eval`.")
        return 0
    if args.diff is not None:
        i, j = args.diff
        try:
            a, b = entries[-i], entries[-j]
        except IndexError:
            console.error(f"Only {len(entries)} entries — indices 1..{len(entries)} (1 = most recent).")
            return 1
        console.title(f"diff -{i} vs -{j}:")
        for line in diff(a, b):
            console.info("  " + line)
        return 0
    for e in entries[-args.latest:]:
        commit = e.get("git_commit") or "?"
        if e.get("git_dirty"):
            commit += "*"
        console.info(
            f"{str(e.get('kind','?')):10s} {str(e.get('at',''))[:19]}  commit={commit}  "
            f"model={e.get('model')}  vocab=v{e.get('vocab_version')}  "
            f"prompt={(e.get('prompt') or '')[:50]!r}"
        )
    console.info(f"{len(entries)} total entrie(s) in metadata/repro_manifest.jsonl")
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


def cmd_artists(args) -> int:
    from .artists import ARTISTS, GENRES, MOODS, get_artist

    if args.show:
        a = get_artist(args.show)
        if a is None:
            console.error(f"Unknown artist {args.show!r}. Run `musictrain artists` to list.")
            return 1
        console.title(f"{a.name}")
        console.info(f"aliases:   {', '.join(a.aliases) or '-'}")
        console.info(f"flow:      {', '.join(a.flow)}")
        console.info(f"rhyme:     {a.rhyme_scheme}  cadence={a.cadence}  density={a.density}/5  energy={a.energy}/5")
        console.info(f"autotune:  {a.autotune}  bpm range: {a.bpm_range[0]}-{a.bpm_range[1]}")
        console.info(f"ad-libs:   {', '.join(a.ad_libs) or '-'}")
        console.info(f"slang:     {', '.join(a.slang) or '-'}")
        console.info(f"topics:    {', '.join(a.topics) or '-'}")
        console.info(f"delivery:  {a.vibe()}")
        return 0

    if args.moods:
        console.title(f"{len(MOODS)} moods")
        console.info(", ".join(MOODS))
        return 0

    if args.genres:
        console.title(f"{len(GENRES)} genre templates")
        for g in GENRES:
            console.info(f"{g.name:14s} {g.description}")
        return 0

    console.title(f"{len(ARTISTS)} artist style profiles")
    for a in ARTISTS:
        console.info(
            f"{a.name:16s} flow={'/'.join(a.flow[:2])}  "
            f"cadence={a.cadence:6s} ad-libs={len(a.ad_libs)}  "
            f"topics={', '.join(a.topics[:3])}"
        )
    console.info("Use `musictrain artists --show <name>` for a full profile.")
    return 0


def _lyric_recipe(args, root) -> dict:
    """Resolve CLI inputs (+ random/favorite/auto) into a canonical recipe dict."""
    from .lyricsprefs import autopilot, get_favorite, normalize_recipe, random_recipe

    if args.favorite and args.favorite not in ("list",):
        fav = get_favorite(root, args.favorite)
        if fav:
            return dict(fav)
        console.warn(f"favorite {args.favorite!r} not found; building from flags")
    if args.auto:
        rec = autopilot(root, seed=args.seed)
        console.info(f"autopilot: {rec.get('artist')} / {rec.get('mood')} / {rec.get('topic')}")
        return rec
    if args.random:
        return random_recipe(root, seed=args.seed)
    rec = normalize_recipe(
        artist=args.artist or "",
        mood=args.mood or "",
        topic=args.topic or "",
        genre=args.genre or "",
        seed=args.seed,
    )
    return rec


def _print_section(sec: dict) -> None:
    console.info(f"[{sec['role']} — {sec['flow']} @ {sec['cadence']} cadence, {sec['bars']} bars]")
    for ln in sec["lines"]:
        print("  " + ln)
    if sec["ad_libs"]:
        console.info("  ad-libs: " + ", ".join(f"({a})" for a in sec["ad_libs"]))


def cmd_lyrics(args) -> int:
    import json as _json

    from .lyrics import (BeatContext, SectionSpec, arrangement_specs,
                         beat_context_from_analysis, default_structure, generate,
                         regenerate_section, restyle)
    from .lyricsprefs import (add_favorite, favorite_keys, history, history_diff,
                              negatives, record_history, weights)
    from .lyrictools import (ARRANGEMENTS, annotate_section, lrc, metrics, suggest_from_chords)
    from .util import sanitize_slug

    cfg = _build_config(args)
    root = cfg.project_root

    # ---- non-generating modes -------------------------------------------------
    if args.list_favorites:
        keys = favorite_keys(root)
        if keys:
            console.info("Favorites: " + ", ".join(keys))
        else:
            console.warn("No favorites yet — use `musictrain lyrics --favorite NAME` to save one.")
        return 0

    if args.history is not None:
        rows = history(root, limit=args.history)
        if not rows:
            console.warn("No lyric history yet.")
            return 0
        for i, h in enumerate(rows):
            console.info(
                f"[{i}] {h.get('artist')} | {h.get('mood')} | {h.get('topic')} "
                f"| genre={h.get('genre') or '-'} | seed={h.get('seed')}"
            )
        return 0

    if args.diff is not None:
        rows = history(root)
        try:
            a, b = rows[-args.diff[0]], rows[-args.diff[1]]
        except IndexError:
            console.error(f"Only {len(rows)} history entries — indices 1..{len(rows)} (1 = most recent).")
            return 1
        console.title(f"lyric diff -{args.diff[0]} vs -{args.diff[1]}:")
        for line in history_diff(a, b):
            console.info("  " + line)
        return 0

    # ---- build the recipe + context -------------------------------------------
    loaded_structure = None
    if args.project_load:
        from .lyricproject import load_project

        proj = load_project(root, args.project_load)
        if proj is None:
            console.error(
                f"project {args.project_load!r} not found — run "
                f"`musictrain lyrics --project-save NAME` first"
            )
            return 1
        recipe = proj.get("recipe") or {}
        loaded_structure = proj.get("structure") or None
        console.ok(f"loaded project {args.project_load!r}")
    else:
        recipe = _lyric_recipe(args, root)
    artist = recipe["artist"]
    mood = recipe.get("mood", "")
    topic = recipe.get("topic", "")
    seed = int(recipe.get("seed", args.seed or 42))

    if args.analysis:
        try:
            analysis = _json.loads(Path(args.analysis).read_text())
        except (OSError, ValueError) as exc:
            console.error(f"Could not read analysis {args.analysis!r}: {exc}")
            return 1

        # chord → mood/topic suggestion (feature #8)
        if args.suggest:
            s = suggest_from_chords(
                analysis.get("chords", []), (analysis.get("key") or {}).get("key", "")
            )
            console.info(f"chord→mood suggestion: mood={s['mood']} topic={s['topic']} ({s['reason']})")
            return 0

        # vocal-detection gate (feature #7)
        if args.vocals:
            v = (analysis.get("vocal") or {}).get("verdict", "unknown")
            console.info(f"vocal verdict: {v}")
            return 0

        ctx = beat_context_from_analysis(analysis, artist=artist, mood=mood, topic=topic, seed=seed)
    else:
        ctx = BeatContext(
            bpm=float(args.bpm or recipe.get("bpm", 140.0)),
            key=args.key or recipe.get("key", "A minor"),
            artist=artist, mood=mood, topic=topic, seed=seed,
        )

    # apply negatives + weights (features #28/#29)
    ctx.negative = negatives(root) + list(args.negative or [])
    ctx.weights = dict(weights(root))
    for w in args.weight or []:
        if "=" in w:
            k, v = w.split("=", 1)
            ctx.weights[k.strip()] = float(v)

    # make sure there is a structure to mutate (preset/feature/duet need it)
    if not ctx.structure:
        ctx.structure = default_structure()

    # restored project structure (feature #10)
    if loaded_structure:
        ctx.structure = [
            SectionSpec(
                role=s.get("role", "verse"), bars=s.get("bars", 8),
                topic=s.get("topic", ""), artist=s.get("artist", ""),
                artist2=s.get("artist2", ""), energy=s.get("energy", 0.0),
            )
            for s in loaded_structure
        ]

    # arrangement preset (feature #5)
    if args.preset:
        specs = arrangement_specs(args.preset)
        if not specs:
            console.error(
                f"Unknown arrangement {args.preset!r}. Options: {', '.join(ARRANGEMENTS)}"
            )
            return 1
        ctx.structure = specs

    # multi-artist feature mode (#6) + duet (#5): --feature role=artist[+artist2]
    if args.feature:
        from .artists import get_artist

        for f in args.feature:
            if "=" not in f:
                console.warn(f"ignoring malformed --feature {f!r} (expected role=artist)")
                continue
            role, aid = f.split("=", 1)
            parts = [p.strip() for p in aid.split("+") if p.strip()]
            a = get_artist(parts[0]) if parts else None
            if a is None:
                console.warn(f"unknown feature artist {aid.strip()!r}")
                continue
            a2 = get_artist(parts[1]) if len(parts) > 1 else None
            found = False
            for s in ctx.structure:
                if s.role == role.strip().lower():
                    s.artist = a.id
                    if a2 is not None:
                        s.artist2 = a2.id
                    found = True
            if not found:
                console.warn(f"no {role.strip()!r} section in the structure; ignoring feature")

    # ---- single-section regeneration (feature #40) ----------------------------
    if args.section:
        sec = regenerate_section(ctx, args.section, seed=seed)
        _print_section(sec)
        return 0

    # ---- style transfer (feature #41) -----------------------------------------
    if args.restyle:
        from .artists import get_artist

        target = get_artist(args.restyle)
        if target is None:
            console.error(f"Unknown artist {args.restyle!r}. Run `musictrain artists` to list.")
            return 1
        base_result = generate(ctx)
        result = restyle(base_result, target.id, seed=seed)
        console.title(f"--- restyled as {result.artist} (from {base_result.artist}) ---")
        print(result.full_text())
        if not args.no_save:
            slug = sanitize_slug(
                f"{result.artist}_{int(result.bpm)}bpm_{result.key.replace(' ', '')}"
                f"_{result.mood}_{result.topic}_seed{result.seed}"
            )
            out_path = root / "outputs" / "lyrics" / (slug + ".txt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.full_text())
            console.ok(f"saved -> {out_path}")
            record_history(root, result.as_dict())
        return 0

    # ---- generate N variants (seed-lock re-roll, feature #39) -----------------
    n = max(1, int(args.variants or 1))
    results = []
    for i in range(n):
        vctx = ctx
        if i:
            vctx = BeatContext(
                bpm=ctx.bpm, key=ctx.key, swing=ctx.swing, energy=ctx.energy,
                artist=ctx.artist, mood=ctx.mood, topic=ctx.topic,
                structure=ctx.structure, negative=ctx.negative,
                weights=ctx.weights, seed=seed + i,
            )
        results.append(generate(vctx))

    for idx, result in enumerate(results):
        console.title(f"--- variant {idx + 1}/{n} (seed={result.seed}, backend={result.backend}) ---")
        print(result.full_text())

        # rhyme + syllable annotations (feature #2)
        if args.annotate:
            console.info("--- annotations (syllables per line) ---")
            for sec in result.sections:
                console.info(f"[{sec['role']} · target ~{sec.get('syllable_target')} syll · rhyme {sec.get('rhyme')}]")
                for ann in annotate_section(sec):
                    print(f"  ({ann['syllables']:2d}) {ann['line']}")

        slug = sanitize_slug(
            f"{result.artist}_{int(result.bpm)}bpm_{result.key.replace(' ', '')}"
            f"_{result.mood}_{result.topic}_seed{result.seed}"
        )

        # persist + history (features #30/#48)
        if args.out or not args.no_save:
            out_path = (Path(args.out) if args.out and n == 1 else root / "outputs" / "lyrics" / (slug + ".txt"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.full_text())
            console.ok(f"saved -> {out_path}")
            record_history(root, result.as_dict())

        # studio sheet export (feature #9)
        if args.sheet is not None:
            sheet_path = (
                Path(args.sheet)
                if args.sheet
                else root / "outputs" / "lyrics" / (slug + ".md")
            )
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            sheet_path.write_text(result.to_sheet())
            console.ok(f"studio sheet -> {sheet_path}")

        # lyrical metrics (feature #6)
        if args.metrics:
            m = metrics(result)
            console.info(
                f"metrics: bars={m['bars']} lines={m['lines']} "
                f"rhyme_density={m['rhyme_density']} avg_syll={m['avg_syllables']} "
                f"std={m['syllable_std']} flow_score={m['flow_score']}/100"
            )

        # LRC karaoke export (feature #9)
        if args.lrc is not None:
            lrc_path = (Path(args.lrc) if args.lrc else root / "outputs" / "lyrics" / (slug + ".lrc"))
            lrc_path.parent.mkdir(parents=True, exist_ok=True)
            lrc_path.write_text(lrc(result))
            console.ok(f"LRC -> {lrc_path}")

        # project save (feature #10)
        if args.project_save:
            from .lyricproject import save_project

            save_project(root, args.project_save, {
                "beat": args.analysis or "",
                "recipe": {"artist": recipe["artist"], "mood": mood, "topic": topic,
                           "genre": recipe.get("genre", ""), "seed": result.seed},
                "structure": [{"role": s.role, "bars": s.bars, "topic": s.topic,
                               "artist": s.artist, "artist2": s.artist2, "energy": s.energy}
                              for s in ctx.structure],
                "weights": ctx.weights,
                "negatives": ctx.negative,
                "result": result.as_dict(),
            })
            console.ok(f"project saved as {args.project_save!r}")

        if args.favorite and args.favorite != "list":
            add_favorite(root, args.favorite, {
                "artist": recipe["artist"], "mood": mood, "topic": topic,
                "genre": recipe.get("genre", ""), "seed": result.seed,
            })
            console.ok(f"favorite saved as {args.favorite!r}")

    return 0


def cmd_lyrate(args) -> int:
    from .lyricrating import build_profile, build_queue, ratings, record_rating

    cfg = _build_config(args)
    root = cfg.project_root

    if args.task == "record":
        if not args.item:
            console.error("rating record requires --item (an artist/genre/topic tag or clip id)")
            return 1
        record_rating(root, {
            "item": args.item,
            "artist": args.artist or "",
            "mood": args.mood or "",
            "topic": args.topic or "",
            "genre": args.genre or "",
            "score": args.score,
            "choice": args.choice or "",
        })
        console.ok(f"recorded rating for {args.item!r} (score={args.score})")
        return 0

    if args.task == "queue":
        items = [{"id": str(i), "label": str(i)} for i in range(max(1, int(args.n or 10)))]
        q = build_queue(root, items, n=int(args.n or 10), seed=args.seed or 0)
        console.title(f"blind A/B queue ({len(q)} pairs)")
        for i, pair in enumerate(q):
            console.info(f"pair {i}: A={pair['A']['label']}  B={pair['B']['label']}")
        return 0

    if args.task == "profile":
        profile = build_profile(root)
        console.title(f"style profile ({profile['n_ratings']} ratings)")
        for dim, label in (("artists", "artists"), ("moods", "moods"), ("topics", "topics"), ("genres", "genres")):
            top = list((profile.get(dim) or {}).items())[:5]
            if top:
                console.info(f"{label}: " + ", ".join(f"{k} ({v:.2f})" for k, v in top))
        return 0

    console.warn(f"ratings: {len(ratings(root))} total")
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
    prompts = build(
        cfg.project_root,
        force=args.force,
        adversarial=getattr(args, "adversarial", 0),
        negatives=getattr(args, "negatives", 0),
        paraphrases=getattr(args, "paraphrases", 0),
    )
    console.ok(f"Prompt set: {len(prompts)} prompts ready")
    return 0


def cmd_eval(args) -> int:
    from .evalset import run_eval

    cfg = _build_config(args)
    if args.no_clap:
        cfg.clap.enabled = False
    if getattr(args, "adapter", None):
        cfg.inference.adapter = args.adapter
    results = run_eval(
        cfg,
        limit=args.limit,
        check_bpm=not args.no_check,
        section=args.section,
        seeds=args.seeds,
        incremental=getattr(args, "incremental", False),
    )
    return 0 if results else 1


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


def cmd_metrics(args) -> int:
    from .metrics import compute

    cfg = _build_config(args)
    record = compute(
        cfg,
        Path(args.ref).resolve(),
        Path(args.gen).resolve(),
        limit=args.limit,
        fad_kind=args.fad,
    )
    return 0 if record else 1


def cmd_adherence(args) -> int:
    from .adherence import run

    cfg = _build_config(args)
    record = run(cfg.project_root, cfg, limit=args.limit)
    return 0 if record else 1


def cmd_significance(args) -> int:
    from .significance import compare, from_checkpoints, load_results, meta_analyze

    cfg = _build_config(args)
    if getattr(args, "meta", None):
        # advanced #5 — combine effect sizes from several experiment JSONs
        studies = []
        for p in args.meta:
            data = json.loads(Path(p).read_text())
            if isinstance(data, dict) and "delta" in data:
                data["label"] = p
                studies.append(data)
            elif isinstance(data, list):
                for s in data:
                    if isinstance(s, dict) and "delta" in s:
                        s["label"] = p
                        studies.append(s)
        if not studies:
            console.error("No studies with delta/se found in --meta files.")
            return 1
        out = meta_analyze(studies)
        if out.get("pooled_delta") is not None:
            console.ok(
                f"Meta-analysis: pooled delta = {out['pooled_delta']} "
                f"(se {out['se']}, z {out['z']}, p {out['p_value']}) "
                f"across {out['n_pooled']}/{out['n_studies']} studies"
            )
        path = cfg.project_root / "metadata" / "metaanalysis.json"
        path.write_text(json.dumps(out, indent=2))
        console.ok(f"Meta-analysis -> {path.relative_to(cfg.project_root)}")
        return 0

    if args.checkpoint_a and args.checkpoint_b:
        out = from_checkpoints(cfg, args.checkpoint_a, args.checkpoint_b)
    else:
        a_rows = load_results(Path(args.a).resolve())
        b_rows = load_results(Path(args.b).resolve())
        if not a_rows or not b_rows:
            console.error("Both --a and --b files must contain eval results.")
            return 1
        out = compare(cfg, a_rows, b_rows, label_a=args.a, label_b=args.b)
    if not out:
        return 1
    console.ok(out.get("summary", ""))
    return 0


def cmd_ab_eval(args) -> int:
    from .ab_eval import run_ab_eval

    cfg = _build_config(args)
    if args.no_clap:
        cfg.clap.enabled = False
    out = run_ab_eval(
        cfg,
        args.adapter,
        limit=args.limit,
        seeds=args.seeds,
        section=args.section,
    )
    if not out:
        return 1
    console.ok(out.get("summary", ""))
    return 0


def cmd_difficulty(args) -> int:
    from .difficulty import run

    cfg = _build_config(args)
    return 0 if run(cfg.project_root, cfg) else 1


def cmd_leaderboard(args) -> int:
    from .leaderboard import build

    cfg = _build_config(args)
    return 0 if build(cfg) else 1


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
    from .dedup import dedup_segments, find_duplicates

    cfg = _build_config(args)
    if args.move:
        cfg.dedup.action = "move"
    if args.segments:
        dedup_segments(cfg.project_root, cfg, which="segments")
    else:
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


def cmd_deep(args) -> int:
    from .audio.deep import deep

    cfg = _build_config(args)
    path = Path(args.path).resolve() if args.path else None
    records = deep(cfg.project_root, cfg, which=args.dir, limit=args.limit, path=path)
    return 0 if records else 1


def cmd_dataeng(args) -> int:
    from . import dataeng as de

    cfg = _build_config(args)
    root = cfg.project_root
    task = args.task
    (root / "metadata").mkdir(parents=True, exist_ok=True)

    if task == "transcribe":
        rec = de.transcribe(Path(args.path).resolve(), model_name=args.model)
        if rec:
            console.ok(f"ASR: {rec['text'][:120]}")
        return 0 if rec else 1

    if task == "dedup":
        from .embeddings import embed_dir

        emb = embed_dir(root, cfg, which=args.dir, limit=args.limit)
        if not emb:
            console.error("No embeddings produced.")
            return 1
        import numpy as np

        mat = np.stack(list(emb.values()))
        report = de.corpus_dedup(mat, list(emb.keys()), threshold=args.threshold)
        (root / "metadata" / "corpus_duplicates.json").write_text(json.dumps(report, indent=2))
        console.ok(f"Corpus dedup: {report['n_duplicates']} duplicate(s) in {report['n']} files")
        return 0

    if task == "quality":
        from .audio.inventory import AUDIO_GLOB

        target = root / "data" / args.dir
        files = [p for pat in AUDIO_GLOB for p in sorted(target.glob(pat))]
        recs = [de.sample_quality(p) for p in files[:args.limit] if args.limit] or \
               [de.sample_quality(p) for p in files]
        (root / "metadata" / "sample_quality.json").write_text(json.dumps(recs, indent=2))
        console.ok(f"Sample quality -> metadata/sample_quality.json ({len(recs)} files)")
        return 0

    if task == "snapshot":
        return 0 if de.snapshot(root, cfg, label=args.label, which=args.dir) else 1

    if task == "expand":
        prompts = de.expand_prompts(args.n, seed=args.seed)
        out = root / "metadata" / "synthetic_prompts.jsonl"
        with out.open("w") as fh:
            for p in prompts:
                fh.write(json.dumps(p) + "\n")
        console.ok(f"Expanded prompts -> {out.relative_to(root)} ({len(prompts)})")
        return 0

    if task == "cooccur":
        from .report import load_results

        rows = load_results(root)
        report = de.tag_cooccurrence(rows)
        (root / "metadata" / "cooccurrence.json").write_text(json.dumps(report, indent=2))
        console.ok(f"Co-occurrence -> metadata/cooccurrence.json ({report['n_combos']} combos)")
        return 0

    if task == "sample":
        from .report import load_results

        rows = load_results(root)
        labels = [r.get("section") or "?" for r in rows]
        idx = de.balanced_sample(rows, labels, args.n, seed=args.seed)
        picked = [rows[i] for i in idx]
        (root / "metadata" / "balanced_sample.json").write_text(json.dumps(picked, indent=2))
        console.ok(f"Balanced sample -> metadata/balanced_sample.json ({len(picked)} rows)")
        return 0

    if task == "provenance":
        from .report import load_results

        rows = de.annotate_provenance(load_results(root), source_url=args.source,
                                      license_name=args.license, origin=args.origin)
        (root / "metadata" / "provenance.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
        console.ok(f"Provenance -> metadata/provenance.jsonl ({len(rows)} rows)")
        return 0

    if task == "annotate":
        rows = de.pre_annotate(root, cfg, which=args.dir, limit=args.limit)
        return 0 if rows else 1

    console.error(f"Unknown task: {task}")
    return 1


def cmd_resynth(args) -> int:
    from .resynth import resynth, rebuild_instrumental

    cfg = _build_config(args)
    gains = None
    if args.gain:
        gains = {}
        for pair in args.gain:
            if "=" in pair:
                name, val = pair.split("=", 1)
                gains[name.strip()] = float(val)
    if args.instrumental:
        rebuild_instrumental(cfg.project_root, cfg, limit=args.limit)
    else:
        resynth(cfg.project_root, cfg, which=args.dir, gains=gains, limit=args.limit)
    return 0


def cmd_invert(args) -> int:
    from .invert import invert

    cfg = _build_config(args)
    invert(cfg.project_root, cfg, Path(args.path).resolve(), top_k=args.top)
    return 0


def cmd_active(args) -> int:
    from .active import rank_unlabeled

    cfg = _build_config(args)
    rank_unlabeled(cfg.project_root, cfg, which=args.dir, top_k=args.top)
    return 0


def cmd_augment(args) -> int:
    from .augment import augment

    cfg = _build_config(args)
    ops = args.ops.split(",") if args.ops else None
    augment(cfg.project_root, cfg, which=args.dir, ops=ops, limit=args.limit)
    return 0


def cmd_sections(args) -> int:
    from .sections import auto_sections

    cfg = _build_config(args)
    auto_sections(cfg.project_root, cfg, force=args.force)
    return 0


def cmd_drift(args) -> int:
    from .drift import drift_report

    cfg = _build_config(args)
    drift_report(
        cfg.project_root, cfg,
        reference=args.reference, current=args.current, threshold=args.threshold,
    )
    return 0


def cmd_curation(args) -> int:
    from .curation import curation_score

    cfg = _build_config(args)
    curation_score(cfg.project_root, cfg, which=args.dir, top_k=args.top)
    return 0


def cmd_embed_refresh(args) -> int:
    from .embeddings import refresh

    cfg = _build_config(args)
    refresh(cfg.project_root, cfg, which=args.dir, limit=args.limit)
    return 0


def cmd_labelprop(args) -> int:
    from .labelprop import propagate, leakage_check

    cfg = _build_config(args)
    if args.check_leakage:
        leakage_check(cfg.project_root, cfg)
    else:
        propagate(cfg.project_root, cfg, which=args.dir, min_confidence=args.min_confidence)
    return 0


def cmd_registry(args) -> int:
    from .registry import scan_registry

    cfg = _build_config(args)
    scan_registry(cfg.project_root, cfg)
    return 0


def cmd_diff_weights(args) -> int:
    from .registry import diff_weights

    cfg = _build_config(args)
    a = (cfg.project_root / "checkpoints" / args.a).resolve()
    b = (cfg.project_root / "checkpoints" / args.b).resolve()
    diff_weights(a, b, top_k=args.top)
    return 0


def cmd_archive(args) -> int:
    from .registry import archive

    cfg = _build_config(args)
    archive(cfg.project_root, cfg, args.checkpoint)
    return 0


def cmd_prune(args) -> int:
    from .registry import prune_checkpoints

    cfg = _build_config(args)
    report = prune_checkpoints(cfg.project_root, cfg, keep=args.keep, delete=args.delete)
    return 0 if report else 1


def cmd_gate(args) -> int:
    from .gates import eval_gate

    cfg = _build_config(args)
    report = eval_gate(
        cfg.project_root, cfg,
        baseline=args.baseline, candidate=args.candidate,
        max_clap_drop=args.max_clap_drop, max_deviation_increase=args.max_dev,
    )
    return 0 if report.get("passed", False) else 1


def cmd_drift_check(args) -> int:
    from .gates import drift_detector

    cfg = _build_config(args)
    report = drift_detector(
        cfg.project_root, cfg,
        reference=args.reference, current=args.current,
        ks_threshold=args.ks, psi_threshold=args.psi,
    )
    return 0 if report.get("passed", False) else 1


def cmd_promote(args) -> int:
    from .gates import promotion_report

    cfg = _build_config(args)
    promotion_report(cfg.project_root, cfg, args.checkpoint, baseline=args.baseline)
    return 0


def cmd_monitor(args) -> int:
    from .monitor import training_monitor

    cfg = _build_config(args)
    training_monitor(cfg, limit=args.limit)
    return 0


def cmd_matrix(args) -> int:
    from .monitor import experiment_matrix

    cfg = _build_config(args)
    experiment_matrix(cfg)
    return 0


def cmd_modelcard(args) -> int:
    from .monitor import model_card

    cfg = _build_config(args)
    model_card(cfg, checkpoint=args.checkpoint or "")
    return 0


def cmd_early_stop(args) -> int:
    from .monitor import early_stop

    if not args.series:
        console.error("Pass --series as comma-separated CLAP values, e.g. 0.30,0.32,0.31")
        return 2
    series = [float(x) for x in args.series.split(",")]
    result = early_stop(series, patience=args.patience, min_delta=args.min_delta)
    console.info(f"best={result['best_clap']} at step {result['best_step']} "
                 f"({result['steps_since_best']} step(s) since)")
    if result["should_stop"]:
        console.warn(f"STOP: {result['reason']}")
        return 0
    console.ok(f"KEEP TRAINING: {result['reason']}")
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    cfg = _build_config(args)
    return serve(cfg, port=args.port, token=getattr(args, "token", ""))


def cmd_register(args) -> int:
    from .registry_ml import register_model

    cfg = _build_config(args)
    register_model(cfg, args.checkpoint, stage=args.stage, update=args.update)
    return 0


def cmd_models(args) -> int:
    from .registry_ml import list_models

    cfg = _build_config(args)
    models = list_models(cfg)
    if not models:
        console.info("No registered models yet — run `musictrain register`.")
        return 0
    for m in models:
        console.info(f"{m['name']}")
        for v in m["versions"]:
            console.info(f"  v{v['version']} [{v['stage']}] run {v['run_id'][:8]} ({v['status']})")
    return 0


def cmd_stage(args) -> int:
    from .registry_ml import transition

    cfg = _build_config(args)
    if not transition(cfg, args.name, args.version, args.stage):
        return 1
    return 0


def cmd_modelops(args) -> int:
    from . import modelops as mo

    cfg = _build_config(args)
    root = cfg.project_root
    task = args.task
    (root / "metadata").mkdir(parents=True, exist_ok=True)

    if task == "migrate-aliases":
        mo.migrate_stages_to_aliases(cfg, args.name)
        return 0
    if task == "ab":
        from .report import load_results

        rows = load_results(root)
        a = [r["clap_score"] for r in rows
             if r.get("checkpoint") == args.champion and r.get("clap_score") is not None]
        b = [r["clap_score"] for r in rows
             if r.get("checkpoint") == args.challenger and r.get("clap_score") is not None]
        result = mo.ab_win_rate(a, b, higher_is_better=True)
        console.ok(json.dumps(result, indent=2))
        return 0 if result["n"] else 1
    if task == "auto-promote":
        mo.auto_promote(cfg, args.candidate, args.baseline)
        return 0
    if task == "lineage":
        mo.record_lineage(cfg, args.parent, args.child, note=args.note or "")
        return 0
    if task == "lineage-graph":
        console.ok(json.dumps(mo.lineage_graph(cfg), indent=2))
        return 0
    if task == "checksum":
        model_dir = Path(args.path).resolve()
        manifest = mo.checksum_dir(model_dir)
        (root / "metadata" / "checksum_manifest.json").write_text(json.dumps(manifest, indent=2))
        console.ok(f"Checksum manifest -> metadata/checksum_manifest.json ({manifest['n_files']} files)")
        return 0
    if task == "verify":
        manifest = json.loads((root / "metadata" / "checksum_manifest.json").read_text())
        ok = mo.verify_checksum(Path(args.path).resolve(), manifest)
        console.ok("Checksum verified." if ok else "Checksum MISMATCH.")
        return 0 if ok else 1
    if task == "rollback":
        return 0 if mo.rollback(cfg, args.name).get("rolled_back") else 1
    if task == "cost-breakdown":
        out = mo.cost_breakdown(args.model, args.prompts, args.seeds, tokens_per_clip=args.tokens)
        console.ok(json.dumps(out, indent=2))
        return 0
    if task == "lint":
        return 0 if mo.lint(cfg)["valid"] else 1

    console.error(f"Unknown task: {task}")
    return 1


def cmd_export_eval(args) -> int:
    from .telemetry import export_wandb

    cfg = _build_config(args)
    export_wandb(cfg, project=args.project)
    return 0


def cmd_runlog(args) -> int:
    from .telemetry import read_runlog

    cfg = _build_config(args)
    rows = read_runlog(cfg.project_root, event=args.event, limit=args.limit)
    if not rows:
        console.info("No runlog entries.")
        return 0
    for r in rows:
        console.info(json.dumps(r))
    return 0


def cmd_alert(args) -> int:
    from .alerts import alert

    cfg = _build_config(args)
    alert(
        cfg,
        min_clap=args.min_clap,
        max_abs_deviation=args.max_dev,
        min_ok_pct=args.min_ok,
        slack_webhook=args.slack_webhook or "",
        discord_webhook=args.discord_webhook or "",
        telegram_token=args.telegram_token or "",
        telegram_chat=args.telegram_chat or "",
        smtp_host=args.smtp_host or "",
        smtp_user=args.smtp_user or "",
        smtp_password=args.smtp_password or "",
        smtp_to=args.smtp_to or "",
    )
    return 0


def cmd_warm_cache(args) -> int:
    from .cache_warm import warm

    cfg = _build_config(args)
    out = warm(cfg, model_name=args.model or None)
    if out.get("warmed"):
        console.ok(f"Cache warm: {out['model']} ({out.get('cached_bytes')} bytes cached)")
    return 0 if out.get("warmed") else 1


def cmd_dataversion(args) -> int:
    from . import dataversion as dv

    cfg = _build_config(args)
    root = cfg.project_root
    if args.task == "commit":
        return 0 if dv.commit(root, which=args.which, label=args.label or "") else 1
    if args.task == "list":
        for v in dv.load_versions(root):
            console.info(f"{v['name']:6s} {v['label']:20s} {v['n_files']} file(s) @ {v['at'][:19]}")
        return 0
    if args.task == "diff":
        return 0 if dv.diff(root, args.v1, args.v2) else 1
    if args.task == "rollback":
        return 0 if dv.rollback(root, args.version) else 1
    console.error(f"Unknown dataversion task {args.task!r}")
    return 1


def cmd_campaign(args) -> int:
    from . import listening_campaign as lc

    cfg = _build_config(args)
    root = cfg.project_root
    if args.task == "start":
        out = lc.start(root, args.name, mode=args.mode, seed=args.seed, limit=args.limit)
        return 0 if out.get("n_items") else 1
    if args.task == "record":
        return 0 if lc.record(root, args.name, args.rater, args.item, args.choice,
                              rating=args.rating, note=args.note or "") else 1
    if args.task == "agreement":
        a = lc.agreement(root, args.name)
        console.ok(json.dumps(a, indent=2))
        return 0
    if args.task == "unblind":
        console.ok(json.dumps(lc.unblind(root, args.name), indent=2))
        return 0
    console.error(f"Unknown campaign task {args.task!r}")
    return 1


def cmd_backup(args) -> int:
    from .backup import run

    cfg = _build_config(args)
    result = run(
        cfg, args.task, label=args.label or "", archive=args.archive or "",
        force=args.force, include_mlflow=not args.no_mlflow,
    )
    return 1 if result.get("error") else 0


def cmd_audioext(args) -> int:
    from .audioext import run

    cfg = _build_config(args)
    result = run(
        cfg, args.task, path=args.path or "", bpm=args.bpm, key=args.key,
        semitones=args.semitones, tempo_ratio=args.tempo_ratio,
        which=args.which, dest=args.dest or "", limit=args.limit,
    )
    return 1 if result.get("error") else 0


def cmd_cost(args) -> int:
    from .cost import estimate, log_cost, cost_summary

    cfg = _build_config(args)
    if args.estimate:
        est = estimate(args.estimate, args.clips, tokens_per_clip=args.tokens)
        console.info(json.dumps(est, indent=2))
        return 0
    log_cost(cfg, args.task, args.model, args.clips, tokens_per_clip=args.tokens,
             n_epochs=args.epochs, lora_rank=args.lora)
    summary = cost_summary(cfg)
    console.info(f"Total: {summary['total_kwh']} kWh across {summary['runs']} run(s)")
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
        sp.add_argument("--verbose", action="store_true", help="DEBUG logging + tracebacks")
        sp.add_argument("--quiet", action="store_true", help="Only warnings/errors on console")

    # init
    sp = sub.add_parser("init", help="Create project layout + default config")
    sp.add_argument("--root", default=".")
    sp.add_argument("--verbose", action="store_true", help="DEBUG logging + tracebacks")
    sp.add_argument("--quiet", action="store_true", help="Only warnings/errors on console")
    sp.set_defaults(func=cmd_init)

    # config
    sp = sub.add_parser("config", help="Print the effective configuration")
    add_common(sp)
    sp.add_argument("--migrate", action="store_true",
                    help="Upgrade an old config schema in place (renames + backfill defaults) (#19)")
    sp.add_argument("--no-backup", action="store_true", help="Skip the .bak backup before migrating")
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
    sp.add_argument("--check-leakage", action="store_true",
                    help="Check train/val/test content-hash disjunction instead of splitting (#8)")
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
    sp.add_argument("--dtype", default=None, choices=["float32", "float16", "bf16"],
                    help="Model dtype (advanced #11)")
    sp.add_argument("--guidance", type=float, default=None)
    sp.add_argument("--max-new-tokens", type=int, default=None)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--out", default=None, help="Output directory")
    sp.add_argument("--preset", default=None, help="Sampling preset: standard | creative | precise (#37)")
    sp.add_argument("--target-seconds", type=float, default=None, help="Generate ~N seconds of audio (#39)")
    sp.add_argument("--negative", default=None, help="CLAP-checked 'no X' constraints (#33)")
    sp.add_argument("--negative-retries", type=int, default=None, help="Auto-regenerate until negative constraint passes (#33)")
    sp.add_argument("--continue-from", default=None, help="Continue from an existing audio clip (#35)")
    sp.add_argument("--melody-from", default=None, help="Follow a clip's melody (use musicgen-melody) (#36)")
    sp.add_argument("--adapter", default=None, help="LoRA adapter dir to load onto the base model (#1)")
    sp.add_argument("--cache", action="store_true",
                    help="Deterministic cache: identical settings reuse prior output (advanced #20)")
    sp.set_defaults(func=cmd_infer)

    # presets
    sp = sub.add_parser("presets", help="List sampling presets (#37)")
    add_common(sp)
    sp.set_defaults(func=cmd_presets)

    # manifest
    sp = sub.add_parser("manifest", help="Show/diff reproducibility manifest entries (#38)")
    add_common(sp)
    sp.add_argument("--latest", type=int, default=5, help="Show the N most recent entries")
    sp.add_argument("--diff", type=int, nargs=2, default=None, metavar=("IDX", "IDX"),
                    help="Diff two entries by recency index (1 = most recent), e.g. --diff 1 2")
    sp.set_defaults(func=cmd_manifest)

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

    # artists (rap/lyrics pivot — 22 style profiles, genres, moods)
    sp = sub.add_parser(
        "artists",
        help="List rapper style profiles, genre templates, and moods",
    )
    add_common(sp)
    sp.add_argument("--show", default=None, help="Show one artist's full profile (id/name/alias)")
    sp.add_argument("--moods", action="store_true", help="List the expanded mood catalog")
    sp.add_argument("--genres", action="store_true", help="List genre templates")
    sp.set_defaults(func=cmd_artists)

    # lyrics (rap/lyrics pivot — beat-driven lyric generation)
    sp = sub.add_parser(
        "lyrics",
        help="Generate rapper-style lyrics from a beat analysis (or manual flags)",
    )
    add_common(sp)
    sp.add_argument("--analysis", default=None, help="Path to a metadata/analysis.jsonl record (JSON file)")
    sp.add_argument("--artist", default=None, help="Artist id/name/alias (e.g. lil-durk, Future)")
    sp.add_argument("--mood", default=None, help="Mood (see `musictrain artists --moods`)")
    sp.add_argument("--topic", default=None, help="Topic (e.g. pain, loyalty, success)")
    sp.add_argument("--genre", default=None, help="Genre template to apply as defaults")
    sp.add_argument("--bpm", type=float, default=None, help="BPM override (ignored with --analysis)")
    sp.add_argument("--key", default=None, help="Key override, e.g. 'A minor' (ignored with --analysis)")
    sp.add_argument("--seed", type=int, default=None, help="Deterministic seed")
    sp.add_argument("--section", default=None, help="Regenerate only this section (intro/verse/hook/...)")
    sp.add_argument("--restyle", default=None, help="Re-render the lyrics in another artist's style (id/name/alias)")
    sp.add_argument("--variants", type=int, default=None, help="Generate N seed variants for A/B (re-roll)")
    sp.add_argument("--preset", default=None, help="Arrangement preset (standard/hook-first/double-verse/16-bar-opener/short-form/long-form)")
    sp.add_argument("--feature", action="append", default=None, help="Per-section artist override role=artist (repeatable; artist1+artist2 = duet)")
    sp.add_argument("--auto", action="store_true", help="Style-profile autopilot (auto-pick artist/mood/topic)")
    sp.add_argument("--annotate", action="store_true", help="Print rhyme + syllable annotations per line")
    sp.add_argument("--metrics", action="store_true", help="Print lyrical metrics (rhyme density, flow score)")
    sp.add_argument("--sheet", nargs="?", const="", default=None, help="Export a markdown studio sheet (optional path)")
    sp.add_argument("--lrc", nargs="?", const="", default=None, help="Export LRC (karaoke) timestamps (optional path)")
    sp.add_argument("--project-save", default=None, help="Save beat+lyrics+settings as a named project")
    sp.add_argument("--project-load", default=None, help="Load a saved project by name")
    sp.add_argument("--suggest", action="store_true", help="Suggest mood/topic from the beat's chords (needs --analysis)")
    sp.add_argument("--vocals", action="store_true", help="Report the beat's vocal/instrumental verdict (needs --analysis)")
    sp.add_argument("--random", action="store_true", help="Assemble a surprise-me recipe")
    sp.add_argument("--negative", action="append", default=None, help="Banned word/topic (repeatable)")
    sp.add_argument("--weight", action="append", default=None, help="Prompt weight key=value (repeatable)")
    sp.add_argument("--favorite", default=None, help="Save current recipe as a named favorite")
    sp.add_argument("--list-favorites", action="store_true", help="List saved favorites")
    sp.add_argument("--history", type=int, default=None, help="Show the last N generations")
    sp.add_argument("--diff", type=int, nargs=2, default=None, metavar=("IDX", "IDX"),
                    help="Diff two history entries by recency (1 = most recent)")
    sp.add_argument("--out", default=None, help="Output file (default: auto-named under outputs/lyrics/)")
    sp.add_argument("--no-save", action="store_true", help="Don't write output or history")
    sp.set_defaults(func=cmd_lyrics)

    # lyrate (rap/lyrics pivot — rating queue + style profile)
    sp = sub.add_parser(
        "lyrate",
        help="Rate lyrics to build a personal style profile",
    )
    add_common(sp)
    sp.add_argument("--task", default="profile", choices=["record", "queue", "profile"],
                    help="record: log a rating; queue: blind A/B queue; profile: show taste profile")
    sp.add_argument("--item", default="", help="Item id/tag being rated (record)")
    sp.add_argument("--artist", default="", help="Artist tag (record)")
    sp.add_argument("--mood", default="", help="Mood tag (record)")
    sp.add_argument("--topic", default="", help="Topic tag (record)")
    sp.add_argument("--genre", default="", help="Genre tag (record)")
    sp.add_argument("--score", type=float, default=0.5, help="Rating score 0..1 or MOS (record)")
    sp.add_argument("--choice", default="", help="X|Y|tie (record)")
    sp.add_argument("--n", type=int, default=10, help="Queue length (queue)")
    sp.add_argument("--seed", type=int, default=0, help="Shuffle seed (queue)")
    sp.set_defaults(func=cmd_lyrate)

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
    sp.add_argument("--adversarial", type=int, default=0,
                    help="Append N adversarial (tricky) prompts (advanced #10)")
    sp.add_argument("--negatives", type=int, default=0,
                    help="Append N negative-control (nonsense) prompts (advanced #11)")
    sp.add_argument("--paraphrases", type=int, default=0,
                    help="Append N paraphrase groups for robustness eval (advanced #12)")
    sp.set_defaults(func=cmd_evalset)

    # eval
    sp = sub.add_parser("eval", help="Run batch inference + BPM checks over the eval set")
    add_common(sp)
    sp.add_argument("--limit", type=int, default=0, help="Only run the first N prompts")
    sp.add_argument("--section", default=None, help="Only run prompts for this section (comma-separated for multiple)")
    sp.add_argument("--seeds", type=int, default=1, help="Seeds per prompt (e.g. 3) for majority verdicts")
    sp.add_argument("--no-check", action="store_true", help="Skip BPM post-check")
    sp.add_argument("--no-clap", action="store_true", help="Skip CLAP prompt-adherence scoring")
    sp.add_argument("--incremental", action="store_true",
                    help="Keep passed rows, re-run only failed/new prompts (advanced #49)")
    sp.add_argument("--adapter", default=None, help="LoRA adapter dir to eval against (#1)")
    sp.set_defaults(func=cmd_eval)

    # difficulty (advanced #6-#9)
    sp = sub.add_parser(
        "difficulty",
        help="Prompt difficulty, section x BPM interaction, CLAP z-scores, threshold calibration",
    )
    add_common(sp)
    sp.set_defaults(func=cmd_difficulty)

    # sweep (advanced #14/#15/#18/#19)
    sp = sub.add_parser(
        "sweep",
        help="Guidance/seed search, prompt ensembling, and conditioning chaining (advanced)",
    )
    add_common(sp)
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--kind", choices=["sweep", "ensemble", "chain", "seeds"], default="sweep",
                    help="sweep: guidance grid; seeds: seed grid; ensemble: prompt variants; chain: melody chaining")
    sp.add_argument("--guidance", default="2.0,3.0,4.0", help="Comma-separated guidance values")
    sp.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    sp.add_argument("--n", type=int, default=4, help="Variants (ensemble) or steps (chain)")
    sp.add_argument("--no-cache", action="store_true", help="Disable the deterministic cache")
    sp.set_defaults(func=cmd_sweep)

    # finetune (advanced #12)
    sp = sub.add_parser("finetune", help="LoRA fine-tune the decoder on segments + labels (advanced)")
    add_common(sp)
    sp.add_argument("--steps", type=int, default=5, help="Gradient steps (0 = dry run)")
    sp.add_argument("--lr", type=float, default=1e-4)
    sp.add_argument("--limit", type=int, default=0, help="Limit training pairs")
    sp.add_argument("--out", default=None, help="Adapter output dir (default adapters/)")
    sp.add_argument("--r", type=int, default=8, help="LoRA rank")
    sp.add_argument("--warmup", type=int, default=0, help="LR warmup steps (#23)")
    sp.add_argument("--lr-mode", choices=["constant", "cosine"], default="cosine",
                    help="LR schedule after warmup (#23)")
    sp.add_argument("--grad-ckpt", action="store_true", help="Gradient checkpointing (#22)")
    sp.add_argument("--bf16", action="store_true", help="bf16 mixed precision on CUDA (#24)")
    sp.add_argument("--stream", action="store_true", help="Stream batches from disk (#26)")
    sp.add_argument("--curriculum", action="store_true", help="Easy-first curriculum ordering (#27)")
    sp.add_argument("--ema", action="store_true", help="Weight EMA (#29)")
    sp.add_argument("--ema-decay", type=float, default=0.999)
    sp.add_argument("--ddp", action="store_true", help="Multi-GPU DDP (#30)")
    sp.add_argument("--cfg-base", type=float, default=3.0, help="Base guidance scale for sweep (#28)")
    sp.add_argument("--cfg-sweep", type=int, default=0, help="Number of CFG candidates to log (#28)")
    sp.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps (#4)")
    sp.add_argument("--val-split", type=float, default=0.0, help="Fraction held out for per-epoch validation (#4)")
    sp.add_argument("--full", action="store_true", help="Full fine-tune of the decoder (no LoRA) (#5)")
    sp.add_argument("--resume", default=None, help="Resume from a prior adapter/checkpoint dir (#3)")
    sp.add_argument("--no-leakage", action="store_true", help="Skip the train/val/test leakage guard (#8)")
    sp.set_defaults(func=cmd_finetune)

    # tuning (gap #1-#7)
    sp = sub.add_parser(
        "tuning",
        help="Training helpers: resume, HPO, MLX, quantization, tokenizer, inversion, grad-accum plan",
    )
    add_common(sp)
    sp.add_argument("--task", required=True,
                    choices=["resume", "hpo", "mlx", "quantize", "tokens", "inversion", "plan",
                             "lr-find", "pick-lr", "auto-batch"],
                    help="Which tuning helper to run")
    sp.add_argument("--adapters", default=None, help="Adapters dir for resume")
    sp.add_argument("--metric", default="leaderboard_score", help="HPO objective metric")
    sp.add_argument("--trials", type=int, default=10, help="HPO trial count")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--model", default=None, help="Model name for quantize")
    sp.add_argument("--bits", type=int, default=8, choices=[4, 8], help="Quantization bits")
    sp.add_argument("--tokens", nargs="*", default=None, help="Custom style tokens to register")
    sp.add_argument("--tokenizer", default=None, help="Tokenizer id to actually extend (e.g. facebook/musicgen-small)")
    sp.add_argument("--concept", default=None, help="Concept for textual inversion")
    sp.add_argument("--examples", nargs="*", default=None, help="Example audio for inversion")
    sp.add_argument("--model-bytes", type=int, default=0, help="Model size for grad-accum plan")
    sp.add_argument("--vram-bytes", type=int, default=0, help="VRAM budget for grad-accum plan")
    sp.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    sp.add_argument("--min-lr", type=float, default=1e-6, help="LR range-test lower bound (#6)")
    sp.add_argument("--max-lr", type=float, default=1e-2, help="LR range-test upper bound (#6)")
    sp.add_argument("--n", type=int, default=10, help="LR range-test candidate count (#6)")
    sp.add_argument("--losses", default=None, help="Comma-separated losses for pick-lr")
    sp.add_argument("--lrs", default=None, help="Comma-separated LRs for pick-lr")
    sp.add_argument("--per-sample-bytes", type=int, default=0, help="Bytes/sample for auto-batch (#6)")
    sp.add_argument("--headroom", type=float, default=0.15, help="VRAM headroom for auto-batch (#6)")
    sp.set_defaults(func=cmd_tuning)

    # evalx (gap #8-#13)
    sp = sub.add_parser(
        "evalx",
        help="Extended eval: mos, ab, leakage, robust, fad-gate, genre-gate",
    )
    add_common(sp)
    sp.add_argument("--task", required=True,
                    choices=["mos", "ab", "leakage", "robust", "fad-gate", "genre-gate"])
    sp.add_argument("--clap", type=float, default=None, help="CLAP score (mos)")
    sp.add_argument("--clipping", type=float, default=0.0, help="Clip fraction 0..1 (mos)")
    sp.add_argument("--silence", type=float, default=0.0, help="Silence fraction 0..1 (mos)")
    sp.add_argument("--snr-db", type=float, default=None, help="SNR dB (mos)")
    sp.add_argument("--a-wins", type=int, default=0, help="A wins (ab)")
    sp.add_argument("--b-wins", type=int, default=0, help="B wins (ab)")
    sp.add_argument("--ties", type=int, default=0, help="Ties (ab)")
    sp.add_argument("--threshold", type=float, default=None,
                    help="Leakage similarity (default .95) or FAD gate (default 10.0)")
    sp.add_argument("--n", type=int, default=0, help="Number of prompts to perturb (robust)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--prompts", nargs="*", default=None, help="Prompts to perturb (robust)")
    sp.add_argument("--ref", default=None, help="Reference audio dir (fad-gate)")
    sp.add_argument("--gen", default=None, help="Generated audio dir (fad-gate)")
    sp.add_argument("--limit", type=int, default=0, help="File limit per dir (fad-gate)")
    sp.set_defaults(func=cmd_evalx)

    # merge (advanced #13)
    sp = sub.add_parser(
        "merge",
        help="Average weights of 2+ checkpoints into a merged model dir",
    )
    add_common(sp)
    sp.add_argument("--models", nargs="+", required=True, help="Checkpoint dirs to average")
    sp.add_argument("--out", required=True, help="Output model dir")
    sp.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Optional per-model weights (defaults to equal)")
    sp.set_defaults(func=cmd_merge)

    # report
    sp = sub.add_parser("report", help="Export eval results to CSV + HTML")
    add_common(sp)
    sp.set_defaults(func=cmd_report)

    # metrics
    sp = sub.add_parser(
        "metrics",
        help="FAD (CLAP) + spectral KL between reference and generated audio (#41)",
    )
    add_common(sp)
    sp.add_argument("--ref", required=True, help="Reference audio dir (e.g. data/clean)")
    sp.add_argument("--gen", required=True, help="Generated audio dir (e.g. outputs/eval)")
    sp.add_argument("--limit", type=int, default=0, help="Limit files per set")
    sp.add_argument("--fad", choices=["clap", "vggish"], default="clap",
                    help="FAD embedding: clap (default) or vggish via fadtk (advanced #1)")
    sp.set_defaults(func=cmd_metrics)

    # adherence
    sp = sub.add_parser(
        "adherence",
        help="Adherence metrics: onset alignment, key, structure, duration, "
             "instruments, seed diversity, reliability, genre gates (advanced eval #1-#10)",
    )
    add_common(sp)
    sp.add_argument("--limit", type=int, default=0, help="Limit unique clips scored")
    sp.set_defaults(func=cmd_adherence)

    # significance
    sp = sub.add_parser(
        "significance",
        help="Paired significance between two eval result sets (#44)",
    )
    add_common(sp)
    sp.add_argument("--a", default=None, help="First eval_results.jsonl")
    sp.add_argument("--b", default=None, help="Second eval_results.jsonl")
    sp.add_argument("--checkpoint-a", default=None, help="Checkpoint name in eval_results.jsonl")
    sp.add_argument("--checkpoint-b", default=None, help="Checkpoint name in eval_results.jsonl")
    sp.add_argument("--meta", nargs="*", default=None,
                    help="JSON file(s) with {delta, se} per study for meta-analysis (advanced #5)")
    sp.set_defaults(func=cmd_significance)

    # ab-eval (gap #2: base vs fine-tuned on the fixed eval set)
    sp = sub.add_parser(
        "ab-eval",
        help="Run the fixed eval set on base vs adapter and diff with significance",
    )
    add_common(sp)
    sp.add_argument("--adapter", required=True, help="LoRA adapter dir to compare against the base model")
    sp.add_argument("--limit", type=int, default=0, help="Only run the first N prompts")
    sp.add_argument("--section", default=None, help="Only run prompts for this section (comma-separated)")
    sp.add_argument("--seeds", type=int, default=1, help="Seeds per prompt")
    sp.add_argument("--no-clap", action="store_true", help="Skip CLAP prompt-adherence scoring")
    sp.set_defaults(func=cmd_ab_eval)

    # leaderboard
    sp = sub.add_parser(
        "leaderboard",
        help="Rank checkpoints by adherence + per-tag CLAP (#45)",
    )
    add_common(sp)
    sp.set_defaults(func=cmd_leaderboard)

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
    sp.add_argument("--segments", action="store_true",
                    help="Dedup data/segments instead (post-segment dedup, advanced #25)")
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

    # deep
    sp = sub.add_parser(
        "deep",
        help="Deep signal analysis: tempo drift, groove, loudness, stereo, "
             "artifacts, spectral profile, onset density, frequency masking (advanced #13-#20)",
    )
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to scan (ignored with --path)")
    sp.add_argument("--path", default=None, help="Analyze a single audio file")
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_deep)

    # dataeng
    sp = sub.add_parser(
        "dataeng",
        help="Data-engineering tasks: transcribe, dedup, quality, snapshot, "
             "expand, cooccur, sample, provenance, annotate (advanced #31-#40)",
    )
    add_common(sp)
    sp.add_argument("--task", required=True,
                    choices=["transcribe", "dedup", "quality", "snapshot", "expand",
                             "cooccur", "sample", "provenance", "annotate"])
    sp.add_argument("--dir", default="clean", help="data/<dir> to operate on")
    sp.add_argument("--path", default=None, help="Single file (transcribe)")
    sp.add_argument("--model", default="base", help="Whisper model (transcribe)")
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--threshold", type=float, default=0.97, help="Dedup similarity threshold")
    sp.add_argument("--label", default=None, help="Snapshot label")
    sp.add_argument("--n", type=int, default=10, help="Number of items (expand/sample)")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--source", default="", help="Default source_url (provenance)")
    sp.add_argument("--license", default="", help="Default license (provenance)")
    sp.add_argument("--origin", default="", help="Default origin (provenance)")
    sp.set_defaults(func=cmd_dataeng)

    # score
    sp = sub.add_parser("score", help="Score audio-text similarity with CLAP")
    add_common(sp)
    sp.add_argument("--path", required=True, help="WAV file to score")
    sp.add_argument("--text", required=True, help="Prompt text to compare against")
    sp.set_defaults(func=cmd_score)

    # resynth
    sp = sub.add_parser("resynth", help="Re-mix separated stems with per-stem gains")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="Source dir for messaging")
    sp.add_argument("--gain", action="append", default=None,
                    help="Stem gain, e.g. --gain vocals=1.2 --gain drums=0.5")
    sp.add_argument("--instrumental", action="store_true", help="Drop vocals entirely")
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_resynth)

    # invert
    sp = sub.add_parser("invert", help="Invert audio into a text prompt")
    add_common(sp)
    sp.add_argument("--path", required=True, help="Audio file to invert")
    sp.add_argument("--top", type=int, default=5, help="Top retrieved prompts")
    sp.set_defaults(func=cmd_invert)

    # active
    sp = sub.add_parser("active", help="Rank unlabeled tracks for labeling priority")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to rank")
    sp.add_argument("--top", type=int, default=20, help="How many to report")
    sp.set_defaults(func=cmd_active)

    # augment
    sp = sub.add_parser("augment", help="Generate audio training variants")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to augment")
    sp.add_argument("--ops", default=None,
                    help="Comma list: pitch_up,pitch_down,stretch,noise,eq,quiet")
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_augment)

    # sections
    sp = sub.add_parser("sections", help="Auto-label segments with section roles")
    add_common(sp)
    sp.add_argument("--force", action="store_true", help="Re-write labels.csv section_type")
    sp.set_defaults(func=cmd_sections)

    # drift
    sp = sub.add_parser("drift", help="Monitor feature drift reference vs current")
    add_common(sp)
    sp.add_argument("--reference", default="clean", help="Baseline data/<dir>")
    sp.add_argument("--current", default="train", help="Current data/<dir>")
    sp.add_argument("--threshold", type=float, default=0.05, help="KS p-value threshold")
    sp.set_defaults(func=cmd_drift)

    # curation
    sp = sub.add_parser("curation", help="Score tracks 0-100 for curation priority")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to score")
    sp.add_argument("--top", type=int, default=0, help="Only report top N (0 = all)")
    sp.set_defaults(func=cmd_curation)

    # embed-refresh
    sp = sub.add_parser("embed-refresh", help="Prune stale + re-embed changed audio")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to refresh")
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_embed_refresh)

    # labelprop
    sp = sub.add_parser("labelprop", help="Propagate labels to similar unlabeled tracks")
    add_common(sp)
    sp.add_argument("--dir", default="clean", help="data/<dir> to pseudo-label")
    sp.add_argument("--min-confidence", type=float, default=0.55)
    sp.add_argument("--check-leakage", action="store_true",
                    help="Check train/val/test for cross-split duplicates instead")
    sp.set_defaults(func=cmd_labelprop)

    # registry
    sp = sub.add_parser("registry", help="Index checkpoints under checkpoints/")
    add_common(sp)
    sp.set_defaults(func=cmd_registry)

    # diff
    sp = sub.add_parser("diff", help="Compare weight deltas between two checkpoints")
    add_common(sp)
    sp.add_argument("--a", required=True, help="Checkpoint dir name in checkpoints/")
    sp.add_argument("--b", required=True, help="Checkpoint dir name in checkpoints/")
    sp.add_argument("--top", type=int, default=10, help="Largest deltas to print")
    sp.set_defaults(func=cmd_diff_weights)

    # archive
    sp = sub.add_parser("archive", help="Zip a checkpoint + config + eval report")
    add_common(sp)
    sp.add_argument("--checkpoint", required=True, help="Checkpoint dir name")
    sp.set_defaults(func=cmd_archive)

    # prune
    sp = sub.add_parser("prune", help="Keep top-N checkpoints by score, archive the rest (#25)")
    add_common(sp)
    sp.add_argument("--keep", type=int, default=3, help="Number of checkpoints to keep")
    sp.add_argument("--delete", action="store_true", help="Delete instead of archiving")
    sp.set_defaults(func=cmd_prune)

    # gate
    sp = sub.add_parser("gate", help="Block checkpoint promotion on eval regression")
    add_common(sp)
    sp.add_argument("--baseline", required=True)
    sp.add_argument("--candidate", required=True)
    sp.add_argument("--max-clap-drop", type=float, default=0.02)
    sp.add_argument("--max-dev", type=float, default=0.05)
    sp.set_defaults(func=cmd_gate)

    # drift-check
    sp = sub.add_parser("drift-check", help="CI gate: fail when features drift")
    add_common(sp)
    sp.add_argument("--reference", default="clean")
    sp.add_argument("--current", default="train")
    sp.add_argument("--ks", type=float, default=0.05)
    sp.add_argument("--psi", type=float, default=0.25)
    sp.set_defaults(func=cmd_drift_check)

    # promote
    sp = sub.add_parser("promote", help="Render a promotion report for a checkpoint")
    add_common(sp)
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--baseline", default=None, help="Baseline checkpoint to diff against")
    sp.set_defaults(func=cmd_promote)

    # monitor
    sp = sub.add_parser("monitor", help="Summarize MLflow eval/inference CLAP trend")
    add_common(sp)
    sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(func=cmd_monitor)

    # matrix
    sp = sub.add_parser("matrix", help="Flatten MLflow runs into a metrics matrix")
    add_common(sp)
    sp.set_defaults(func=cmd_matrix)

    # modelcard
    sp = sub.add_parser("modelcard", help="Render a markdown model card")
    add_common(sp)
    sp.add_argument("--checkpoint", default=None, help="Checkpoint name (default: most rows)")
    sp.set_defaults(func=cmd_modelcard)

    # early-stop
    sp = sub.add_parser("early-stop", help="Decide whether to halt fine-tuning on CLAP")
    add_common(sp)
    sp.add_argument("--series", default="", help="Comma-separated CLAP history")
    sp.add_argument("--patience", type=int, default=3)
    sp.add_argument("--min-delta", type=float, default=0.005)
    sp.set_defaults(func=cmd_early_stop)

    # serve (FastAPI)
    sp = sub.add_parser("serve", help="Launch the FastAPI REST backend (#41/#42)")
    add_common(sp)
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--token", default="", help="Optional bearer token to protect the API")
    sp.set_defaults(func=cmd_serve)

    # warm-cache (#13)
    sp = sub.add_parser("warm-cache", help="Pre-pull the HF model weights into the local cache (#13)")
    add_common(sp)
    sp.add_argument("--model", default=None, help="Model id (default: cfg.inference.model_name)")
    sp.set_defaults(func=cmd_warm_cache)

    # register (MLflow registry)
    sp = sub.add_parser("register", help="Register a checkpoint in the MLflow model registry")
    add_common(sp)
    sp.add_argument("--checkpoint", required=True, help="Checkpoint dir name in checkpoints/")
    sp.add_argument("--stage", default="None", help="None | Staging | Production | Archived")
    sp.add_argument("--update", action="store_true",
                    help="Refresh the latest version with the current eval summary instead of creating a new version")
    sp.set_defaults(func=cmd_register)

    # models
    sp = sub.add_parser("models", help="List registered MLflow models + stages")
    add_common(sp)
    sp.set_defaults(func=cmd_models)

    # stage
    sp = sub.add_parser("stage", help="Move a model version between stages")
    add_common(sp)
    sp.add_argument("--name", required=True, help="Registered model name")
    sp.add_argument("--version", type=int, required=True)
    sp.add_argument("--stage", required=True, help="None | Staging | Production | Archived")
    sp.set_defaults(func=cmd_stage)

    # modelops
    sp = sub.add_parser(
        "modelops",
        help="Model-ops tasks: migrate-aliases, ab, auto-promote, lineage, "
             "lineage-graph, checksum, verify, rollback, cost-breakdown, lint "
             "(advanced #41-#46, #49-#50)",
    )
    add_common(sp)
    sp.add_argument("--task", required=True,
                    choices=["migrate-aliases", "ab", "auto-promote", "lineage",
                             "lineage-graph", "checksum", "verify", "rollback",
                             "cost-breakdown", "lint"])
    sp.add_argument("--name", default=None, help="Registered model name (aliases/rollback)")
    sp.add_argument("--champion", default=None, help="Champion checkpoint (ab)")
    sp.add_argument("--challenger", default=None, help="Challenger checkpoint (ab)")
    sp.add_argument("--candidate", default=None, help="Candidate checkpoint (auto-promote)")
    sp.add_argument("--baseline", default=None, help="Baseline checkpoint (auto-promote)")
    sp.add_argument("--parent", default=None, help="Parent checkpoint (lineage)")
    sp.add_argument("--child", default=None, help="Child checkpoint (lineage)")
    sp.add_argument("--note", default="", help="Lineage note")
    sp.add_argument("--path", default=None, help="Model dir (checksum/verify)")
    sp.add_argument("--model", default="musicgen-small", help="Model for cost-breakdown")
    sp.add_argument("--prompts", type=int, default=44)
    sp.add_argument("--seeds", type=int, default=3)
    sp.add_argument("--tokens", type=int, default=256)
    sp.set_defaults(func=cmd_modelops)

    # export-eval
    sp = sub.add_parser("export-eval", help="Push eval results to W&B (or CSV fallback) (#45)")
    add_common(sp)
    sp.add_argument("--project", default=None, help="W&B project name")
    sp.set_defaults(func=cmd_export_eval)

    # runlog
    sp = sub.add_parser("runlog", help="Tail structured JSON run log (#46)")
    add_common(sp)
    sp.add_argument("--event", default=None, help="Filter by event type")
    sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(func=cmd_runlog)

    # alert
    sp = sub.add_parser("alert", help="Alert when eval metrics cross thresholds (#47)")
    add_common(sp)
    sp.add_argument("--min-clap", type=float, default=0.30)
    sp.add_argument("--max-dev", type=float, default=0.20)
    sp.add_argument("--min-ok", type=float, default=0.5)
    sp.add_argument("--slack-webhook", default="")
    sp.add_argument("--discord-webhook", default="")
    sp.add_argument("--telegram-token", default="")
    sp.add_argument("--telegram-chat", default="")
    sp.add_argument("--smtp-host", default="")
    sp.add_argument("--smtp-user", default="")
    sp.add_argument("--smtp-password", default="")
    sp.add_argument("--smtp-to", default="")
    sp.set_defaults(func=cmd_alert)

    # backup/restore (gap #16)
    sp = sub.add_parser("backup", help="Snapshot/restore MLflow + metadata state (#16)")
    add_common(sp)
    sp.add_argument("--task", required=True, choices=["snapshot", "restore", "list"])
    sp.add_argument("--label", default="", help="Backup label (snapshot)")
    sp.add_argument("--archive", default="", help="Archive path (restore)")
    sp.add_argument("--force", action="store_true", help="Overwrite existing files (restore)")
    sp.add_argument("--no-mlflow", action="store_true", help="Exclude local MLflow state (snapshot)")
    sp.set_defaults(func=cmd_backup)

    # dataversion (gap #14 — DVC-style content-addressed dataset versioning)
    sp = sub.add_parser("dataversion", help="Content-addressed dataset versioning: commit/diff/rollback/list")
    add_common(sp)
    sp.add_argument("--task", required=True, choices=["commit", "diff", "rollback", "list"])
    sp.add_argument("--which", default="clean", help="data/<dir> to version (commit)")
    sp.add_argument("--label", default="", help="Version label (commit)")
    sp.add_argument("--v1", default="", help="First version ref (diff)")
    sp.add_argument("--v2", default="", help="Second version ref (diff)")
    sp.add_argument("--version", default="", help="Version ref to restore (rollback)")
    sp.set_defaults(func=cmd_dataversion)

    # campaign (gap #7 — blind listening campaigns)
    sp = sub.add_parser("campaign", help="Blind listening campaigns: start/record/agreement/unblind")
    add_common(sp)
    sp.add_argument("--task", required=True, choices=["start", "record", "agreement", "unblind"])
    sp.add_argument("--name", default="campaign1", help="Campaign name")
    sp.add_argument("--mode", default="ab", choices=["ab", "mos"], help="Campaign mode (start)")
    sp.add_argument("--seed", type=int, default=0, help="Shuffle seed (start)")
    sp.add_argument("--limit", type=int, default=0, help="Item limit (start)")
    sp.add_argument("--rater", default="rater1", help="Rater id (record)")
    sp.add_argument("--item", default="", help="Item id (record)")
    sp.add_argument("--choice", default="", help="X | Y | tie (record)")
    sp.add_argument("--rating", type=int, default=None, help="1-5 MOS score (record)")
    sp.add_argument("--note", default="", help="Optional note (record)")
    sp.set_defaults(func=cmd_campaign)

    # audioext (gap #20-#24)
    sp = sub.add_parser(
        "audioext",
        help="Extended audio/data: diarize, midi, augment, bundle, verify-bundle, fad-cache",
    )
    add_common(sp)
    sp.add_argument("--task", required=True,
                    choices=["diarize", "midi", "augment", "bundle", "verify-bundle", "fad-cache"])
    sp.add_argument("--path", default="", help="Audio file (diarize/midi)")
    sp.add_argument("--bpm", type=float, default=120.0, help="BPM (augment)")
    sp.add_argument("--key", default="C major", help="Key, e.g. 'A minor' (augment)")
    sp.add_argument("--semitones", type=int, default=0, help="Transpose semitones (augment)")
    sp.add_argument("--tempo-ratio", type=float, default=None, help="Tempo ratio fold (augment)")
    sp.add_argument("--which", default="clean", help="data/<dir> (bundle/fad-cache)")
    sp.add_argument("--dest", default="", help="Output archive (bundle/verify-bundle)")
    sp.add_argument("--limit", type=int, default=0, help="File limit (fad-cache)")
    sp.set_defaults(func=cmd_audioext)

    # cost
    sp = sub.add_parser("cost", help="Estimate + log run cost (#48)")
    add_common(sp)
    sp.add_argument("--estimate", default="", help="Model name to estimate only (no log)")
    sp.add_argument("--task", default="inference", help="Task label for the log")
    sp.add_argument("--model", default="musicgen-small")
    sp.add_argument("--clips", type=int, default=44, help="Number of clips")
    sp.add_argument("--tokens", type=int, default=256, help="Tokens per clip")
    sp.add_argument("--epochs", type=int, default=0)
    sp.add_argument("--lora", type=int, default=0, help="LoRA rank if fine-tuning")
    sp.set_defaults(func=cmd_cost)

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

    root = Path(getattr(args, "root", ".") or ".")
    verbose = bool(getattr(args, "verbose", False))
    quiet = bool(getattr(args, "quiet", False))
    setup_logging(root=root, verbose=verbose, quiet=quiet)
    log = get_logger("cli")

    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.error("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - last-resort guard
        # Full traceback goes to the log file (and to the console in verbose
        # mode, where the handler is at DEBUG level).
        log.exception("command %r failed", getattr(args, "command", "?"))
        console.error(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
