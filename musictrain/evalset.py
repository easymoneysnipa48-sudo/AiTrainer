"""Fixed evaluation prompt set + batch eval runner.

Produces a deterministic, version-controlled set of prompts spanning sections,
BPMs, and keys (plus out-of-distribution prompts), then runs batch inference,
BPM post-checks, and MLflow logging against it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import jsonlines

from . import console
from .config import Config

# Per-section prompt phrasing. `instruments`/`mood` are controlled-vocabulary
# terms (for bookkeeping); `phrasing` + `tail` build the conditioning text.
SECTIONS = {
    "intro": {
        "instruments": ["piano", "pads"],
        "mood": ["dark", "atmospheric"],
        "energy": 0.3,
        "phrasing": "dark piano loop, atmospheric pads",
        "tail": "low energy",
    },
    "verse": {
        "instruments": ["guitar loop", "808 bass", "snare"],
        "mood": ["dark", "emotional"],
        "energy": 0.55,
        "phrasing": "restrained snare, deep 808 bass",
        "tail": "narrative mood, medium energy",
    },
    "pre-chorus": {
        "instruments": ["strings", "trap hi-hats", "snare"],
        "mood": ["tense", "determined"],
        "energy": 0.7,
        "phrasing": "building strings, driving snare, trap hi-hats",
        "tail": "building tension",
    },
    "chorus": {
        "instruments": ["piano", "808 bass", "autotune vocals", "trap hi-hats"],
        "mood": ["dark", "emotional", "aggressive"],
        "energy": 0.9,
        "phrasing": "heavy 808 bass, autotune vocals, trap hi-hats",
        "tail": "memorable melodic hook, high energy",
    },
    "bridge": {
        "instruments": ["piano", "strings"],
        "mood": ["reflective", "melancholic"],
        "energy": 0.45,
        "phrasing": "stripped-back piano, soft strings",
        "tail": "reflective mood",
    },
    "outro": {
        "instruments": ["piano", "pads"],
        "mood": ["reflective", "calm"],
        "energy": 0.25,
        "phrasing": "fading piano, reduced drums",
        "tail": "fading out, reflective mood",
    },
}

# Trap tempo conventions: half-time (NBA YoungBoy / pain music) 72-80,
# mid melodic trap 84-96, standard trap 130-140, bounce (Quavo) ~155.
BPMs = [72, 78, 84, 96, 130, 140, 155]
KEYS = ["A minor", "C minor", "F minor", "E minor", "D minor"]

OUT_OF_DISTRIBUTION = [
    {
        "id": "ood_ambient_060",
        "section": "full-song",
        "genre": "ambient",
        "bpm": 60,
        "key": "G minor",
        "mood": ["dreamy", "calm"],
        "instruments": ["pads", "flute"],
        "energy": 0.2,
        "description": "ambient full-song, 60 BPM, G minor, slow pads, airy flute, dreamy calm atmosphere",
    },
    {
        "id": "ood_orchestral_100",
        "section": "full-song",
        "genre": "orchestral",
        "bpm": 100,
        "key": "D major",
        "mood": ["epic", "uplifting"],
        "instruments": ["strings", "brass", "choir"],
        "energy": 0.8,
        "description": "orchestral full-song, 100 BPM, D major, sweeping strings, heroic brass, epic choir",
    },
]


def _key_slug(key: str) -> str:
    return key.lower().replace(" ", "")


def adversarial_prompts(root: Path, n: int = 10, start_seed: int = 1000) -> List[dict]:
    """Advanced #10 — deliberately tricky prompts to stress prompt adherence.

    Categories: impossible BPM for the section, conflicting/ambiguous tags,
    out-of-vocabulary descriptors, and near-empty phrasing. Each carries
    ``adversarial: true`` so eval/report code can flag them.
    """
    traps = [
        ("full-song", 42, "melodic trap", "A minor", "fast chaotic full song, 42 BPM, breakcore drums, A minor"),
        ("intro", 220, "ambient", "F# major", "intro, 220 BPM, ambient pads, F# major, breakneck hi-hats"),
        ("chorus", 55, "melodic trap", "C minor", "chorus, 55 BPM, melodic trap, C minor, no drums, no bass"),
        ("verse", 96, "orchestral", "B minor", "verse, 96 BPM, orchestral strings, B minor, trap hi-hats, aggressive"),
        ("bridge", 140, "lofi", "D major", "bridge, 140 BPM, lofi, D major, distorted 808, gentle piano"),
        ("outro", 200, "gospel", "E minor", "outro, 200 BPM, gospel choir, E minor, half-time feel"),
        ("pre-chorus", 60, "techno", "G minor", "pre-chorus, 60 BPM, techno, G minor, kick every bar, dreamy pads"),
        ("chorus", 96, "melodic trap", "A minor", "chorus with absolutely no describable instrumentation, 96 BPM"),
        ("verse", 140, "melodic trap", "B minor", "blorp glorp zorp splat, 140 BPM, B minor, reverse piano, sidechain everything"),
        ("full-song", 84, "ambient", "C# minor", "sparse, 84 BPM, C# minor"),
        ("intro", 70, "melodic trap", "A minor", "intro, 70 BPM, A minor, quiet then extremely loud"),
        ("chorus", 128, "orchestral", "D minor", "chorus, 128 BPM, D minor, orchestra playing trap rhythms"),
    ]
    out: List[dict] = []
    for i, (section, bpm, genre, key, desc) in enumerate(traps[:n]):
        out.append(
            {
                "id": f"adv_{i:02d}",
                "section": section,
                "genre": genre,
                "bpm": bpm,
                "key": key,
                "mood": "",
                "instruments": "",
                "energy": 0.5,
                "seed": start_seed + i,
                "adversarial": True,
                "description": desc,
            }
        )
    return out


def build(root: Path, force: bool = False, adversarial: int = 0) -> List[dict]:
    prompts: List[dict] = []
    i = 0
    for section, tmpl in SECTIONS.items():
        for j, bpm in enumerate(BPMs):
            key = KEYS[j % len(KEYS)]
            prompts.append(
                {
                    "id": f"{section}_{bpm:03d}_{_key_slug(key)}",
                    "section": section,
                    "genre": "melodic trap",
                    "bpm": bpm,
                    "key": key,
                    "mood": tmpl["mood"],
                    "instruments": tmpl["instruments"],
                    "energy": tmpl["energy"],
                    "seed": 42 + i,
                    "description": (
                        f"{section}, {bpm} BPM, {key}, {tmpl['phrasing']}, {tmpl['tail']}"
                    ),
                }
            )
            i += 1

    for ood in OUT_OF_DISTRIBUTION:
        ood = dict(ood)
        ood["seed"] = 42 + i
        prompts.append(ood)
        i += 1

    out = root / "metadata" / "eval_prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        console.warn(f"{out.relative_to(root)} exists; use --force to overwrite")
        return load(root)

    if adversarial:
        adv = adversarial_prompts(root, n=adversarial, start_seed=1000)
        prompts.extend(adv)
        console.step(f"Appending {len(adv)} adversarial prompt(s)")

    with jsonlines.open(out, mode="w") as w:
        for p in prompts:
            w.write(p)
    console.ok(f"Wrote {len(prompts)} eval prompts -> {out.relative_to(root)}")
    return prompts


def load(root: Path) -> List[dict]:
    path = root / "metadata" / "eval_prompts.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def tag_phrases(prompt: dict) -> dict:
    """Per-tag text phrases for #46 (per-tag CLAP adherence)."""
    phrases = {}
    if prompt.get("section"):
        phrases["section"] = prompt["section"]
    if prompt.get("genre"):
        phrases["genre"] = prompt["genre"]
    if prompt.get("key"):
        phrases["key"] = prompt["key"]
    moods = prompt.get("mood") or []
    if moods:
        phrases["mood"] = ", ".join(moods) if isinstance(moods, list) else str(moods)
    instrs = prompt.get("instruments") or []
    if instrs:
        phrases["instruments"] = ", ".join(instrs) if isinstance(instrs, list) else str(instrs)
    if prompt.get("bpm"):
        phrases["bpm"] = f"{int(prompt['bpm'])} BPM"
    return phrases


def _aggregate(prompt: dict, seed_records: List[dict], cfg: Config) -> dict:
    import statistics

    detected = [r["detected_bpm"] for r in seed_records if r.get("detected_bpm") is not None]
    claps = [r["clap_score"] for r in seed_records if r.get("clap_score") is not None]
    oks = sum(1 for r in seed_records if r.get("status") == "ok")
    n = len(seed_records)

    med_det = statistics.median(detected) if detected else None
    target = prompt.get("bpm")
    deviation = None
    if med_det is not None and target:
        deviation = round((med_det - float(target)) / float(target), 4)

    mean_clap = round(sum(claps) / len(claps), 4) if claps else None

    # -- #43: auto-reject thresholds -------------------------------------
    reasons: List[str] = []
    if cfg.eval.min_clap_score > 0 and mean_clap is not None \
            and mean_clap < cfg.eval.min_clap_score:
        reasons.append(f"CLAP {mean_clap:.3f} < min {cfg.eval.min_clap_score:.3f}")
    if cfg.eval.max_abs_deviation > 0 and deviation is not None \
            and abs(deviation) > cfg.eval.max_abs_deviation:
        reasons.append(
            f"|dev| {abs(deviation):.3f} > max {cfg.eval.max_abs_deviation:.3f}"
        )

    bpm_ok = oks >= (n + 1) // 2
    status = "ok" if bpm_ok and not reasons else "rejected"
    notes = f"{oks}/{n} seeds in-tolerance" if n > 1 else ""
    if reasons:
        notes = (notes + "; " if notes else "") + "auto-reject: " + ", ".join(reasons)

    # -- #46: per-tag CLAP (mean across seeds) ----------------------------
    clap_per_tag = None
    tag_lists = [
        (r.get("clap_per_tag") or {}) for r in seed_records
        if r.get("clap_per_tag")
    ]
    if tag_lists:
        keys = sorted({k for t in tag_lists for k in t})
        clap_per_tag = {}
        for k in keys:
            vals = [t[k] for t in tag_lists if k in t]
            if vals:
                clap_per_tag[k] = round(sum(vals) / len(vals), 4)

    return {
        "experiment_id": cfg.mlflow.experiment_name,
        "checkpoint": cfg.inference.model_name,
        "prompt": prompt["description"],
        "seed": prompt.get("seed"),
        "n_seeds": n,
        "ok_seeds": oks,
        "section": prompt.get("section"),
        "genre": prompt.get("genre"),
        "key": prompt.get("key"),
        "bpm_target": target,
        "audio_path": seed_records[0]["audio_path"] if seed_records else None,
        "detected_bpm": round(med_det, 2) if med_det is not None else None,
        "deviation": deviation,
        "clap_score": mean_clap,
        "clap_per_tag": clap_per_tag,
        "human_rating": None,
        "status": status,
        "notes": notes,
        "per_seed": seed_records,
    }


def run_eval(
    cfg: Config,
    limit: int = 0,
    check_bpm: bool = True,
    out_dir: Optional[Path] = None,
    section: Optional[str] = None,
    seeds: int = 1,
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
    incremental: bool = False,
) -> List[dict]:
    from .evaluate import check
    from .experiments import log_eval, log_inference
    from .inference import generate, load_model
    from .similarity import score

    prompts = load(cfg.project_root)
    if not prompts:
        console.error("No eval prompts found — run `musictrain evalset` first.")
        return []
    if section:
        wanted = {s.strip() for s in section.split(",") if s.strip()}
        prompts = [p for p in prompts if p.get("section") in wanted]
        if not prompts:
            console.error(f"No prompts for section(s) {section!r}")
            return []
    if limit:
        prompts = prompts[:limit]

    # -- Advanced #49: incremental eval ----------------------------------
    # Keep already-passing rows for this checkpoint; re-run only prompts that
    # failed, rejected, or are new. Existing rows are preserved on write.
    kept: List[dict] = []
    if incremental:
        from .report import load_results

        existing = load_results(cfg.project_root)
        passed = {
            (r.get("checkpoint"), r.get("prompt"))
            for r in existing
            if r.get("status") == "ok"
        }
        kept = [r for r in existing if (r.get("checkpoint"), r.get("prompt")) in passed]
        before = len(prompts)
        prompts = [
            p
            for p in prompts
            if (cfg.inference.model_name, p["description"]) not in passed
        ]
        console.info(
            f"Incremental: {len(kept)} passed row(s) kept, "
            f"re-running {len(prompts)} of {before} prompt(s)"
        )
        if not prompts:
            console.ok("Nothing to re-run — all prompts already pass for this checkpoint.")
            return kept

    out_dir = Path(out_dir) if out_dir else cfg.project_root / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, model, device = load_model(cfg.inference)
    results: List[dict] = []
    console.step(f"Running eval over {len(prompts)} prompts x {seeds} seed(s) (device={device})")

    total = len(prompts)
    for done, p in enumerate(prompts, 1):
        if cancel and cancel():
            console.warn("Eval cancelled by user.")
            break
        seed_records: List[dict] = []
        for i in range(seeds):
            seed = p["seed"] + i
            result = generate(
                cfg,
                p["description"],
                out_dir=out_dir,
                name=f"{p['id']}_s{seed}",
                seed=seed,
                processor=processor,
                model=model,
                device=device,
            )
            rec = {
                "seed": seed,
                "audio_path": result["path"],
                "detected_bpm": None,
                "deviation": None,
                "clap_score": None,
                "status": None,
                "note": "",
            }
            if check_bpm and p.get("bpm"):
                report = check(cfg, Path(result["path"]), target_bpm=float(p["bpm"]))
                rec["detected_bpm"] = report.get("detected_bpm")
                rec["deviation"] = report.get("deviation")
                rec["status"] = report.get("status")
                rec["note"] = report.get("note", "")
                log_eval(cfg, report)

            clap_score = None
            if cfg.clap.enabled:
                try:
                    clap_score = score(cfg, Path(result["path"]), p["description"])
                    rec["clap_score"] = clap_score
                    if cfg.eval.per_tag_clap:
                        from .similarity import score_multi

                        rec["clap_per_tag"] = score_multi(
                            cfg, Path(result["path"]), tag_phrases(p)
                        )
                except Exception as exc:  # noqa: BLE001 - scoring must not break eval
                    console.warn(f"CLAP scoring failed for {p['id']} seed {seed}: {exc}")
            log_inference(cfg, result, clap_score=clap_score)
            seed_records.append(rec)

        results.append(_aggregate(p, seed_records, cfg))
        if progress:
            progress(done, total)

    del model

    out = cfg.project_root / "metadata" / "eval_results.jsonl"
    with jsonlines.open(out, mode="w") as w:
        for r in (kept + results) if incremental else results:
            w.write(r)

    ok = sum(1 for r in results if r.get("status") == "ok")
    total_written = len(kept) + len(results) if incremental else len(results)
    console.ok(
        f"Eval complete: {total_written} prompts -> {out.relative_to(cfg.project_root)} "
        f"({ok} BPM in-tolerance by majority, {len(kept)} kept from prior runs)"
    )

    from .reproduce import capture_run

    capture_run(
        cfg,
        "eval",
        extra={
            "n_prompts": len(results),
            "seeds": seeds,
            "section": section or "",
            "limit": limit or 0,
            "ok_majority": ok,
            "results_file": "metadata/eval_results.jsonl",
        },
    )
    return results
