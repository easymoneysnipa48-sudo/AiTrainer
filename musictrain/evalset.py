"""Fixed evaluation prompt set + batch eval runner.

Produces a deterministic, version-controlled set of prompts spanning sections,
BPMs, and keys (plus out-of-distribution prompts), then runs batch inference,
BPM post-checks, and MLflow logging against it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

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


def build(root: Path, force: bool = False) -> List[dict]:
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


def run_eval(
    cfg: Config,
    limit: int = 0,
    check_bpm: bool = True,
    out_dir: Optional[Path] = None,
    section: Optional[str] = None,
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
        prompts = [p for p in prompts if p.get("section") == section]
        if not prompts:
            console.error(f"No prompts for section {section!r}")
            return []
    if limit:
        prompts = prompts[:limit]

    out_dir = Path(out_dir) if out_dir else cfg.project_root / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, model, device = load_model(cfg.inference)
    results: List[dict] = []
    console.step(f"Running eval over {len(prompts)} prompts (device={device})")

    for p in prompts:
        result = generate(
            cfg,
            p["description"],
            out_dir=out_dir,
            name=p["id"],
            seed=p["seed"],
            processor=processor,
            model=model,
            device=device,
        )
        entry = {
            "experiment_id": cfg.mlflow.experiment_name,
            "checkpoint": cfg.inference.model_name,
            "prompt": p["description"],
            "seed": p["seed"],
            "section": p.get("section"),
            "genre": p.get("genre"),
            "key": p.get("key"),
            "bpm_target": p.get("bpm"),
            "audio_path": result["path"],
            "detected_bpm": None,
            "deviation": None,
            "clap_score": None,
            "human_rating": None,
            "status": None,
            "notes": "",
        }
        if check_bpm and p.get("bpm"):
            report = check(cfg, Path(result["path"]), target_bpm=float(p["bpm"]))
            entry["detected_bpm"] = report.get("detected_bpm")
            entry["deviation"] = report.get("deviation")
            entry["status"] = report.get("status")
            entry["notes"] = report.get("note", "")
            log_eval(cfg, report)

        clap_score = None
        if cfg.clap.enabled:
            try:
                clap_score = score(cfg, Path(result["path"]), p["description"])
                entry["clap_score"] = clap_score
            except Exception as exc:  # noqa: BLE001 - scoring must not break eval
                console.warn(f"CLAP scoring failed for {p['id']}: {exc}")
        log_inference(cfg, result, clap_score=clap_score)
        results.append(entry)

    del model

    out = cfg.project_root / "metadata" / "eval_results.jsonl"
    with jsonlines.open(out, mode="w") as w:
        for r in results:
            w.write(r)

    ok = sum(1 for r in results if r.get("status") in ("ok",))
    console.ok(
        f"Eval complete: {len(results)} runs -> {out.relative_to(cfg.project_root)} "
        f"({ok} BPM in-tolerance)"
    )
    return results
