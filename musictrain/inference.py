"""MusicGen text-conditional inference on Apple Silicon (MPS) with CPU fallback.

Phase 5 additions:
* #33  negative prompting — CLAP-scored "no X" constraints with auto-retry
* #34  batch prompt files — plain lines or JSONL with per-item options
* #35  continuation — condition on an existing audio clip (input_values)
* #36  melody conditioning — same mechanism, intended for musicgen-melody
* #37  sampling presets — standard/creative/precise knob sets
* #38  repro manifest — every run pins config/vocab/git state
* #39  target seconds — derive max_new_tokens from a desired duration
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import console
from .config import Config, InferenceCfg
from .logging import get_logger
from .util import now_stamp, sanitize_slug

log = get_logger("inference")

TOKENS_PER_SECOND = 50  # MusicGen codec rate at 32 kHz


def resolve_device(preferred: str) -> str:
    import torch

    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def load_model(cfg: InferenceCfg) -> Tuple:
    import torch
    from transformers import AutoModelForTextToWaveform, AutoProcessor

    device = resolve_device(cfg.device)
    # advanced #11 — dtype control (float32 | float16 | bf16)
    dtype = torch.float32
    if cfg.torch_dtype == "float16":
        dtype = torch.float16
    elif cfg.torch_dtype == "bf16":
        dtype = torch.bfloat16

    console.info(f"Loading {cfg.model_name} on {device} (dtype={cfg.torch_dtype})…")
    # AutoModelForTextToWaveform dispatches to MusicgenForConditionalGeneration
    # for small/medium and MusicgenMelodyForConditionalGeneration for the
    # melody checkpoint (base class would leave encoder_attn uninitialized).
    model = AutoModelForTextToWaveform.from_pretrained(
        cfg.model_name, torch_dtype=dtype
    ).to(device)
    processor = AutoProcessor.from_pretrained(cfg.model_name)
    model = _load_adapter(model, cfg, device)
    return processor, model, device


def _load_adapter(model, cfg: InferenceCfg, device: str):
    """Load a LoRA adapter dir onto the base model, if configured (gap #1).

    ``cfg.adapter`` points at a directory produced by ``musictrain finetune``
    (a PeftModel saved via ``model.decoder.save_pretrained``). Returns the base
    model unchanged when no adapter is set or the dir is missing/not a PEFT
    checkpoint, so inference never hard-fails on a bad adapter path.
    """
    if not cfg.adapter:
        return model
    adir = Path(cfg.adapter)
    if not adir.exists():
        console.warn(f"adapter dir not found: {adir} — falling back to base model")
        log.warning("adapter dir %s does not exist; using base model", adir)
        return model
    try:
        from peft import PeftModel
    except ImportError:
        console.error("Loading adapters needs `pip install peft`.")
        return model
    try:
        model.decoder = PeftModel.from_pretrained(model.decoder, adir)
        model.decoder = model.decoder.to(device)
        console.ok(f"Loaded adapter -> {adir}")
        log.info("loaded LoRA adapter from %s", adir)
    except Exception as exc:  # noqa: BLE001 - never hard-fail generation
        console.warn(f"adapter load failed ({exc}) — using base model")
        log.warning("adapter load failed: %s", exc)
    return model


# --------------------------------------------------------------------------- #
# advanced #20 — deterministic generation cache
# --------------------------------------------------------------------------- #
_CACHE_FILE = "metadata/generation_cache.json"


def _cache_key(cfg: Config, prompt: str, seed: Optional[int],
               cond_path: Optional[Path] = None,
               conditioning_kind: Optional[str] = None) -> str:
    icfg = cfg.inference
    parts = [
        icfg.model_name, prompt, str(seed or ""), str(icfg.guidance_scale),
        str(icfg.max_new_tokens), icfg.preset or "", str(icfg.temperature),
        str(icfg.top_k), str(icfg.top_p), str(cond_path or ""),
        conditioning_kind or "", str(icfg.negative_prompt or ""),
        str(icfg.adapter or ""),
    ]
    import hashlib

    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()


def _load_cache(cfg: Config) -> dict:
    p = cfg.project_root / _CACHE_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 - corrupt cache -> start fresh
        return {}


def _save_cache(cfg: Config, cache: dict) -> None:
    p = cfg.project_root / _CACHE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2))


def generate_cached(cfg: Config, prompt: str, out_dir: Optional[Path] = None,
                    seed: Optional[int] = None, use_cache: bool = True,
                    melody_from: Optional[Path] = None,
                    continue_from: Optional[Path] = None,
                    **kwargs) -> dict:
    """generate() with a deterministic content-addressed cache.

    Same model + prompt + sampling settings + seed + conditioning -> same file
    (the cache key is a hash of all of them). Cached hits skip the model
    entirely and return a result dict with the stored path.
    """
    import soundfile as sf

    cond_path = Path(melody_from) if melody_from else (
        Path(continue_from) if continue_from else None
    )
    kind = "melody" if melody_from else ("continuation" if continue_from else None)
    key = _cache_key(cfg, prompt, seed, cond_path, kind)

    cache = _load_cache(cfg) if use_cache else {}
    hit = cache.get(key)
    if use_cache and hit and Path(hit).exists():
        console.info(f"[cache] hit {Path(hit).name} for {prompt[:40]!r}")
        info = sf.info(hit)
        return {
            "path": hit, "prompt": prompt, "seed": seed,
            "device": "cache", "sample_rate": int(info.samplerate),
            "duration": round(float(info.duration), 3),
            "shape": [1, int(info.frames)],
            "max_new_tokens": cfg.inference.max_new_tokens,
            "target_seconds": None, "conditioned_on": str(cond_path) if cond_path else None,
            "conditioning_kind": kind, "cached": True,
        }

    result = generate(
        cfg, prompt, out_dir=out_dir, seed=seed,
        melody_from=melody_from, continue_from=continue_from, **kwargs,
    )
    if result and use_cache:
        cache[key] = result["path"]
        _save_cache(cfg, cache)
    return result


def _extract_audio(out):
    """Return the audio_values tensor from a generate() result across versions."""
    audio_values = getattr(out, "audio_values", None)
    if audio_values is None and isinstance(out, dict):
        audio_values = out.get("audio_values")
    if audio_values is None:
        audio_values = out  # plain tensor (default return_dict_in_generate=False)
    return audio_values


def _sample_rate(model) -> int:
    """Recover the audio sample rate from the model config, with fallbacks."""
    try:
        return int(model.config.audio_encoder.sampling_rate)
    except Exception as exc:  # noqa: BLE001
        log.debug("no sampling_rate on model config (%s); trying codec_sample_rate", exc)
    try:
        return int(model.config.audio_encoder.codec_sample_rate)
    except Exception as exc:  # noqa: BLE001
        log.debug("no codec_sample_rate on model config (%s); defaulting to 32 kHz", exc)
    return 32000


# --------------------------------------------------------------------------- #
# Phase 5 helpers
# --------------------------------------------------------------------------- #


def _sampling_kwargs(icfg: InferenceCfg, preset: Optional[str] = None) -> Dict[str, Any]:
    """Build generate() sampling kwargs, optionally from a named preset (#37)."""
    kw: Dict[str, Any] = dict(
        do_sample=icfg.do_sample,
        guidance_scale=icfg.guidance_scale,
        max_new_tokens=icfg.max_new_tokens,
    )
    if icfg.do_sample:
        kw["temperature"] = icfg.temperature
        kw["top_k"] = icfg.top_k
        kw["top_p"] = icfg.top_p

    if preset:
        p = icfg.presets.get(preset)
        if not p:
            console.error(
                f"Unknown preset {preset!r} — available: {', '.join(sorted(icfg.presets))}"
            )
            raise KeyError(preset)
        kw["do_sample"] = True
        kw["guidance_scale"] = float(p.get("guidance_scale", kw["guidance_scale"]))
        if "temperature" in p:
            kw["temperature"] = float(p["temperature"])
        if "top_k" in p:
            kw["top_k"] = int(p["top_k"])
        if "top_p" in p:
            kw["top_p"] = float(p["top_p"])
    return kw


def _conditioning_audio(model, path: Path, device: str):
    """Load + resample a conditioning clip to the model's audio rate -> [1, 1, T]"""
    import librosa
    import numpy as np
    import soundfile as sf
    import torch

    sr_target = getattr(model.config.audio_encoder, "sampling_rate", None) or 32000
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != sr_target:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_target)
    return torch.from_numpy(audio)[None, None].to(device)  # [batch, channels, samples]


def _clap_score(cfg: Config, path: Path, text: str) -> Optional[float]:
    """CLAP similarity between a wav and the negative text (None if disabled/failed)."""
    if not cfg.clap.enabled:
        return None
    from .similarity import score

    try:
        return score(cfg, path, text)
    except Exception as exc:  # noqa: BLE001 - negative scoring must not kill generation
        console.warn(f"CLAP negative scoring failed: {exc}")
        return None


def _gen_audio(
    cfg: Config,
    icfg: InferenceCfg,
    processor,
    model,
    device: str,
    prompt: str,
    seed: Optional[int],
    gen_kwargs: Dict[str, Any],
    input_values=None,
) -> Tuple[Any, int, str]:
    """Run model.generate once, with MPS->CPU fallback. Returns (audio, sr, device)."""
    import torch

    if seed is not None:
        torch.manual_seed(seed)

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    if input_values is not None:
        inputs["input_values"] = input_values

    console.info(f"Generating: {prompt!r} (device={device})")
    try:
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
    except (RuntimeError, NotImplementedError) as exc:
        if str(device) != "cpu":
            console.warn(f"{device} failed ({exc}); retrying on CPU")
            del model
            icfg.device = "cpu"
            processor, model, device = load_model(icfg)
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
            if input_values is not None:
                inputs["input_values"] = input_values.to(device)
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
        else:
            raise

    audio = _extract_audio(out)[0].cpu().numpy()
    return audio, _sample_rate(model), device


def generate(
    cfg: Config,
    prompt: str,
    out_dir: Optional[Path] = None,
    name: Optional[str] = None,
    seed: Optional[int] = None,
    processor=None,
    model=None,
    device: Optional[str] = None,
    # -- Phase 5 (#33-#39) --
    negative_prompt: Optional[str] = None,
    negative_retries: Optional[int] = None,
    target_seconds: Optional[float] = None,
    continue_from: Optional[Path] = None,
    melody_from: Optional[Path] = None,
    preset: Optional[str] = None,
    manifest: bool = True,
) -> dict:
    import soundfile as sf

    icfg = cfg.inference
    loaded = processor is None or model is None
    if loaded:
        processor, model, device = load_model(icfg)
    assert processor is not None and model is not None

    if seed is not None:
        icfg.seed = seed

    gen_kwargs = _sampling_kwargs(icfg, preset=preset or icfg.preset)
    if target_seconds is None:
        target_seconds = icfg.target_seconds
    if target_seconds:
        gen_kwargs["max_new_tokens"] = max(8, int(target_seconds * TOKENS_PER_SECOND))

    # conditioning (#35 continuation / #36 melody)
    conditioning_kind = None
    cond_path: Optional[Path] = None
    if continue_from or melody_from:
        if continue_from and melody_from:
            console.error("Provide only one of --continue-from / --melody-from.")
            return {}
        cond_path = Path(continue_from or melody_from).resolve()
        conditioning_kind = "melody" if melody_from else "continuation"
        if melody_from and "musicgen-melody" not in icfg.model_name:
            console.warn(
                "--melody-from is intended for facebook/musicgen-melody; "
                "with other models it behaves like continuation."
            )
        input_values = _conditioning_audio(model, cond_path, device)
    else:
        input_values = None

    neg = negative_prompt if negative_prompt is not None else (icfg.negative_prompt or None)
    retries = negative_retries if negative_retries is not None else icfg.negative_retries

    out_dir = Path(out_dir) if out_dir else cfg.project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_slug(name or prompt[:40])

    # -- generation loop with negative-prompt retries (#33) ------------------
    audio = None
    neg_score: Optional[float] = None
    violation = False
    attempts = 0
    while True:
        attempts += 1
        audio, sr, device = _gen_audio(
            cfg, icfg, processor, model, device, prompt, seed, gen_kwargs, input_values
        )
        if not neg:
            break
        tmp = out_dir / f".negcheck_{now_stamp()}.wav"
        sf.write(tmp, audio.T, sr)
        neg_score = _clap_score(cfg, tmp, neg)
        tmp.unlink(missing_ok=True)
        violation = neg_score is not None and neg_score >= icfg.negative_threshold
        if not violation or attempts > retries:
            break
        console.warn(
            f"Negative-prompt violation (CLAP {neg_score:.3f} >= {icfg.negative_threshold}); "
            f"retrying ({attempts}/{retries + 1})"
        )
        if seed is not None:
            seed += 1

    out_path = out_dir / f"{slug}_{now_stamp()}.wav"
    sf.write(out_path, audio.T, sr)

    result = {
        "path": str(out_path),
        "prompt": prompt,
        "seed": seed,
        "device": device,
        "sample_rate": sr,
        "duration": round(audio.shape[-1] / sr, 3),
        "shape": list(audio.shape),
        # -- Phase 5 --
        "preset": preset or icfg.preset or None,
        "max_new_tokens": gen_kwargs["max_new_tokens"],
        "target_seconds": round(target_seconds, 3) if target_seconds else None,
        "conditioned_on": str(cond_path) if cond_path else None,
        "conditioning_kind": conditioning_kind,
        "negative_prompt": neg,
        "negative_clap": round(neg_score, 4) if neg_score is not None else None,
        "negative_violation": violation if neg else None,
        "attempts": attempts,
    }
    console.ok(f"Saved {out_path} ({result['duration']}s)")
    if neg:
        console.info(
            f"negative CLAP {result['negative_clap']} "
            f"({'violation' if violation else 'clean'}, {attempts} attempt(s))"
        )

    if manifest:
        from .reproduce import capture_run

        capture_run(
            cfg,
            "inference",
            extra={
                "prompt": prompt,
                "seed": seed,
                "preset": result["preset"],
                "max_new_tokens": gen_kwargs["max_new_tokens"],
                "target_seconds": result["target_seconds"],
                "conditioned_on": result["conditioned_on"],
                "conditioning_kind": conditioning_kind,
                "negative_prompt": neg,
                "negative_violation": violation if neg else None,
                "attempts": attempts,
                "duration": result["duration"],
            },
        )

    if loaded:
        del model
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def generate_batch(
    cfg: Config,
    prompts: List[Any],
    out_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> List[dict]:
    """Batch generation over plain prompts or JSONL items with per-item options (#34)."""
    items: List[dict] = [p if isinstance(p, dict) else {"prompt": p} for p in prompts]
    processor, model, device = load_model(cfg.inference)
    results = []
    for i, it in enumerate(items):
        s = it.get("seed", seed + i if seed is not None else None)
        results.append(
            generate(
                cfg,
                it["prompt"],
                out_dir=out_dir,
                name=f"batch_{i:03d}",
                seed=s,
                processor=processor,
                model=model,
                device=device,
                negative_prompt=it.get("negative_prompt"),
                negative_retries=it.get("negative_retries"),
                target_seconds=it.get("target_seconds"),
                continue_from=Path(it["continue_from"]) if it.get("continue_from") else None,
                melody_from=Path(it["melody_from"]) if it.get("melody_from") else None,
                preset=it.get("preset"),
            )
        )
    del model
    return results
