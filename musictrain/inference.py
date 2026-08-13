"""MusicGen text-conditional inference on Apple Silicon (MPS) with CPU fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from . import console
from .config import Config, InferenceCfg
from .util import now_stamp, sanitize_slug


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
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = resolve_device(cfg.device)
    dtype = torch.float32
    if device == "cuda" and cfg.torch_dtype == "float16":
        dtype = torch.float16

    console.info(f"Loading {cfg.model_name} on {device} (dtype={cfg.torch_dtype})…")
    model = MusicgenForConditionalGeneration.from_pretrained(
        cfg.model_name, torch_dtype=dtype
    ).to(device)
    processor = AutoProcessor.from_pretrained(cfg.model_name)
    return processor, model, device


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
    except Exception:  # noqa: BLE001
        pass
    try:
        return int(model.config.audio_encoder.codec_sample_rate)
    except Exception:  # noqa: BLE001
        pass
    return 32000


def generate(
    cfg: Config,
    prompt: str,
    out_dir: Optional[Path] = None,
    name: Optional[str] = None,
    seed: Optional[int] = None,
    processor=None,
    model=None,
    device: Optional[str] = None,
) -> dict:
    import torch
    import soundfile as sf

    icfg = cfg.inference
    loaded = processor is None or model is None
    if loaded:
        processor, model, device = load_model(icfg)
    assert processor is not None and model is not None

    if seed is not None:
        torch.manual_seed(seed)
        icfg.seed = seed

    inputs = processor(
        text=[prompt], padding=True, return_tensors="pt"
    ).to(device)

    gen_kwargs = dict(
        do_sample=icfg.do_sample,
        guidance_scale=icfg.guidance_scale,
        max_new_tokens=icfg.max_new_tokens,
    )
    if icfg.do_sample:
        gen_kwargs["temperature"] = icfg.temperature
        gen_kwargs["top_k"] = icfg.top_k

    console.info(f"Generating: {prompt!r} (device={device})")
    try:
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
    except (RuntimeError, NotImplementedError) as exc:
        if loaded and str(device) != "cpu":
            console.warn(f"{device} failed ({exc}); retrying on CPU")
            del model
            icfg.device = "cpu"
            processor, model, device = load_model(icfg)
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
        else:
            raise

    audio = _extract_audio(out)[0].cpu().numpy()
    sr = _sample_rate(model)

    out_dir = Path(out_dir) if out_dir else cfg.project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_slug(name or prompt[:40])
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
    }
    console.ok(f"Saved {out_path} ({result['duration']}s)")
    if loaded:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def generate_batch(
    cfg: Config,
    prompts: list[str],
    out_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> list[dict]:
    processor, model, device = load_model(cfg.inference)
    results = []
    for i, prompt in enumerate(prompts):
        results.append(
            generate(
                cfg, prompt, out_dir=out_dir, name=f"batch_{i:03d}",
                seed=seed + i if seed is not None else None,
                processor=processor, model=model, device=device,
            )
        )
    del model
    return results
