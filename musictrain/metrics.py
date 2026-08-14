"""Distribution metrics between reference and generated audio (#41).

Two complementary scores:

* **FAD** — Frechet Audio Distance computed on CLAP audio embeddings. The
  classic FAD uses VGGish; here we substitute the project's existing CLAP
  embedding pipeline (already cached in metadata/audio_embeddings.json) so no
  extra model is downloaded. Lower = generated distribution closer to real.

* **Spectral KL** — mean KL divergence between the normalized log-mel
  spectrogram distributions of the two sets. Pure librosa/numpy, no model
  needed. Lower = closer spectral texture.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import console
from .audio.inventory import AUDIO_GLOB
from .config import Config


# --------------------------------------------------------------------------- #
# FAD on CLAP embeddings
# --------------------------------------------------------------------------- #

def frechet_distance(mu1: np.ndarray, cov1: np.ndarray,
                     mu2: np.ndarray, cov2: np.ndarray) -> float:
    """Frechet distance between two multivariate Gaussians (FAD core)."""
    diff = mu1 - mu2
    # sqrtm of cov1 @ cov2; guard singular/ill-conditioned inputs
    try:
        from scipy.linalg import sqrtm

        covmean = sqrtm(cov1 @ cov2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean))
    except Exception:  # noqa: BLE001 - degenerate covariances -> diagonal approx
        return float(diff @ diff + np.trace(cov1 + cov2))


def _gaussian(embs: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    embs = np.asarray(embs, dtype=np.float64)
    mu = embs.mean(axis=0)
    cov = np.cov(embs, rowvar=False)
    if cov.ndim != 2:
        cov = np.atleast_2d(cov)
    # regularize the diagonal so the matrix is positive-definite
    cov = cov + np.eye(cov.shape[0]) * eps
    return mu, cov


def fad_from_embeddings(ref_embs: np.ndarray, gen_embs: np.ndarray,
                        eps: float = 1e-6) -> Optional[float]:
    """Frechet distance between two embedding sets (n1 x d, n2 x d)."""
    if len(ref_embs) < 2 or len(gen_embs) < 2:
        return None
    mu1, cov1 = _gaussian(ref_embs, eps)
    mu2, cov2 = _gaussian(gen_embs, eps)
    return frechet_distance(mu1, cov1, mu2, cov2)


def fad_between_dirs(cfg: Config, ref_dir: Path, gen_dir: Path,
                     limit: int = 0) -> Optional[float]:
    """Embed every file in both dirs with CLAP and return the FAD."""
    from .embeddings import embed_audio

    ref_embs, gen_embs = [], []
    for label, target, out in (
        ("reference", ref_dir, ref_embs),
        ("generated", gen_dir, gen_embs),
    ):
        files = _scan(target)
        if not files:
            console.warn(f"No audio in {label} dir {target}")
            return None
        console.step(f"Embedding {label} audio ({len(files)} files)…")
        for i, p in enumerate(files[:limit] if limit else files, 1):
            try:
                out.append(embed_audio(cfg, p))
            except Exception as exc:  # noqa: BLE001
                console.warn(f"Embedding failed {p.name}: {exc}")
        console.ok(f"{label}: {len(out)}/{len(files)} embedded")

    if len(ref_embs) < 2 or len(gen_embs) < 2:
        console.warn("Need >= 2 embedded files per set for FAD.")
        return None
    value = fad_from_embeddings(
        np.stack(ref_embs), np.stack(gen_embs), eps=cfg.metrics.fad_eps
    )
    console.ok(f"FAD = {value:.4f}" if value is not None else "FAD unavailable")
    return value


# --------------------------------------------------------------------------- #
# Spectral KL divergence
# --------------------------------------------------------------------------- #

def _mel_db(path: Path, cfg: Config) -> np.ndarray:
    import librosa

    y, sr = librosa.load(str(path), sr=cfg.metrics.sr, mono=True)
    if y.size == 0:
        return np.zeros((cfg.metrics.n_mels, 1))
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=cfg.metrics.n_mels,
        hop_length=cfg.metrics.hop_length, n_fft=cfg.metrics.n_fft,
    )
    return librosa.power_to_db(S, ref=np.max, top_db=120.0)


def _dist(matrices: List[np.ndarray]) -> np.ndarray:
    """Mean per-frame mel-DB vector across all files (a spectral distribution)."""
    frames = [m.T for m in matrices if m.size]
    if not frames:
        return np.zeros(64)
    flat = np.concatenate(frames, axis=0)  # (n_frames, n_mels)
    # per-frame normalization -> each frame sums to 1
    rowsum = flat.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    flat = flat / rowsum
    return flat.mean(axis=0)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = p + 1e-12
    q = q + 1e-12
    return float(np.sum(p * np.log(p / q)))


def kl_spectral(ref_files: List[Path], gen_files: List[Path],
                cfg: Config) -> Optional[float]:
    """Symmetric mean KL between reference and generated log-mel distributions."""
    if not ref_files or not gen_files:
        return None
    ref_mats = [_mel_db(p, cfg) for p in ref_files]
    gen_mats = [_mel_db(p, cfg) for p in gen_files]
    p, q = _dist(ref_mats), _dist(gen_mats)
    # symmetric KL = KL(ref||gen) + KL(gen||ref) averaged
    return round((_kl(p, q) + _kl(q, p)) / 2.0, 6)


# --------------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------------- #

def _scan(dir_path: Path) -> List[Path]:
    found: List[Path] = []
    for pattern in AUDIO_GLOB:
        found.extend(sorted(dir_path.glob(pattern)))
    return sorted(set(found))


def compute(cfg: Config, ref_dir: Path, gen_dir: Path,
            limit: int = 0) -> dict:
    """Compute FAD + spectral KL between two audio directories."""
    ref_files = _scan(ref_dir)
    gen_files = _scan(gen_dir)
    if not ref_files or not gen_files:
        console.error("Both --ref and --gen dirs need audio files.")
        return {}

    console.step(f"KL: {len(ref_files)} ref vs {len(gen_files)} gen (spectral)")
    kl = kl_spectral(ref_files, gen_files, cfg)
    console.ok(f"Spectral KL = {kl:.4f} nats" if kl is not None else "KL unavailable")

    fad = None
    if cfg.clap.enabled:
        fad = fad_between_dirs(cfg, ref_dir, gen_dir, limit=limit)
    else:
        console.warn("CLAP disabled — skipping FAD (spectral KL only).")

    record = {
        "ref_dir": str(ref_dir),
        "gen_dir": str(gen_dir),
        "n_ref": len(ref_files),
        "n_gen": len(gen_files),
        "spectral_kl": kl,
        "fad_clap": round(fad, 4) if fad is not None else None,
        "note": "FAD computed on CLAP embeddings (VGGish-substitute)" if fad is not None
        else "FAD skipped (CLAP disabled or <2 files/set)",
    }

    out = cfg.project_root / "metadata" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    console.ok(f"Metrics -> {out.relative_to(cfg.project_root)}")
    return record
