"""Distribution metrics between reference and generated audio (#41).

Scores:

* **FAD** — Frechet Audio Distance on CLAP audio embeddings (default,
  reuses the existing CLAP pipeline). A VGGish variant is available via the
  optional ``fadtk`` package (``--fad vggish``). Lower = closer to real.

* **Spectral KL** — mean KL divergence between the normalized log-mel
  spectrogram distributions of the two sets. Pure librosa/numpy.

* **Embedding two-sample tests** — closed-form Gaussian KLD, unbiased
  RBF-MMD, and a 1-NN classifier accuracy test on the same embeddings
  (advanced feature #2). All numpy, no extra models.
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
# VGGish FAD (advanced #1) — optional, via the `fadtk` package
# --------------------------------------------------------------------------- #

def fad_vggish(cfg: Config, ref_dir: Path, gen_dir: Path,
               limit: int = 0) -> Optional[float]:
    """FAD on the classic VGGish embedding via ``fadtk``.

    Requires `pip install fadtk`; the first run downloads the VGGish weights
    (~370 MB). Returns None (with a console note) when the package or enough
    files are missing, so the CLI degrades gracefully.
    """
    try:
        from fadtk.model import get_model
        from fadtk.fad import FrechetAudioDistance
    except Exception as exc:  # noqa: BLE001
        console.warn(f"VGGish FAD unavailable (pip install fadtk): {exc}")
        return None

    ref_files = _scan(ref_dir)[:limit] if limit else _scan(ref_dir)
    gen_files = _scan(gen_dir)[:limit] if limit else _scan(gen_dir)
    if len(ref_files) < 2 or len(gen_files) < 2:
        console.warn("VGGish FAD needs >= 2 files per set.")
        return None

    try:
        model = get_model("vggish")
        fad = FrechetAudioDistance(model, model.batch_size, model.audio_sample_rate)
        value = float(fad.score(str(ref_dir), str(gen_dir)))
        console.ok(f"FAD (VGGish) = {value:.4f}")
        return value
    except Exception as exc:  # noqa: BLE001
        console.warn(f"VGGish FAD failed: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Embedding two-sample tests (advanced #2) — pure numpy
# --------------------------------------------------------------------------- #

def kld_gaussian(ref_embs: np.ndarray, gen_embs: np.ndarray) -> float:
    """Closed-form symmetric KL between two fitted Gaussians."""
    mu1, cov1 = _gaussian(np.asarray(ref_embs, dtype=np.float64), 1e-6)
    mu2, cov2 = _gaussian(np.asarray(gen_embs, dtype=np.float64), 1e-6)
    d = mu1.shape[0]
    cov2_inv = np.linalg.pinv(cov2)
    cov1_inv = np.linalg.pinv(cov1)
    t1 = np.trace(cov2_inv @ cov1) + (mu2 - mu1) @ cov2_inv @ (mu2 - mu1) - d
    t2 = np.trace(cov1_inv @ cov2) + (mu1 - mu2) @ cov1_inv @ (mu1 - mu2) - d
    return round(float(t1 + t2) / 2.0, 6)


def mmd_rbf(ref_embs: np.ndarray, gen_embs: np.ndarray,
            sigma: Optional[float] = None) -> float:
    """Unbiased maximum-mean-discrepancy with an RBF kernel (median heuristic)."""
    x = np.asarray(ref_embs, dtype=np.float64)
    y = np.asarray(gen_embs, dtype=np.float64)
    if sigma is None:
        diff = x[np.newaxis, :, :] - y[:, np.newaxis, :]
        med = np.median(np.sqrt((diff ** 2).sum(-1)))
        sigma = float(med) if med > 0 else 1.0

    def _k(a: np.ndarray, b: np.ndarray) -> float:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return float(np.exp(-d2 / (2 * sigma ** 2)).mean())

    return round(_k(x, x) + _k(y, y) - 2 * _k(x, y), 6)


def one_nn_acc(ref_embs: np.ndarray, gen_embs: np.ndarray) -> float:
    """1-NN two-sample test: leave-one-out accuracy of a 1-NN classifier.

    Accuracy near 0.5 means the sets are indistinguishable; near 1.0 means
    they are easily separable (i.e., very different distributions).
    """
    x = np.asarray(ref_embs, dtype=np.float64)
    y = np.asarray(gen_embs, dtype=np.float64)
    X = np.vstack([x, y])
    labels = np.array([0] * len(x) + [1] * len(y))
    if len(X) < 2:
        return 0.5
    # exact duplicate sets are indistinguishable by definition (every point's
    # nearest neighbor is its twin in the other set) -> report chance level
    cross = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    if float(cross.min()) < 1e-12:
        return 0.5
    correct = 0
    for i in range(len(X)):
        d = ((X - X[i]) ** 2).sum(-1)
        d[i] = np.inf
        j = int(np.argmin(d))
        correct += int(labels[j] == labels[i])
    return round(correct / len(X), 6)


def two_sample_metrics(ref_embs: np.ndarray, gen_embs: np.ndarray) -> dict:
    """KLD / MMD / 1-NN in one call (advanced #2)."""
    return {
        "kld_gaussian": kld_gaussian(ref_embs, gen_embs),
        "mmd_rbf": mmd_rbf(ref_embs, gen_embs),
        "one_nn_acc": one_nn_acc(ref_embs, gen_embs),
    }


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


def _embed_dir(cfg: Config, dir_path: Path, limit: int = 0) -> List[np.ndarray]:
    """Embed every file in a dir with CLAP, returning the list of vectors."""
    from .embeddings import embed_audio

    files = _scan(dir_path)
    out: List[np.ndarray] = []
    console.step(f"Embedding {dir_path.name} audio ({len(files)} files)…")
    for i, p in enumerate(files[:limit] if limit else files, 1):
        try:
            out.append(embed_audio(cfg, p))
        except Exception as exc:  # noqa: BLE001
            console.warn(f"Embedding failed {p.name}: {exc}")
    console.ok(f"{dir_path.name}: {len(out)}/{len(files)} embedded")
    return out


def compute(cfg: Config, ref_dir: Path, gen_dir: Path,
            limit: int = 0, fad_kind: str = "clap") -> dict:
    """Compute FAD + spectral KL + two-sample tests between two dirs."""
    ref_files = _scan(ref_dir)
    gen_files = _scan(gen_dir)
    if not ref_files or not gen_files:
        console.error("Both --ref and --gen dirs need audio files.")
        return {}

    console.step(f"KL: {len(ref_files)} ref vs {len(gen_files)} gen (spectral)")
    kl = kl_spectral(ref_files, gen_files, cfg)
    console.ok(f"Spectral KL = {kl:.4f} nats" if kl is not None else "KL unavailable")

    record: dict = {
        "ref_dir": str(ref_dir),
        "gen_dir": str(gen_dir),
        "n_ref": len(ref_files),
        "n_gen": len(gen_files),
        "spectral_kl": kl,
        "fad_clap": None,
        "fad_vggish": None,
        "two_sample": None,
        "note": "",
    }

    ref_embs = gen_embs = None
    if cfg.clap.enabled:
        ref_embs = _embed_dir(cfg, ref_dir, limit=limit)
        gen_embs = _embed_dir(cfg, gen_dir, limit=limit)
        if len(ref_embs) >= 2 and len(gen_embs) >= 2:
            fad = fad_from_embeddings(
                np.stack(ref_embs), np.stack(gen_embs), eps=cfg.metrics.fad_eps
            )
            record["fad_clap"] = round(fad, 4) if fad is not None else None
            record["two_sample"] = two_sample_metrics(
                np.stack(ref_embs), np.stack(gen_embs)
            )
            console.ok(f"FAD (CLAP) = {fad:.4f}" if fad is not None else "FAD unavailable")
        else:
            record["note"] = "FAD/two-sample skipped: need >= 2 embedded files per set."
        if fad_kind == "vggish":
            record["fad_vggish"] = fad_vggish(cfg, ref_dir, gen_dir, limit=limit)
    else:
        record["note"] = "CLAP disabled — spectral KL only; FAD/two-sample skipped."

    out = cfg.project_root / "metadata" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    console.ok(f"Metrics -> {out.relative_to(cfg.project_root)}")
    return record


def fad_from_cached_stats(cfg: Config, gen_dir: Path, limit: int = 0) -> Optional[float]:
    """Compute FAD against a cached reference distribution (gap #9).

    Reuses ``metadata/fad_reference_stats.json`` (written by
    ``musictrain audioext --task fad-cache``) so CI can score generated audio
    without re-embedding the reference set on every run. Returns None when the
    cache is missing or the generated dir can't be embedded.
    """
    cache_path = cfg.project_root / "metadata" / "fad_reference_stats.json"
    if not cache_path.exists():
        console.warn("No cached reference stats (run `musictrain audioext --task fad-cache`).")
        return None
    try:
        cached = json.loads(cache_path.read_text())
        mu_ref = np.asarray(cached["mean"], dtype=np.float64)
        cov_ref = np.asarray(cached["cov"], dtype=np.float64) + np.eye(mu_ref.shape[0]) * cfg.metrics.fad_eps
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        console.warn(f"Cached reference stats unreadable ({exc}).")
        return None

    if not cfg.clap.enabled:
        console.warn("CLAP disabled — FAD from cached stats needs CLAP.")
        return None
    gen_embs = _embed_dir(cfg, Path(gen_dir), limit=limit)
    if len(gen_embs) < 2:
        console.warn("FAD from cache needs >= 2 generated clips.")
        return None
    gen = np.stack(gen_embs)
    mu_gen, cov_gen = _gaussian(gen, cfg.metrics.fad_eps)
    fad = frechet_distance(mu_ref, cov_ref, mu_gen, cov_gen)
    return round(fad, 4) if fad is not None else None
