"""Extended audio/data tools (gap #20-#24).

* ``diarize``          — heuristic speaker segmentation (energy + timbre), with a
  thin pyannote hook for real diarization when installed.
* ``transcribe_midi``  — autocorrelation pitch tracker → MIDI note list.
* ``tempo_key_augment``— key/tempo-consistent augmentation specs (respects the
  beat grid by folding tempo ratios instead of naive stretches).
* ``bundle``           — one-click dataset export (zip + SHA-256 manifest).
* ``fad_stats_cache``  — precompute reference CLAP mean/cov once for CI reuse.

All signal math is pure NumPy so it runs (and is testable) without optional
model dependencies; the heavy backends (pyannote/librosa) degrade gracefully.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import console
from .config import Config
from .logging import get_logger

log = get_logger("audioext")


# --------------------------------------------------------------------------- #
# #20 speaker diarization
# --------------------------------------------------------------------------- #
def diarize(y: np.ndarray, sr: int, hop: int = 512, frame: int = 2048,
            min_segment_s: float = 0.4, energy_threshold: float = 0.02) -> List[dict]:
    """Heuristic speaker segmentation: split on silence, then cluster segments.

    Returns ``[{speaker, start, end}]``. A real diarizer (pyannote) is preferred
    when available — this is a dependency-free fallback for quick checks.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) == 0:
        return []

    frame_len = int(sr * 0.025)
    hop_f = max(1, int(sr * 0.010))
    n_frames = 1 + max(0, (len(y) - frame_len) // hop_f)

    energy = np.zeros(n_frames, dtype=np.float32)
    centroid = np.zeros(n_frames, dtype=np.float32)
    window = np.hanning(frame_len)
    for i in range(n_frames):
        seg = y[i * hop_f:i * hop_f + frame_len]
        if len(seg) < frame_len:
            seg = np.pad(seg, (0, frame_len - len(seg)))
        x = seg * window
        energy[i] = float(np.sqrt(np.mean(x * x)))
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(frame_len, 1 / sr)
        denom = float(spec.sum()) or 1.0
        centroid[i] = float((freqs * spec).sum() / denom)

    # voice frames = non-silent
    active = energy > energy_threshold
    if not active.any():
        return []

    # segment boundaries where active transitions (silence gaps)
    bounds: List[int] = []
    prev = active[0]
    for i in range(1, n_frames):
        if active[i] != prev:
            bounds.append(i)
            prev = active[i]

    # split into voiced regions
    regions: List[Tuple[int, int]] = []
    start = 0
    for b in bounds:
        if active[b] and (active[b - 1] if b > 0 else False) is False:
            start = b
        elif not active[b]:
            if b - start >= int(min_segment_s * sr / hop_f):
                regions.append((start, b))
            start = b
    if active[-1] and n_frames - 1 - start >= int(min_segment_s * sr / hop_f):
        regions.append((start, n_frames - 1))
    if not regions and active.any():
        regions = [(0, n_frames - 1)]

    # cluster regions by mean spectral centroid (simple 2-way split by median)
    segs = []
    for i, (a, b) in enumerate(regions):
        segs.append({
            "speaker": i, "start": round(a * hop_f / sr, 3),
            "end": round(min(len(y), b * hop_f) / sr, 3),
            "mean_centroid_hz": round(float(centroid[a:b + 1].mean()), 1),
        })
    if len(segs) > 1:
        cents = [s["mean_centroid_hz"] for s in segs]
        med = float(np.median(cents))
        for s in segs:
            s["speaker"] = 0 if s["mean_centroid_hz"] <= med else 1
    return segs


def diarize_pyannote(path: Path, hf_token: str = "") -> List[dict]:
    """Real speaker diarization via pyannote.audio (optional dependency)."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        console.warn("pyannote.audio not installed — using heuristic diarization.")
        import soundfile as sf

        y, sr = sf.read(str(path))
        return diarize(y, sr)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token or None
    )
    out: List[dict] = []
    for turn, _, speaker in pipeline(str(path)).itertracks(yield_label=True):
        out.append({"speaker": speaker, "start": round(turn.start, 3),
                    "end": round(turn.end, 3)})
    return out


# --------------------------------------------------------------------------- #
# #21 MIDI transcription
# --------------------------------------------------------------------------- #
def _f0_autocorr(y: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 1200.0) -> Optional[float]:
    """Estimate fundamental frequency of a frame via autocorrelation."""
    if len(y) < 64 or float(np.sqrt(np.mean(y * y))) < 1e-3:
        return None
    y = y - y.mean()
    n = len(y)
    ac = np.correlate(y, y, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo = int(sr / fmax)
    hi = int(sr / fmin)
    if hi >= len(ac):
        hi = len(ac) - 1
    if lo >= hi:
        return None
    # find first strong peak in plausible lag range
    peak = lo + int(np.argmax(ac[lo:hi]))
    if ac[peak] < 0.3:
        return None
    return float(sr / peak)


def _hz_to_midi(hz: float) -> int:
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def transcribe_midi(y: np.ndarray, sr: int, hop: float = 0.05) -> List[dict]:
    """Frame-wise pitch tracking → monophonic MIDI note list (onset/note/dur)."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    frame_len = int(sr * 0.05)
    hop_n = max(1, int(sr * hop))
    notes: List[dict] = []
    current: Optional[dict] = None
    t = 0.0
    for start in range(0, max(1, len(y) - frame_len + 1), hop_n):
        seg = y[start:start + frame_len]
        f0 = _f0_autocorr(seg, sr)
        midi = _hz_to_midi(f0) if f0 is not None else None
        if midi is not None:
            if current is None:
                current = {"note": midi, "onset": round(t, 3), "duration": 0.0}
            elif current["note"] == midi:
                current["duration"] = round(t - current["onset"] + hop, 3)
            else:
                notes.append(current)
                current = {"note": midi, "onset": round(t, 3), "duration": hop}
        else:
            if current is not None:
                notes.append(current)
                current = None
        t += hop
    if current is not None:
        notes.append(current)
    return notes


# --------------------------------------------------------------------------- #
# #22 tempo/key-aware augmentation
# --------------------------------------------------------------------------- #
_NOTE_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _parse_key(key: str) -> Tuple[str, int]:
    """Parse 'A minor' / 'C# major' → (pitch class, mode 0=major 1=minor)."""
    parts = (key or "").strip().split()
    if not parts:
        return "C", 0
    pc = parts[0].upper()
    mode = 0
    if len(parts) > 1:
        mode = 1 if parts[1].lower().startswith("min") else 0
    elif "minor" in key.lower():
        mode = 1
    return pc, mode


def transposed_key(key: str, semitones: int) -> str:
    pc, mode = _parse_key(key)
    idx = _NOTE_ORDER.index(pc) if pc in _NOTE_ORDER else 0
    new_pc = _NOTE_ORDER[(idx + semitones) % 12]
    return f"{new_pc} {'minor' if mode else 'major'}"


def tempo_key_augment(bpm: float, key: str, semitones: int = 0,
                      tempo_ratio: Optional[float] = None) -> dict:
    """Build a tempo/key-consistent augmentation spec.

    ``semitones`` transposes the key; ``tempo_ratio`` is constrained to the
    musical fold set (1/4, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 4) so the beat grid
    stays aligned instead of a naive non-integer stretch.
    """
    folds = (1 / 4, 1 / 3, 1 / 2, 2 / 3, 1.0, 3 / 2, 2.0, 3.0, 4.0)
    if tempo_ratio is None:
        tempo_ratio = 1.0
    # snap to nearest allowed fold
    ratio = min(folds, key=lambda f: abs(f - tempo_ratio))
    new_key = transposed_key(key, semitones)
    new_bpm = round(bpm * ratio, 3)
    return {
        "key": key, "new_key": new_key, "semitones": semitones,
        "bpm": bpm, "tempo_ratio": ratio, "new_bpm": new_bpm,
        "in_fold_set": ratio in folds,
        "op": f"transpose_{semitones:+d}_ratio_{ratio:g}".replace(" ", ""),
    }


# --------------------------------------------------------------------------- #
# #23 dataset bundle export/import
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle(root: Path, cfg: Config, which: str = "clean",
           dest: Optional[str] = None, include_metadata: bool = True) -> dict:
    """Zip a data dir (+ optional metadata) into a portable, hash-verified bundle."""
    src = root / "data" / which
    if not src.is_dir():
        console.error(f"data/{which} not found.")
        return {"error": "no_dir", "n_files": 0}
    dest_path = Path(dest) if dest else root / f"dataset_{which}.zip"
    files = sorted(p for p in src.rglob("*") if p.is_file())
    manifest: Dict[str, str] = {}

    with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            rel = str(f.relative_to(src))
            zf.write(f, arcname=f"{which}/{rel}")
            manifest[f"{which}/{rel}"] = _sha256_file(f)
        if include_metadata:
            manifest["__manifest__.json"] = ""  # placeholder replaced below

    # write manifest inside the archive for integrity checks
    manifest_bytes = json.dumps({"files": manifest}, indent=2).encode()
    with zipfile.ZipFile(dest_path, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("__manifest__.json", manifest_bytes)

    console.ok(f"Bundled {len(files)} file(s) -> {dest_path.name}")
    log.info("dataset bundle: %s (%d files)", dest_path, len(files))
    return {"archive": str(dest_path), "n_files": len(files),
            "manifest": manifest, "sha256": _sha256_file(dest_path)}


def verify_bundle(archive: Path) -> dict:
    """Check every entry in a bundle against its recorded SHA-256."""
    if not archive.is_file():
        return {"error": "not_found"}
    ok = 0
    bad: List[str] = []
    with zipfile.ZipFile(archive) as zf:
        manifest = json.loads(zf.read("__manifest__.json"))["files"]
        for name, want in manifest.items():
            if not want or name == "__manifest__.json":
                continue
            try:
                data = zf.read(name)
            except KeyError:
                bad.append(name)
                continue
            got = hashlib.sha256(data).hexdigest()
            if got != want:
                bad.append(name)
            else:
                ok += 1
    return {"verified": ok, "corrupt": bad, "ok": not bad}


# --------------------------------------------------------------------------- #
# #24 reference FAD-set stats cache
# --------------------------------------------------------------------------- #
def fad_stats_cache(cfg: Config, ref_dir: str = "clean", limit: int = 0) -> dict:
    """Precompute reference CLAP mean/cov and cache it for CI reuse."""
    from .embeddings import embed_dir

    embs = embed_dir(cfg.project_root, cfg, which=ref_dir, limit=limit)
    if not embs:
        console.warn("No embeddings for the reference set — cache skipped.")
        return {"error": "no_embeddings"}
    mat = np.stack(list(embs.values()))
    mean = mat.mean(axis=0)
    cov = np.cov(mat, rowvar=False)
    out = {
        "ref_dir": ref_dir,
        "n": int(mat.shape[0]),
        "dim": int(mat.shape[1]),
        "mean": mean.tolist(),
        "cov": cov.tolist(),
    }
    path = cfg.project_root / "metadata" / "fad_reference_stats.json"
    path.write_text(json.dumps(out))
    console.ok(f"FAD reference stats cached -> {path.name} ({out['n']} clips)")
    return {"path": str(path), "n": out["n"], "dim": out["dim"]}


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #
def run(cfg: Config, task: str, path: str = "", bpm: float = 120.0,
        key: str = "C major", semitones: int = 0, tempo_ratio: Optional[float] = None,
        which: str = "clean", dest: str = "", limit: int = 0) -> dict:
    if task == "diarize":
        import soundfile as sf

        y, sr = sf.read(path)
        segs = diarize(y, sr)
        console.ok(f"Diarization: {len(segs)} segment(s)")
        for s in segs:
            console.info(f"  speaker {s['speaker']}: {s['start']}s-{s['end']}s")
        return {"task": task, "segments": segs}

    if task == "midi":
        import soundfile as sf

        y, sr = sf.read(path)
        notes = transcribe_midi(y, sr)
        console.ok(f"MIDI: {len(notes)} note(s) transcribed")
        return {"task": task, "n_notes": len(notes), "notes": notes[:200]}

    if task == "augment":
        out = tempo_key_augment(bpm, key, semitones, tempo_ratio)
        console.ok(f"Augment: {key} {bpm} BPM -> {out['new_key']} {out['new_bpm']} BPM")
        return {"task": task, **out}

    if task == "bundle":
        return bundle(cfg.project_root, cfg, which=which, dest=dest or None)

    if task == "verify-bundle":
        from pathlib import Path as P

        return verify_bundle(P(dest) if dest else cfg.project_root / f"dataset_{which}.zip")

    if task == "fad-cache":
        return fad_stats_cache(cfg, ref_dir=which, limit=limit)

    console.error(f"Unknown audioext task {task!r}")
    return {"error": f"unknown task {task}"}
