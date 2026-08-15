"""Data-engineering helpers (advanced data batch #31-#40).

* **ASR transcription (#31)** — whisper (lazy) for vocal segments.
* **Stem separation (#32)** — already provided by ``musictrain stems`` (Demucs).
* **Corpus-wide embedding dedup (#33)** — cosine clustering over the whole
  corpus (not just within segments).
* **Sample quality fields (#34)** — SNR + clipping + loudness + DC offset
  baked into per-sample metadata.
* **Dataset snapshots (#35)** — timestamped, hash-verified manifest snapshots.
* **Synthetic prompt expansion (#36)** — extra prompts from the controlled
  vocabulary for data-augmentation of the eval set.
* **Tag co-occurrence mining (#37)** — over/under-represented
  genre x mood x instrument combos.
* **Label-balanced sampling (#38)** — draw a class-balanced subset.
* **Provenance fields (#39)** — source URL / license / origin per sample.
* **Batch pre-annotation (#40)** — BPM/key/section tagging at import time.

Everything here is numpy/librosa or pure Python — no model loads except the
optional whisper path in :func:`transcribe`.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import console
from .config import Config


# --------------------------------------------------------------------------- #
# #31 ASR transcription (whisper, lazy)
# --------------------------------------------------------------------------- #
def transcribe(path: Path, model_name: str = "base",
               device: str = "auto") -> Optional[dict]:
    """Transcribe a vocal segment with whisper. None when whisper is absent."""
    try:
        import whisper
    except ImportError:
        console.warn("whisper is not installed — `pip install openai-whisper` for ASR.")
        return None
    try:
        model = whisper.load_model(model_name, device=None if device == "auto" else device)
        result = model.transcribe(str(path))
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language"),
            "segments": [
                {"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
                 "text": s["text"].strip()}
                for s in result.get("segments", [])
            ],
        }
    except Exception as exc:  # noqa: BLE001
        console.warn(f"ASR failed for {path.name}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# #33 corpus-wide embedding dedup
# --------------------------------------------------------------------------- #
def corpus_dedup(embeddings: np.ndarray, names: Sequence[str],
                 threshold: float = 0.97) -> Dict:
    """Greedy cosine-similarity clustering over the whole embedding matrix.

    Returns duplicate clusters (each with the canonical representative and its
    near-duplicates) plus a per-name duplicate flag.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2 or len(emb) < 2:
        return {"n": int(len(emb)), "clusters": [], "n_duplicates": 0}

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms
    sim = unit @ unit.T

    n = len(emb)
    visited = [False] * n
    clusters: List[dict] = []
    for i in range(n):
        if visited[i]:
            continue
        group = [j for j in range(i + 1, n) if not visited[j] and sim[i, j] >= threshold]
        if group:
            for j in group:
                visited[j] = True
            clusters.append({
                "representative": str(names[i]),
                "duplicates": [str(names[j]) for j in group],
                "mean_similarity": round(float(np.mean([sim[i, j] for j in group])), 4),
            })
        visited[i] = True

    dup_count = sum(len(c["duplicates"]) for c in clusters)
    return {
        "n": n,
        "threshold": threshold,
        "clusters": clusters,
        "n_duplicates": dup_count,
    }


# --------------------------------------------------------------------------- #
# #34 sample quality fields (SNR + clipping + loudness + DC)
# --------------------------------------------------------------------------- #
def snr_estimate(y: np.ndarray, sr: int, hop_length: int = 512,
                 noise_quantile: float = 0.10) -> Optional[float]:
    """Signal-to-noise ratio in dB from the frame-RMS distribution.

    Noise floor = median RMS of the quietest frames; signal = the 90th
    percentile RMS. Robust to short tonal content.
    """
    import librosa

    if len(y) < hop_length:
        return None
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if rms.size < 4:
        return None
    noise = float(np.percentile(rms, noise_quantile * 100))
    signal = float(np.percentile(rms, 90))
    if noise < 1e-9:
        return None
    return round(20.0 * np.log10(signal / noise), 2)


def sample_quality(path: Path) -> dict:
    """Per-sample quality fields: SNR, clipping, DC offset, LUFS."""
    from .audio.features import estimate_lufs, load_audio

    y, sr = load_audio(path, sr=None)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return {
        "path": str(path),
        "snr_db": snr_estimate(y, sr),
        "clipping_ratio": round(float(np.mean(np.abs(y) >= 0.999)), 5),
        "dc_offset": round(float(np.abs(np.mean(y))), 5),
        "lufs": estimate_lufs(y, sr),
        "peak": round(peak, 5),
    }


# --------------------------------------------------------------------------- #
# #35 dataset snapshots
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(root: Path, cfg: Config, label: Optional[str] = None,
             which: str = "segments") -> dict:
    """Timestamped, hash-verified manifest snapshot of data/<which>."""
    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return {}

    stamp = label or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    files = sorted(target.glob("*.wav")) or sorted(target.glob("*.flac"))
    entries = []
    console.step(f"Hashing {len(files)} file(s) in data/{which}…")
    for p in files:
        try:
            entries.append({
                "path": str(p.relative_to(root)),
                "sha256": _sha256(p),
                "size_bytes": p.stat().st_size,
            })
        except OSError as exc:
            console.warn(f"Skip {p.name}: {exc}")

    manifest = {
        "snapshot": stamp,
        "which": which,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_files": len(entries),
        "files": entries,
    }
    snap_dir = root / "metadata" / "snapshots" / stamp
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # also copy the labels + config alongside for reproducibility
    for src in ("metadata/labels.csv", "config.yaml"):
        p = root / src
        if p.exists():
            import shutil

            shutil.copy(p, snap_dir / p.name)
    console.ok(f"Snapshot -> {snap_dir.relative_to(root)} ({len(entries)} files)")
    return manifest


# --------------------------------------------------------------------------- #
# #36 synthetic prompt expansion
# --------------------------------------------------------------------------- #
def expand_prompts(n: int, seed: int = 42,
                   genres: Optional[Sequence[str]] = None,
                   moods: Optional[Sequence[str]] = None,
                   instruments: Optional[Sequence[str]] = None) -> List[dict]:
    """Generate N extra prompts from the controlled vocabulary (#36)."""
    from .evalset import BPMs, KEYS, SECTIONS

    genres = list(genres or ["melodic trap", "lofi", "drill"])
    moods = list(moods or ["dark", "emotional", "tense", "reflective", "aggressive", "calm"])
    instruments = list(instruments or ["piano", "808 bass", "trap hi-hats", "strings", "pads", "guitar loop"])

    rng = np.random.default_rng(seed)
    sections = list(SECTIONS.keys())
    out: List[dict] = []
    for i in range(n):
        sec = sections[i % len(sections)]
        bpm = int(BPMs[i % len(BPMs)])
        key = KEYS[i % len(KEYS)]
        genre = genres[i % len(genres)]
        mood = moods[rng.integers(0, len(moods))]
        instr = instruments[rng.integers(0, len(instruments))]
        out.append({
            "id": f"synth_{i:03d}",
            "section": sec,
            "genre": genre,
            "bpm": bpm,
            "key": key,
            "mood": [mood],
            "instruments": [instr],
            "energy": 0.5,
            "seed": int(rng.integers(0, 1 << 31)),
            "synthetic": True,
            "description": f"{sec}, {bpm} BPM, {key}, {genre}, {mood}, {instr}",
        })
    return out


# --------------------------------------------------------------------------- #
# #37 tag co-occurrence mining
# --------------------------------------------------------------------------- #
def tag_cooccurrence(rows: Iterable[dict],
                     dims: Tuple[str, ...] = ("genre", "mood", "section"),
                     top_k: int = 12) -> dict:
    """Count genre x mood x section combos and flag under/over-represented."""
    counts: Counter = Counter()
    for r in rows:
        parts = []
        for d in dims:
            v = r.get(d)
            if isinstance(v, list):
                v = ",".join(sorted(map(str, v)))
            parts.append(str(v) if v else "?")
        counts[tuple(parts)] += 1

    total = sum(counts.values())
    entries = [
        {"combo": dict(zip(dims, k)), "count": c, "share": round(c / total, 4) if total else 0.0}
        for k, c in counts.most_common()
    ]
    # under-represented = the least common; over = the most common
    under = entries[::-1][:top_k]
    over = entries[:top_k]
    return {
        "n_combos": len(entries),
        "n_samples": total,
        "top": over,
        "underrepresented": under,
    }


# --------------------------------------------------------------------------- #
# #38 label-balanced sampling
# --------------------------------------------------------------------------- #
def balanced_sample(items: Sequence, labels: Sequence[str], n: int,
                    seed: int = 0) -> List[int]:
    """Return indices of a class-balanced sample of size ~n.

    Distributes slots as evenly as possible across label classes, sampling
    without replacement within each class.
    """
    rng = np.random.default_rng(seed)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[str(lab)].append(i)

    classes = list(by_label.keys())
    if not classes:
        return []
    per_class = max(1, n // len(classes))
    chosen: List[int] = []
    for lab in classes:
        idx = by_label[lab]
        take = min(len(idx), per_class)
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    # top up with remaining slots from the largest leftover classes
    while len(chosen) < n:
        added = False
        for lab in sorted(classes, key=lambda c: -len(by_label[c])):
            idx = [i for i in by_label[lab] if i not in set(chosen)]
            if idx:
                chosen.append(int(rng.choice(idx)))
                added = True
                if len(chosen) >= n:
                    break
        if not added:
            break
    return chosen[:n]


# --------------------------------------------------------------------------- #
# #39 provenance fields
# --------------------------------------------------------------------------- #
def annotate_provenance(rows: List[dict], source_url: str = "",
                        license_name: str = "", origin: str = "") -> List[dict]:
    """Ensure every row carries source_url / license / origin fields."""
    out = []
    for r in rows:
        r = dict(r)
        r.setdefault("source_url", source_url)
        r.setdefault("license", license_name)
        r.setdefault("origin", origin)
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# #40 batch pre-annotation at import
# --------------------------------------------------------------------------- #
def pre_annotate(root: Path, cfg: Config, which: str = "clean",
                 limit: int = 0, out_csv: Optional[Path] = None) -> List[dict]:
    """Extract BPM/key/loudness (+ section guess) and write labels.csv rows."""
    from .audio.analysis import analyze_file
    from .audio.inventory import AUDIO_GLOB

    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return []

    files: List[Path] = []
    for pattern in AUDIO_GLOB:
        files.extend(sorted(target.glob(pattern)))
    files = sorted(set(files))
    if not files:
        console.warn(f"No audio under {target}")
        return []

    rows: List[dict] = []
    console.step(f"Pre-annotating {len(files)} file(s) in data/{which}…")
    for i, p in enumerate(files, 1):
        if limit and i > limit:
            break
        try:
            an = analyze_file(cfg, p, root)
        except Exception as exc:  # noqa: BLE001
            console.warn(f"Analysis failed {p.name}: {exc}")
            continue
        segments = an.get("structure", {}).get("segments", [])
        section = segments[0].get("role") if segments else ""
        rows.append({
            "source_id": p.stem,
            "path": str(p.relative_to(root)),
            "bpm": an.get("beat_grid", {}).get("tempo"),
            "key": an.get("key", {}).get("key"),
            "section": section,
            "duration": an.get("duration"),
            "auto": True,
        })
        console.info(f"[{i}/{len(files)}] {p.name}: key={rows[-1]['key']} bpm={rows[-1]['bpm']}")

    out_csv = out_csv or (root / "metadata" / "preannotated_labels.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_id", "path", "bpm", "key", "section", "duration", "auto"]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    console.ok(f"Pre-annotation -> {out_csv.relative_to(root)} ({len(rows)} rows)")
    return rows
