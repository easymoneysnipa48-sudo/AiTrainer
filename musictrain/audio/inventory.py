"""Validation + inventory: scan a data directory and record audio properties."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

import soundfile as sf

from .. import console
from ..util import format_seconds, human_size, sha256_file

AUDIO_GLOB = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.aiff", "*.aif", "*.ogg")


def _scan(dir_path: Path) -> List[Path]:
    found: List[Path] = []
    for pattern in AUDIO_GLOB:
        found.extend(sorted(dir_path.glob(pattern)))
    return sorted(set(found))


def inventory(root: Path, which: str = "clean", sha256: bool = False) -> List[dict]:
    target = root / "data" / which
    if not target.exists():
        console.error(f"Directory not found: {target}")
        return []

    results: List[dict] = []
    files = _scan(target)
    console.step(f"Scanning {len(files)} files in data/{which}")

    for path in files:
        rec: dict = {"path": str(path.relative_to(root))}
        try:
            info = sf.info(path)
            rec.update(
                duration=round(float(info.duration), 3),
                sample_rate=int(info.samplerate),
                channels=int(info.channels),
                frames=int(info.frames),
                format=info.format,
                subtype=info.subtype,
                size_bytes=path.stat().st_size,
                valid=True,
            )
            if sha256:
                rec["sha256"] = sha256_file(path)
        except Exception as exc:  # noqa: BLE001 - report any failure
            rec["valid"] = False
            rec["error"] = str(exc)
        results.append(rec)

    out = root / "metadata" / "audio_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    _summarize(results)
    return results


def _summarize(results: List[dict]) -> None:
    valid = [r for r in results if r.get("valid")]
    invalid = [r for r in results if not r.get("valid")]

    console.ok(f"Valid: {len(valid)}   Invalid: {len(invalid)}")
    if not valid:
        return

    sr_dist = Counter(r["sample_rate"] for r in valid)
    ch_dist = Counter(r["channels"] for r in valid)
    durations = [r["duration"] for r in valid]
    total_size = sum(r.get("size_bytes", 0) for r in valid)

    console.info("Sample rates: " + ", ".join(f"{sr}Hz x{n}" for sr, n in sorted(sr_dist.items())))
    console.info("Channels: " + ", ".join(f"{ch}ch x{n}" for ch, n in sorted(ch_dist.items())))
    console.info(
        f"Total duration: {format_seconds(sum(durations))} across {len(valid)} files "
        f"({human_size(total_size)})"
    )
    console.info(
        f"Duration range: {format_seconds(min(durations))} .. {format_seconds(max(durations))}"
    )
    if invalid:
        for r in invalid:
            console.warn(f"INVALID {r['path']}: {r.get('error', 'unknown')}")
