"""Validation + inventory: scan a data directory and record audio properties."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List

import soundfile as sf

from .. import console
from ..util import format_seconds, human_size, sha256_file

AUDIO_GLOB = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.aiff", "*.aif", "*.ogg")


def _probe(path: Path) -> dict:
    """ffprobe fallback for formats soundfile can't read (m4a/AAC, ...)."""
    import json as _json
    import subprocess

    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True)  # noqa: S603
        data = _json.loads(out.stdout)
    except Exception:  # noqa: BLE001
        return {}
    streams = data.get("streams") or []
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not audio_stream:
        return {}
    fmt = data.get("format") or {}
    return {
        "duration": round(float(fmt.get("duration") or 0.0), 3),
        "sample_rate": int(audio_stream.get("sample_rate") or 0),
        "channels": int(audio_stream.get("channels") or 0),
        "frames": int(audio_stream.get("duration_ts") or 0),
        "format": audio_stream.get("codec_name") or "?",
        "subtype": "",
        "via": "ffprobe",
    }


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
            probed = _probe(path)
            if probed:
                rec.update(probed)
                rec["valid"] = True
            else:
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
