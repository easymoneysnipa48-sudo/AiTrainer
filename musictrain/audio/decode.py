"""Universal audio decoding: read formats libsndfile can't (m4a/AAC, ...).

librosa/soundfile reject AAC containers with ``Error opening ... Format not
recognised``. This module transparently decodes them with ffmpeg (already a
core dependency of the normalize/segment pipelines) into a temp WAV, so every
analysis path can read any format in ``AUDIO_GLOB`` — wav, flac, mp3, m4a,
aiff, aif, ogg.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def decode_to_wav(path: Path, sr: Optional[int] = None, mono: bool = True) -> Path:
    """Convert ``path`` to a temp WAV via ffmpeg (resampled to ``sr`` if given).

    The caller owns the returned temp file — delete it after loading.
    """
    fd, tmp = tempfile.mkstemp(prefix="mt_decode_", suffix=".wav")
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path)]
    if sr:
        cmd += ["-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-f", "wav", tmp]
    subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603
    return Path(tmp)


def load_any(path: Path, sr: Optional[int] = None, mono: bool = True):
    """Load audio in any supported format -> (y, sr), like ``librosa.load``.

    Tries librosa first (fast path for wav/flac/mp3/ogg); on a decode failure
    (AAC-family containers) transparently converts with ffmpeg and loads the
    temp WAV. Raises the original error if ffmpeg is unavailable or the file
    is genuinely corrupt.
    """
    import librosa

    try:
        return librosa.load(path, sr=sr, mono=mono)
    except Exception:
        if not ffmpeg_available():
            raise
        wav = decode_to_wav(path, sr=sr, mono=mono)
        try:
            return librosa.load(wav, sr=sr, mono=mono)
        finally:
            wav.unlink(missing_ok=True)
