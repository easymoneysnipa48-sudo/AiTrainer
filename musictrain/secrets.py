"""Secrets hygiene (#18).

Validate and redact the credential-shaped strings the toolkit reads from the
environment (HF tokens, webhook URLs, SMTP creds) so they are never logged
verbatim and so obviously-malformed values fail early with a clear message.
"""
from __future__ import annotations

import os
import re
from typing import Any, Mapping
from urllib.parse import urlparse

from .logging import get_logger

log = get_logger("secrets")

_SECRET_KEYS = {
    "hf_token", "hf_hub_token", "huggingface_token", "token", "password",
    "smtp_password", "slack_webhook", "discord_webhook", "telegram_token",
    "api_key", "secret", "authorization",
}

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def redact(value: str, keep: int = 4, min_len: int = 8) -> str:
    """Mask a secret, keeping a short prefix/suffix for recognizability."""
    if not isinstance(value, str):
        return str(value)
    if len(value) <= min_len:
        return "*" * len(value)
    return f"{value[:keep]}…{value[-keep:]}"


def redact_mapping(data: Mapping[str, Any]) -> dict:
    """Recursively redact any key that looks like a secret/credential."""
    out: dict = {}
    for k, v in data.items():
        lk = str(k).lower()
        if isinstance(v, Mapping):
            out[k] = redact_mapping(v)
        elif any(s in lk for s in _SECRET_KEYS) and isinstance(v, (str, bytes)):
            out[k] = redact(v.decode() if isinstance(v, bytes) else v)
        else:
            out[k] = v
    return out


def validate_hf_token(token: str) -> dict:
    """Validate an HF token: non-empty, ``hf_``-prefixed, sane length."""
    if not token:
        return {"valid": False, "reason": "empty token"}
    if not token.startswith("hf_"):
        return {"valid": False, "reason": "HF tokens start with 'hf_'"}
    if len(token) < 20:
        return {"valid": False, "reason": "token too short"}
    return {"valid": True, "masked": redact(token)}


def validate_webhook(url: str) -> dict:
    """Validate a webhook URL: http(s) with a hostname."""
    if not url:
        return {"valid": False, "reason": "empty URL"}
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.netloc:
        return {"valid": False, "reason": "must be an http(s) URL"}
    return {"valid": True, "host": p.netloc, "masked": redact(url, keep=10)}


def scan_env(include_values: bool = False) -> dict:
    """Report which credential-shaped env vars are set, redacted."""
    found = {}
    for key, value in sorted(os.environ.items()):
        lk = key.lower()
        if any(s in lk for s in _SECRET_KEYS):
            found[key] = redact(value) if value else "(empty)"
    if not found:
        return {"secrets": {}}
    log.warning("credential env vars detected: %s", ", ".join(found))
    return {"secrets": found}
