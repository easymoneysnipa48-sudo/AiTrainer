"""Dashboard auth (gap #17).

Lightweight, dependency-free authentication for the Streamlit dashboard:

* ``MUSICTRAIN_PASSWORD`` — single shared password (hashed at rest).
* ``MUSICTRAIN_USERS`` — JSON map of ``{"user": "<salted hash>"}``.
* ``MUSICTRAIN_OAUTH_TOKENS`` — JSON list of bearer tokens accepted as-is.

When no credentials are configured the gate is a no-op (open access), matching
the existing single-user local tool. Passwords are hashed with salted SHA-256
and compared in constant time — good enough for a local dashboard; swap in
Passlib/Argon2 if you ever expose it publicly.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from typing import Dict, Optional

from .logging import get_logger

log = get_logger("auth")

_ENV_PASSWORD = "MUSICTRAIN_PASSWORD"
_ENV_USERS = "MUSICTRAIN_USERS"
_ENV_OAUTH = "MUSICTRAIN_OAUTH_TOKENS"


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Salted SHA-256, ``salt$hexdigest``. For local use, not a KDF substitute."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    if "$" not in stored:
        return False
    salt, digest = stored.split("$", 1)
    candidate = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return hmac.compare_digest(candidate, digest)


def load_credentials() -> Dict[str, str]:
    """Return ``{username: hashed_password}`` from env, if configured."""
    users = os.environ.get(_ENV_USERS, "")
    if users:
        try:
            return json.loads(users)
        except json.JSONDecodeError:
            log.warning("MUSICTRAIN_USERS is not valid JSON — ignoring")
    pw = os.environ.get(_ENV_PASSWORD, "")
    if pw:
        return {"default": hash_password(pw)}
    return {}


def load_oauth_tokens() -> list:
    raw = os.environ.get(_ENV_OAUTH, "")
    if not raw:
        return []
    try:
        tokens = json.loads(raw)
        return tokens if isinstance(tokens, list) else []
    except json.JSONDecodeError:
        log.warning("MUSICTRAIN_OAUTH_TOKENS is not valid JSON — ignoring")
        return []


def is_configured() -> bool:
    return bool(load_credentials() or load_oauth_tokens())


def authenticate(username: str, password: str, token: str = "") -> bool:
    creds = load_credentials()
    tokens = load_oauth_tokens()
    if token and token in tokens:
        return True
    if username in creds and verify_password(password, creds[username]):
        return True
    return False


def streamlit_gate() -> bool:
    """Render a login gate; returns True once authorized. No-op if unconfigured."""
    if not is_configured():
        return True
    import streamlit as st

    if st.session_state.get("_auth_ok"):
        return True

    st.title("🔐 Sign in")
    username = st.text_input("Username", key="auth_user")
    password = st.text_input("Password", type="password", key="auth_pass")
    token = st.text_input("Token (optional)", key="auth_token")
    if st.button("Sign in"):
        if authenticate(username, password, token=token):
            st.session_state["_auth_ok"] = True
            st.rerun()
        else:
            st.error("Invalid credentials.")
            return False
    return False
