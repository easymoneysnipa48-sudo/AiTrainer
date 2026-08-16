"""Retry/backoff with jitter (#16).

A dependency-free helper used to wrap flaky calls (MLflow logging, webhook
delivery, HF Hub downloads). Exponential backoff with full-jitter, honoring a
``retryable`` predicate so only transient errors are retried.
"""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from .logging import get_logger

log = get_logger("retry")

T = TypeVar("T")


def retry(
    fn: Callable[..., T],
    *args,
    retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retryable=None,
    on_error: Callable[[Exception, int], None] = None,
    **kwargs,
) -> T:
    """Call ``fn(*args, **kwargs)`` with exponential backoff + jitter.

    ``retryable(exc)`` decides whether an exception is transient (default: all
    exceptions are retried). Raises the last exception after ``retries``
    attempts. ``on_error(exc, attempt)`` lets the caller log each failure.
    """
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - broad by design
            attempt += 1
            if retryable is not None and not retryable(exc):
                raise
            if on_error is not None:
                try:
                    on_error(exc, attempt)
                except Exception:  # noqa: BLE001
                    pass
            if attempt > retries:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay = random.uniform(0, delay)  # full jitter
            log.warning("retry %d/%d after %.2fs: %s", attempt, retries, delay, exc)
            time.sleep(delay)


def is_transient(exc: Exception) -> bool:
    """Heuristic: network/timeout/rate-limit errors are transient."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    for hint in ("timeout", "connection", "rate limit", "429", "503", "temporar",
                 "reset", "broken pipe", "refused", "unreachable"):
        if hint in name or hint in msg:
            return True
    return False
