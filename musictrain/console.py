"""Small ANSI-aware console helpers (no external dependency)."""
from __future__ import annotations

import sys


class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _paint(code: str, text: object) -> str:
    if not sys.stdout.isatty():
        return str(text)
    return f"{code}{text}{_C.RESET}"


def info(msg: object) -> None:
    print(_paint(_C.BLUE, f"· {msg}"))


def ok(msg: object) -> None:
    print(_paint(_C.GREEN, f"✓ {msg}"))


def warn(msg: object) -> None:
    print(_paint(_C.YELLOW, f"! {msg}"), file=sys.stderr)


def error(msg: object) -> None:
    print(_paint(_C.RED, f"✗ {msg}"), file=sys.stderr)


def step(msg: object) -> None:
    print(_paint(_C.BOLD + _C.CYAN, f"\n▶ {msg}"))


def title(msg: object) -> None:
    print(_paint(_C.BOLD + _C.MAGENTA, str(msg)))
