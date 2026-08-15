"""Shared logging setup for musictrain.

Wires Python's stdlib ``logging`` into the package so that progress and
errors are captured to a rotating file (``logs/musictrain.log``) as well as
the console. The ``console`` helpers remain the user-facing pretty-printers;
this module provides structured, greppable logs and backs the "advanced error
handling" layer: any uncaught exception in the CLI is logged with a full
traceback before being surfaced to the user.

Usage::

    from .logging import get_logger, setup

    setup(root=cfg.project_root, verbose=verbose, quiet=quiet)
    log = get_logger(__name__)
    log.info("processing %s", path)
    try:
        ...
    except Exception:
        log.exception("failed on %s", path)   # logs full traceback
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

LOGGER_NAME = "musictrain"

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a namespaced logger (``musictrain`` or ``musictrain.<name>``).

    Namespacing lets callers control verbosity per subsystem while sharing a
    single root handler configuration.
    """
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


def setup(root: Optional[Path] = None, verbose: bool = False,
          quiet: bool = False, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure the ``musictrain`` logger once per process (idempotent).

    Layering is deliberate: the ``console`` helpers are the user-facing output
    layer, while this logger is the structured, greppable side channel.

    * ``root``      — project root; file logs go to ``<root>/logs/musictrain.log``.
    * ``verbose``   — also mirror DEBUG+ records (incl. tracebacks) to stderr.
    * ``quiet``     — file log keeps only WARNING+ (no console output).
    * ``log_file``  — explicit override for the log file path.

    The file handler rotates at 5 MB and keeps 3 backups. Logging is always
    best-effort: an unwritable log directory never raises.
    """
    global _configured

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)   # let handlers do the filtering
    logger.propagate = False

    if _configured:
        return logger

    fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler (stderr) — a debug aid, only present in verbose mode so
    # normal/quiet runs keep the clean `console.*` output as their only stdout
    # surface.
    if verbose:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    # File handler — never let logging break the app if the dir is unwritable.
    target = log_file
    if target is None and root is not None:
        target = Path(root) / "logs" / "musictrain.log"
    if target is not None:
        try:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                str(target), maxBytes=5_000_000, backupCount=3, encoding="utf-8",
            )
            file_handler.setLevel(logging.WARNING if quiet else logging.INFO)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError:
            # Logging must be a best-effort side channel, never a crash source.
            pass

    _configured = True
    return logger


def log_exception(logger: logging.Logger, message: str, *args,
                  exc_info=True) -> None:
    """Convenience wrapper: log ``message`` with the current exception's traceback."""
    logger.error(message, *args, exc_info=exc_info)
