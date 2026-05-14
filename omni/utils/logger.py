"""Loguru-based logger. Import `log` and use directly.

Configuration via env:
    LOG_LEVEL       INFO/DEBUG/WARNING (default INFO)
    LOG_DIR         override log directory (default <project>/logs)
    LOG_FILE_NAME   override log filename stem (default `omni`)
    LOG_NO_FILE     set to "1" to disable file logging (console only)

Behavior:
    * Pretty colorized console sink (stderr).
    * Rotating file sink in `logs/<stem>-<YYYY-MM-DD>.log` with daily rotation
      and 14-day retention. Uses enqueue=True so logging is process-safe.
    * Standard-library `logging` is intercepted into loguru, so modules using
      `import logging; logging.getLogger(...)` (FastAPI/uvicorn/sklearn/etc.)
      flow through the same pipeline.
    * Uncaught exceptions are logged before the process exits.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from loguru import logger as log

from omni.utils.paths import project_root

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_DIR = Path(os.getenv("LOG_DIR", str(project_root() / "logs")))
_LOG_STEM = os.getenv("LOG_FILE_NAME", "omni")
_NO_FILE = os.getenv("LOG_NO_FILE", "0") == "1"

# ---- Console sink ---------------------------------------------------------
log.remove()
log.add(
    sys.stderr,
    level=_LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
    backtrace=False,
    diagnose=False,
)

# ---- File sink ------------------------------------------------------------
if not _NO_FILE:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log.add(
            _LOG_DIR / f"{_LOG_STEM}-{{time:YYYY-MM-DD}}.log",
            level=_LOG_LEVEL,
            rotation="00:00",        # roll at midnight
            retention="14 days",
            compression=None,
            encoding="utf-8",
            enqueue=True,            # process-safe
            backtrace=True,
            diagnose=False,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{process}:{thread} | {name}:{function}:{line} - {message}"
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive
        # Never crash app startup because of logging; warn on stderr.
        sys.stderr.write(f"[logger] file sink disabled: {exc}\n")


# ---- Stdlib logging -> loguru bridge --------------------------------------
class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            level = log.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Find caller from outside the logging module
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        log.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _install_stdlib_bridge() -> None:
    root = logging.getLogger()
    root.handlers = [_InterceptHandler()]
    root.setLevel(_LOG_LEVEL)
    # Quiet noisy libraries
    for noisy in ("urllib3", "snowflake.connector", "asyncio",
                  "matplotlib", "PIL", "fsspec", "yfinance",
                  "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_install_stdlib_bridge()


# ---- Uncaught-exception hook ---------------------------------------------
def _excepthook(exc_type, exc_value, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, tb)
        return
    log.opt(exception=(exc_type, exc_value, tb)).critical("Uncaught exception")


sys.excepthook = _excepthook


__all__ = ["log"]
