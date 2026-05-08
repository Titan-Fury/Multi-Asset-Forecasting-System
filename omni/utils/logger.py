"""Loguru-based logger. Import `log` and use directly."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger as log

from omni.utils.paths import project_root

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

log.remove()
log.add(
    sys.stderr,
    level=_LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

_log_file: Path = project_root() / "data" / "omni.log"
_log_file.parent.mkdir(parents=True, exist_ok=True)
log.add(
    _log_file,
    level=_LOG_LEVEL,
    rotation="10 MB",
    retention=5,
    encoding="utf-8",
    enqueue=True,
)

__all__ = ["log"]
