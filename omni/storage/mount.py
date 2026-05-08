"""Mount/unmount stub.

In Colab/Drive setups this would mount Google Drive at /content/drive. On
local Windows/Mac/Linux dev boxes it's a no-op so the orchestrator stages
remain importable and the pipeline runs against the local data tier.

Replace with the real implementation when the storage tier moves to Drive.
"""
from __future__ import annotations

import os

from omni.utils.logger import log


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount(path: str = "/content/drive") -> bool:
    """Mount Google Drive in Colab. Returns True if mounted, False if skipped."""
    if not is_colab():
        log.info("storage.mount: not in Colab, skipping (using local data dir)")
        return False
    if os.path.ismount(path):
        log.info(f"storage.mount: already mounted at {path}")
        return True
    try:
        from google.colab import drive
        drive.mount(path)
        return True
    except Exception as e:
        log.warning(f"storage.mount failed: {e}")
        return False


def unmount(path: str = "/content/drive") -> bool:
    if not is_colab():
        return False
    try:
        from google.colab import drive
        drive.flush_and_unmount()
        log.info("storage.mount: unmounted")
        return True
    except Exception as e:
        log.warning(f"storage.mount unmount failed: {e}")
        return False
