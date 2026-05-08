"""Google Drive (rclone-mounted) parquet helpers for cold/raw tier.

Mount must be active (Z:\\) before calling these. Use omni.storage.tiers
for higher-level routing that handles fallback.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from omni.storage import local as _local
from omni.utils.logger import log
from omni.utils.paths import drive_data_root


def is_mounted() -> bool:
    """Cheap check: does the drive root exist and is it readable?"""
    root = drive_data_root().parent  # Z:\
    try:
        return root.exists() and any(root.iterdir()) is not None
    except OSError:
        return False


def ensure_mounted() -> None:
    if not is_mounted():
        raise RuntimeError(
            f"Google Drive not mounted at {drive_data_root().parent}. "
            "Run Mount-OmniDrive.bat or the daily orchestrator's mount step."
        )


def drive_path(*parts: str) -> Path:
    """Build a path under the Drive root, creating parents on demand."""
    p = drive_data_root().joinpath(*parts)
    return p


def write_raw(df: pd.DataFrame, *parts: str, dedup_cols: list[str] | None = None) -> Path:
    """Append-write raw data to Drive. Mount required."""
    ensure_mounted()
    target = drive_path(*parts)
    return _local.append_parquet(df, target, dedup_cols=dedup_cols)


def read_raw(*parts: str, columns: list[str] | None = None) -> pd.DataFrame:
    ensure_mounted()
    target = drive_path(*parts)
    return _local.read_parquet(target, columns=columns)


def exists(*parts: str) -> bool:
    return drive_path(*parts).exists()


def info(*parts: str) -> dict:
    return _local.file_info(drive_path(*parts))


def list_raw(*parts: str) -> list[Path]:
    return _local.list_parquets(drive_path(*parts))
