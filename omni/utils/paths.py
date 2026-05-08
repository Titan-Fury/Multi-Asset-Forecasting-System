"""Centralized path management. All other modules read paths from here."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


def project_root() -> Path:
    return _PROJECT_ROOT


def config_dir() -> Path:
    return _PROJECT_ROOT / "config"


def local_data_root() -> Path:
    """Hot/processed local data on SSD."""
    return Path(os.getenv("LOCAL_DATA_ROOT", str(_PROJECT_ROOT / "data")))


def drive_data_root() -> Path:
    """Cold/raw data on Google Drive (mounted via rclone)."""
    return Path(os.getenv("DRIVE_MOUNT_POINT", "Z:\\")) / "Omni_Stocks"


def hot_dir() -> Path:
    return local_data_root() / "hot"


def features_dir() -> Path:
    return local_data_root() / "features"


def models_dir() -> Path:
    return local_data_root() / "models"


def predictions_dir() -> Path:
    return local_data_root() / "predictions"


def cache_dir() -> Path:
    return local_data_root() / "cache"


def llm_cache_dir() -> Path:
    return cache_dir() / "llm"


def ensure_dirs() -> None:
    """Create all standard local dirs if missing."""
    for d in (
        local_data_root(),
        hot_dir(),
        features_dir(),
        models_dir(),
        predictions_dir(),
        cache_dir(),
        llm_cache_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)
