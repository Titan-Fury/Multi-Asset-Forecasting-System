"""Tiered storage routing.

Layout convention:
  drive/raw/<source>/<asset_class>/<ticker>.parquet     (raw immutable, on Drive)
  local/hot/<asset_class>/<ticker>.parquet              (cleaned OHLCV, on SSD)
  local/features/<asset_class>/<ticker>.parquet         (engineered features)
  local/predictions/<date>/predictions.parquet          (daily output)
  local/models/<name>_<date>.joblib                     (trained models)

`tiers` is the *only* module the rest of the code should call for I/O routing.
It hides the local-vs-drive split and adds a Drive-down fallback.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from omni.storage import drive as _drive
from omni.storage import local as _local
from omni.utils.logger import log
from omni.utils.paths import (
    features_dir,
    hot_dir,
    models_dir,
    predictions_dir,
)

# ---------------- Raw (cold, Drive) ----------------

def save_raw(df: pd.DataFrame, source: str, asset_class: str, ticker: str,
             *, dedup_cols: list[str] | None = None) -> Path:
    """Write raw ingest output to Drive. Falls back to local hot/raw if Drive unavailable."""
    parts = ("raw", source, asset_class, f"{ticker}.parquet")
    try:
        return _drive.write_raw(df, *parts, dedup_cols=dedup_cols)
    except RuntimeError as e:
        log.warning(f"Drive unavailable, falling back to local: {e}")
        fallback = hot_dir().parent / "raw_fallback" / source / asset_class / f"{ticker}.parquet"
        return _local.append_parquet(df, fallback, dedup_cols=dedup_cols)


def load_raw(source: str, asset_class: str, ticker: str, *, columns: list[str] | None = None) -> pd.DataFrame:
    parts = ("raw", source, asset_class, f"{ticker}.parquet")
    try:
        return _drive.read_raw(*parts, columns=columns)
    except (RuntimeError, FileNotFoundError):
        fallback = hot_dir().parent / "raw_fallback" / source / asset_class / f"{ticker}.parquet"
        return _local.read_parquet(fallback, columns=columns)


# ---------------- Hot (cleaned OHLCV, local) ----------------

def save_hot(df: pd.DataFrame, asset_class: str, ticker: str) -> Path:
    target = hot_dir() / asset_class / f"{ticker}.parquet"
    return _local.write_parquet(df, target)


def load_hot(asset_class: str, ticker: str, *, columns: list[str] | None = None) -> pd.DataFrame:
    target = hot_dir() / asset_class / f"{ticker}.parquet"
    return _local.read_parquet(target, columns=columns)


# ---------------- Features (local) ----------------

def save_features(df: pd.DataFrame, asset_class: str, ticker: str) -> Path:
    target = features_dir() / asset_class / f"{ticker}.parquet"
    return _local.write_parquet(df, target)


def load_features(asset_class: str, ticker: str, *, columns: list[str] | None = None) -> pd.DataFrame:
    target = features_dir() / asset_class / f"{ticker}.parquet"
    return _local.read_parquet(target, columns=columns)


# ---------------- Predictions (local) ----------------

def save_predictions(df: pd.DataFrame, run_date: str) -> Path:
    target = predictions_dir() / run_date / "predictions.parquet"
    return _local.write_parquet(df, target)


def load_predictions(run_date: str) -> pd.DataFrame:
    target = predictions_dir() / run_date / "predictions.parquet"
    return _local.read_parquet(target)


# ---------------- Models (local) ----------------

def model_path(name: str, run_date: str, ext: str = "joblib") -> Path:
    p = models_dir() / f"{name}_{run_date}.{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
