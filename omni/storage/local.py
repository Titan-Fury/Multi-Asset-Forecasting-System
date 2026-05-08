"""Local SSD parquet read/write helpers (hot tier)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from omni.utils.logger import log


def write_parquet(df: pd.DataFrame | pl.DataFrame, path: Path, *, partition_cols: list[str] | None = None) -> Path:
    """Write a DataFrame to parquet (snappy). Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    if df.empty:
        log.warning(f"write_parquet: empty df for {path} (skipping)")
        return path

    if partition_cols:
        df.to_parquet(path, partition_cols=partition_cols, compression="snappy", index=False)
    else:
        df.to_parquet(path, compression="snappy", index=False)
    log.debug(f"wrote parquet -> {path} ({len(df)} rows)")
    return path


def read_parquet(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, columns=columns)


def scan_parquet_lazy(path: Path) -> pl.LazyFrame:
    """Return a Polars LazyFrame for memory-efficient column-pruned reads."""
    return pl.scan_parquet(str(path))


def append_parquet(df: pd.DataFrame, path: Path, *, dedup_cols: list[str] | None = None) -> Path:
    """Append rows to an existing parquet, deduplicated by `dedup_cols`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    combined.to_parquet(path, compression="snappy", index=False)
    log.debug(f"appended parquet -> {path} (now {len(combined)} rows)")
    return path


def list_parquets(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.parquet"))


def file_info(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "size_mb": round(st.st_size / 1_048_576, 3),
        "mtime": st.st_mtime,
    }
