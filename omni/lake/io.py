"""Parquet I/O helpers for the data lake.

All functions here are stateless and idempotent. They never raise on missing
files — they return empty DataFrames so callers can branch cleanly.

Design choices:
- Use pyarrow under the hood (via pandas) — fast and stable.
- Per-ticker files for prices (small, parallelisable, easy to inspect).
- Date-partitioned files for daily snapshots (fundamentals, news, predictions).
- Append-only files use a (ticker, date) dedup key on read+write.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from omni.lake import paths
from omni.utils.logger import log


_PRICE_COLS = ["date", "open", "high", "low", "close", "adj_close", "volume",
                "ticker", "asset_class", "source"]


# ---------------------------------------------------------------- prices ----
def read_prices(ticker: str, asset_class: str) -> pd.DataFrame:
    """Read full price history for one ticker. Empty DataFrame if not in lake."""
    p = paths.prices_path(asset_class, ticker)
    if not p.exists():
        return pd.DataFrame(columns=_PRICE_COLS)
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)


def write_prices(df: pd.DataFrame, ticker: str, asset_class: str) -> int:
    """Overwrite the per-ticker parquet. Returns row count written."""
    if df is None or df.empty:
        log.warning(f"lake.write_prices: empty frame for {asset_class}/{ticker}")
        return 0
    p = paths.ensure_parent(paths.prices_path(asset_class, ticker))
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    out.to_parquet(p, index=False)
    return len(out)


def append_prices(new_rows: pd.DataFrame, ticker: str, asset_class: str) -> int:
    """Append rows, dedup on date. Returns total row count after merge."""
    if new_rows is None or new_rows.empty:
        return len(read_prices(ticker, asset_class))
    existing = read_prices(ticker, asset_class)
    merged = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    return write_prices(merged, ticker, asset_class)


def last_price_date(ticker: str, asset_class: str) -> pd.Timestamp | None:
    """Return the most recent date in the lake for this ticker, or None."""
    df = read_prices(ticker, asset_class)
    if df.empty:
        return None
    return pd.Timestamp(df["date"].max())


# ----------------------------------------------------- fundamentals snapshot
def write_fundamentals_snapshot(df: pd.DataFrame, date_iso: str) -> int:
    if df is None or df.empty:
        return 0
    p = paths.ensure_parent(paths.fundamentals_path(date_iso))
    df.to_parquet(p, index=False)
    return len(df)


def read_fundamentals_asof(date_iso: str) -> pd.DataFrame:
    """Read the most recent fundamentals snapshot at or before ``date_iso``."""
    d = paths.fundamentals_dir()
    if not d.exists():
        return pd.DataFrame()
    candidates = sorted([f for f in d.glob("*.parquet") if f.stem <= date_iso],
                          reverse=True)
    if not candidates:
        return pd.DataFrame()
    return pd.read_parquet(candidates[0])


# -------------------------------------------------------------------- news --
def write_news_day(df: pd.DataFrame, date_iso: str) -> int:
    if df is None or df.empty:
        return 0
    p = paths.ensure_parent(paths.news_path(date_iso))
    df.to_parquet(p, index=False)
    return len(df)


def read_news_window(start_iso: str, end_iso: str) -> pd.DataFrame:
    d = paths.news_dir()
    if not d.exists():
        return pd.DataFrame()
    parts = []
    for f in sorted(d.glob("*.parquet")):
        if start_iso <= f.stem <= end_iso:
            parts.append(pd.read_parquet(f))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


# ------------------------------------------------------------------- manifest
def update_manifest(updates: dict) -> dict:
    """Merge ``updates`` into the manifest and persist it. Returns the new manifest."""
    p = paths.manifest_path()
    cur: dict = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"manifest read failed: {e}; rebuilding")
            cur = {}
    cur.update(updates)
    cur["updated_at"] = datetime.now(timezone.utc).isoformat()
    paths.ensure_parent(p)
    p.write_text(json.dumps(cur, indent=2, sort_keys=True), encoding="utf-8")
    return cur


def read_manifest() -> dict:
    p = paths.manifest_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------- summary helpers --
def lake_summary() -> dict:
    """Return a one-shot summary of what's currently in the lake."""
    out = {
        "lake_root": str(paths.lake_root()),
        "tickers_with_prices": 0,
        "by_asset_class": {},
        "fundamentals_snapshots": 0,
        "news_days": 0,
        "predictions_days": 0,
        "manifest": read_manifest(),
    }
    pdir = paths.lake_root() / "prices"
    if pdir.exists():
        for ac_dir in pdir.iterdir():
            if not ac_dir.is_dir():
                continue
            n = len(list(ac_dir.glob("*.parquet")))
            out["by_asset_class"][ac_dir.name] = n
            out["tickers_with_prices"] += n
    out["fundamentals_snapshots"] = (
        len(list(paths.fundamentals_dir().glob("*.parquet")))
        if paths.fundamentals_dir().exists() else 0
    )
    out["news_days"] = (
        len(list(paths.news_dir().glob("*.parquet")))
        if paths.news_dir().exists() else 0
    )
    out["predictions_days"] = (
        len(list(paths.predictions_dir().glob("*.parquet")))
        if paths.predictions_dir().exists() else 0
    )
    return out
