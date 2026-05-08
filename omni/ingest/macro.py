"""Macro ingestion via FRED.

Pulls daily series listed in sources.yaml. Output schema:
  date, series_id, value, source
Dedup key: (series_id, date).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from omni.storage import tiers
from omni.utils.env import get_env, sources
from omni.utils.logger import log

_COLS = ["date", "series_id", "value", "source"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def _fetch_series(series_id: str, *, start: str, end: str) -> pd.DataFrame:
    from fredapi import Fred
    api_key = get_env("FRED_API_KEY", required=True)
    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    if s is None or s.empty:
        return pd.DataFrame(columns=_COLS)
    df = s.reset_index()
    df.columns = ["date", "value"]
    df["series_id"] = series_id
    df["source"] = "fred"
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df[_COLS]


def ingest_fred(*, lookback_years: int = 8) -> dict[str, int]:
    cfg = sources()
    series_ids = cfg.get("macro", {}).get("fred", {}).get("series", [])
    end_dt = datetime.now(timezone.utc).date()
    start_dt = end_dt - timedelta(days=int(lookback_years * 365.25))

    counts: dict[str, int] = {}
    for sid in series_ids:
        try:
            df = _fetch_series(sid, start=start_dt.isoformat(), end=end_dt.isoformat())
            if df.empty:
                counts[sid] = 0
                continue
            tiers.save_raw(df, source="macro_fred", asset_class="macro",
                           ticker=sid, dedup_cols=["series_id", "date"])
            counts[sid] = len(df)
            log.info(f"FRED {sid}: {len(df)} rows")
        except Exception as e:
            log.error(f"FRED ingest failed for {sid}: {e}")
            counts[sid] = -1
    return counts
