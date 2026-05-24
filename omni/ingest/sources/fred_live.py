"""FRED macro time-series client (live, no tier writes).

Free tier: unlimited (with key). Used by macro_live to assemble the
macro feature frame on-demand without depending on the raw tier.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import requests

from omni.utils.logger import log


def _key() -> str | None:
    k = os.getenv("FRED_API_KEY", "").strip()
    return k or None


def series(series_id: str, *, lookback_years: int = 10) -> pd.DataFrame:
    """Fetch a single FRED series. Returns DataFrame[date, series_id, value]."""
    key = _key()
    if not key:
        return pd.DataFrame(columns=["date", "series_id", "value"])
    end = date.today()
    start = end - timedelta(days=int(lookback_years * 365.25))

    # Phase 24.21 — FRED returns intermittent 500s during their nightly
    # maintenance windows. Was failing fast and dropping the macro series for
    # the whole day; now we retry with exponential backoff so a single 500
    # blip doesn't leave us with empty CPI/VIX/UST features.
    last_text = ""
    for attempt in range(3):
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": series_id, "api_key": key, "file_type": "json",
                         "observation_start": start.isoformat(),
                         "observation_end": end.isoformat()},
                timeout=30,
            )
            if r.status_code == 200:
                try:
                    obs = r.json().get("observations", [])
                except ValueError:
                    log.warning(f"fred {series_id}: malformed JSON response")
                    return pd.DataFrame(columns=["date", "series_id", "value"])
                break
            last_text = r.text[:200]
            if r.status_code in (500, 502, 503, 504) and attempt < 2:
                # backoff 2s, 4s
                import time as _t
                _t.sleep(2 ** (attempt + 1))
                continue
            log.warning(f"fred {series_id} {r.status_code}: {last_text}")
            return pd.DataFrame(columns=["date", "series_id", "value"])
        except Exception as e:
            if attempt < 2:
                import time as _t
                _t.sleep(2 ** (attempt + 1))
                continue
            log.warning(f"fred fetch failed for {series_id}: {e}")
            return pd.DataFrame(columns=["date", "series_id", "value"])

    rows = []
    for o in obs:
        v = o.get("value")
        if v in (None, ".", ""):
            continue
        try:
            rows.append({"date": pd.to_datetime(o["date"]),
                          "series_id": series_id, "value": float(v)})
        except (TypeError, ValueError):
            continue
    return pd.DataFrame(rows, columns=["date", "series_id", "value"])


def many(series_ids: list[str], *, lookback_years: int = 10) -> pd.DataFrame:
    """Concat multiple FRED series into one long-form frame."""
    parts = [series(sid, lookback_years=lookback_years) for sid in series_ids]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["date", "series_id", "value"])
    return pd.concat(parts, ignore_index=True)
