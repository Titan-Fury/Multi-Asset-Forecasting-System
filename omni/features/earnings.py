"""Earnings calendar features.

Expected 'days until next earnings' is one of the highest-information
features for short-horizon prediction — vol explodes around the date,
post-earnings drift creates persistent alpha for ~5d after.

Public API:
    add_earnings_features(df, *, ticker, asset_class) -> DataFrame
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from omni.utils.logger import log


def _next_earnings_date(ticker: str, asset_class: str) -> pd.Timestamp | None:
    """Best-effort fetch of the next earnings date via yfinance."""
    if asset_class in ("crypto", "fx"):
        return None
    try:
        import yfinance as yf
        # Idempotent suffix: skip if ticker already has any Yahoo suffix.
        if asset_class == "india" and "." not in ticker and not ticker.startswith("^"):
            sym = f"{ticker}.NS"
        else:
            sym = ticker
        cal = yf.Ticker(sym).calendar
        if cal is None:
            return None
        if isinstance(cal, dict):
            er = cal.get("Earnings Date")
        else:
            er = None
        if er is None:
            return None
        # Unwrap single- or empty-list shapes that yfinance occasionally returns
        if isinstance(er, (list, tuple)):
            if not er:
                return None
            er = er[0]
            if isinstance(er, (list, tuple)):
                if not er:
                    return None
                er = er[0]
        try:
            ts = pd.Timestamp(er)
        except Exception:
            return None
        return ts.tz_localize(None) if ts.tzinfo else ts
    except Exception as e:
        log.warning(f"earnings calendar fetch failed {ticker}: {e}")
        return None


def add_earnings_features(df: pd.DataFrame, *, ticker: str,
                            asset_class: str) -> pd.DataFrame:
    """Add ``days_until_earnings`` and ``earnings_window`` flags."""
    if df is None or df.empty:
        return df
    next_er = _next_earnings_date(ticker, asset_class)
    out = df.copy()
    if next_er is None:
        out["days_until_earnings"] = np.nan
        out["earnings_window"] = 0
        return out
    dates = pd.to_datetime(out["date"]).dt.tz_localize(None)
    delta = (next_er - dates).dt.days
    out["days_until_earnings"] = delta
    # 1 if within 5 trading days (~7 calendar days) before earnings
    out["earnings_window"] = ((delta >= 0) & (delta <= 7)).astype(int)
    return out
