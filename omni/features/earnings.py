"""Earnings calendar features.

Expected 'days until next earnings' is one of the highest-information
features for short-horizon prediction — vol explodes around the date,
post-earnings drift creates persistent alpha for ~5d after.

Public API:
    add_earnings_features(df, *, ticker, asset_class) -> DataFrame
"""
from __future__ import annotations

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
            from omni.ingest.prices import is_bse_only
            sym = f"{ticker}.BO" if is_bse_only(ticker) else f"{ticker}.NS"
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
    """Add ``days_until_earnings`` and ``earnings_window`` flags.

    Phase 24.22 — fixes a critical look-ahead leakage bug. The previous
    implementation called yfinance's *current* next-earnings calendar and
    broadcast that single future date across EVERY historical row, so a
    sample from 6 months ago was getting `days_until_earnings = -180` and
    a sample from 3 months ago was getting `days_until_earnings = -90`.
    The model effectively learned "this row is from N months before today"
    which is pure target leakage.

    Until we wire a point-in-time earnings calendar (per-ticker historical
    dates), only the LAST row in the frame -- which represents the live
    inference moment -- gets a live date. All historical rows get NaN
    days_until_earnings and earnings_window=0 so the model cannot fit on
    a phantom signal.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["days_until_earnings"] = np.nan
    out["earnings_window"] = 0

    # Only the most recent row is "today" -- the live calendar applies
    # there. For historical rows, the next-earnings date that yfinance
    # returns now is in the future relative to the row date, which makes
    # `(next_er - row_date).dt.days` a leak of how far that row is from
    # today.
    next_er = _next_earnings_date(ticker, asset_class)
    if next_er is None or "date" not in out.columns or len(out) == 0:
        return out
    last_idx = out.index[-1]
    last_date = pd.Timestamp(out.loc[last_idx, "date"])
    if pd.isna(last_date):
        return out
    last_date = last_date.tz_localize(None) if last_date.tzinfo else last_date
    delta = int((next_er - last_date).days)
    # Only meaningful in the FUTURE -- if next_er is already past the last
    # row date for some reason (stale yfinance cache), leave NaN.
    if delta < 0:
        return out
    out.loc[last_idx, "days_until_earnings"] = float(delta)
    out.loc[last_idx, "earnings_window"] = int(0 <= delta <= 7)
    return out
