"""Lightweight earnings calendar for the dashboard.

`omni/features/earnings.py` already wraps yfinance for *next* earnings
date (used as a model feature). This module adds the second piece:
a small disk-cached fetcher for **historical** earnings rows
(reported EPS / estimate / surprise %) plus a clean public API the
dashboard's Earnings tab can call.

Sources:
  - yfinance `Ticker(sym).earnings_dates` (last 4 quarters, surprise%)
  - yfinance `Ticker(sym).calendar` (next reporting date)

Both are best-effort; we return None / empty on any failure so the UI
can degrade gracefully.
"""
from __future__ import annotations

import pandas as pd

from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log


def _yf_symbol(asset_class: str, ticker: str) -> str | None:
    """Replicate the convention used by omni.features.earnings.

    Routes known BSE-only India tickers (CIANAGRO etc.) straight to ``.BO``
    via the shared `is_bse_only` registry in ``omni.ingest.prices``; without
    this they'd hit ``.NS`` and 404 on every earnings-calendar fetch."""
    if asset_class in ("crypto", "fx"):
        return None
    sym = str(ticker).upper()
    if asset_class == "india" and not sym.endswith((".NS", ".BO")):
        from omni.ingest.prices import is_bse_only
        sym = sym + (".BO" if is_bse_only(sym) else ".NS")
    return sym


@cached(ttl=TTL.fundamentals, key="earnings_next:{asset_class}:{ticker}")
def next_earnings(ticker: str, asset_class: str) -> pd.Timestamp | None:
    """Next scheduled earnings date for an equity. None if unavailable."""
    sym = _yf_symbol(asset_class, ticker)
    if sym is None:
        return None
    try:
        import yfinance as yf  # type: ignore
        cal = yf.Ticker(sym).calendar
    except Exception as e:  # noqa: BLE001
        log.debug(f"earnings_calendar.next_earnings({ticker}): {e}")
        return None
    if cal is None or len(cal) == 0:
        return None
    # yfinance returns either a DataFrame (legacy) or a dict (newer)
    try:
        if isinstance(cal, dict):
            d = cal.get("Earnings Date")
            if isinstance(d, list) and d:
                return pd.Timestamp(d[0])
            if d is not None:
                return pd.Timestamp(d)
            return None
        # DataFrame path
        if "Earnings Date" in cal.index:
            v = cal.loc["Earnings Date"]
            v = v.iloc[0] if hasattr(v, "iloc") else v
            return pd.Timestamp(v)
    except Exception:
        return None
    return None


@cached(ttl=TTL.fundamentals, key="earnings_hist:{asset_class}:{ticker}")
def earnings_history(ticker: str, asset_class: str,
                      *, lookback_quarters: int = 8) -> pd.DataFrame:
    """Historical earnings rows: reported EPS, estimate, surprise %.

    Returns empty DataFrame on any failure.
    """
    sym = _yf_symbol(asset_class, ticker)
    if sym is None:
        return pd.DataFrame()
    try:
        import yfinance as yf  # type: ignore
        edf = yf.Ticker(sym).earnings_dates
    except Exception as e:  # noqa: BLE001
        log.debug(f"earnings_calendar.earnings_history({ticker}): {e}")
        return pd.DataFrame()
    if edf is None or len(edf) == 0:
        return pd.DataFrame()
    try:
        d = edf.copy().reset_index()
        d.columns = [str(c).strip() for c in d.columns]
        rename_map = {
            "Earnings Date": "earnings_date",
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_reported",
            "Surprise(%)": "surprise_pct",
        }
        d = d.rename(columns={k: v for k, v in rename_map.items() if k in d.columns})
        if "earnings_date" in d.columns:
            d["earnings_date"] = pd.to_datetime(d["earnings_date"], errors="coerce")
            d = d.sort_values("earnings_date", ascending=False)
        keep = [c for c in ("earnings_date", "eps_estimate", "eps_reported",
                              "surprise_pct") if c in d.columns]
        return d[keep].head(lookback_quarters).reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        log.debug(f"earnings_history reshape failed: {e}")
        return pd.DataFrame()
