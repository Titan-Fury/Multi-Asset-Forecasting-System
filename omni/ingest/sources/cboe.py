"""CBOE end-of-day equity options summary — free public feed.

CBOE publishes daily summary CSVs of total option volume + put/call ratio
per stock. Used to populate `oi_pcr` and a related signal for US tickers
so the OI rubric pillar isn't permanently neutral 50 for the US side.

Endpoint (free, no key):
    https://www.cboe.com/us/options/market_statistics/daily/
    https://cdn.cboe.com/data/us/options/market_statistics/daily/<YYYY-MM-DD>_options.json

Schema returned by `get_us_pcr(ticker)`:
    {
      pcr:            put_volume / call_volume,
      total_volume:   sum(put + call) in contracts,
      date:           snapshot date,
      has_options:    True/False — gated like nse_oi.has_option_chain,
    }

Cached on disk via `get_cache()` for 1 hour. Best-effort: returns
has_options=False on any error so the downstream rubric path falls back
to neutral 50 (existing graceful contract).
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests

from omni.ondemand.cache import cached
from omni.utils.logger import log


_UA = os.environ.get("CBOE_USER_AGENT",
                      "Omni_Stocks research-noreply@example.com")
_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}
_TIMEOUT = 12


@cached(ttl=3600, key="cboe:daily_summary:{run_date}")
def _daily_summary(run_date: str) -> dict:
    """Pull the CBOE end-of-day options summary JSON for one date.

    Returns the raw dict (contents vary by CBOE schema version).
    Empty dict on any error.
    """
    # CBOE endpoint pattern (subject to schema drift; documented in their
    # market-statistics portal). We try a couple of known patterns.
    urls = [
        f"https://cdn.cboe.com/data/us/options/market_statistics/"
        f"daily/{run_date}_options.json",
        f"https://www.cboe.com/api/global/delayed_quotes/options/"
        f"daily_summary/{run_date}.json",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            if r.status_code != 200:
                continue
            return r.json()
        except Exception as exc:  # noqa: BLE001
            log.debug(f"cboe daily fetch failed {url}: {exc}")
            continue
    return {}


def _normalize(payload: dict) -> pd.DataFrame:
    """Best-effort: extract per-ticker rows from the CBOE response.

    The exact schema varies; we try a handful of common shapes and
    return [ticker, call_volume, put_volume] when we find something
    parseable. Returns empty DataFrame otherwise.
    """
    if not payload or not isinstance(payload, dict):
        return pd.DataFrame()

    # Shape 1: {"data": [{"symbol": "AAPL", "call_volume": ..., "put_volume": ...}]}
    rows = payload.get("data") if isinstance(payload.get("data"), list) else None
    # Shape 2: top-level list of dicts
    if rows is None and isinstance(payload, list):
        rows = payload
    if rows is None:
        return pd.DataFrame()

    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        sym = r.get("symbol") or r.get("ticker") or r.get("underlying")
        if not sym:
            continue
        cv = r.get("call_volume") or r.get("calls") or r.get("call")
        pv = r.get("put_volume") or r.get("puts") or r.get("put")
        if cv is None and pv is None:
            continue
        try:
            out.append({
                "ticker": str(sym).upper(),
                "call_volume": float(cv or 0),
                "put_volume": float(pv or 0),
            })
        except (TypeError, ValueError):
            continue
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)


def get_us_pcr(ticker: str, *, run_date: str | None = None) -> dict:
    """Return PCR + total volume for one US ticker.

    Returns:
      { pcr, total_volume, date, has_options }

    has_options=False on any failure (no data, parse error, ticker not
    found in the daily summary).
    """
    out = {"pcr": 0.0, "total_volume": 0.0,
            "date": run_date or date.today().isoformat(),
            "has_options": False}
    rd = run_date or date.today().isoformat()

    # Try today, then yesterday (CBOE often lags by 1 business day)
    for days_back in (0, 1, 2, 3):
        try_date = (pd.Timestamp(rd) - pd.Timedelta(days=days_back)).date().isoformat()
        payload = _daily_summary(try_date)
        df = _normalize(payload)
        if df.empty:
            continue
        sub = df[df["ticker"] == ticker.upper()]
        if sub.empty:
            continue
        r = sub.iloc[0]
        cv = float(r.get("call_volume", 0) or 0)
        pv = float(r.get("put_volume", 0) or 0)
        if cv <= 0 and pv <= 0:
            continue
        out.update({
            "pcr": round(pv / cv, 3) if cv > 0 else 0.0,
            "total_volume": cv + pv,
            "date": try_date,
            "has_options": True,
        })
        return out
    return out


__all__ = ["get_us_pcr"]
