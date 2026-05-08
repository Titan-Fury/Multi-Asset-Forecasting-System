"""Polygon.io EOD price client.

Free tier: 5 requests/min, EOD only, ~2y history.
We honor the rate limit with an in-process token bucket.

Schema returned (matches omni.ingest.prices output):
  date, open, high, low, close, adj_close, volume, ticker, asset_class, source
"""
from __future__ import annotations

import os
import threading
import time
from datetime import date, timedelta

import pandas as pd
import requests

from omni.utils.logger import log

_OHLCV = ["date", "open", "high", "low", "close", "adj_close", "volume"]
_BASE = "https://api.polygon.io"
_LOCK = threading.Lock()
_LAST_CALLS: list[float] = []  # epoch seconds of last 5 calls


def _wait_for_slot() -> None:
    """Block until making a call won't exceed 5 calls per 60s."""
    with _LOCK:
        now = time.monotonic()
        # drop calls older than 60s
        cutoff = now - 60.0
        _LAST_CALLS[:] = [t for t in _LAST_CALLS if t > cutoff]
        if len(_LAST_CALLS) >= 5:
            sleep_for = 60.0 - (now - _LAST_CALLS[0]) + 0.5
            if sleep_for > 0:
                log.debug(f"polygon rate-limit wait {sleep_for:.1f}s")
                time.sleep(sleep_for)
        _LAST_CALLS.append(time.monotonic())


def _key() -> str | None:
    k = os.getenv("POLYGON_API_KEY", "").strip()
    return k or None


def eod_prices(asset_class: str, ticker: str, *, lookback_days: int = 730) -> pd.DataFrame:
    """Fetch end-of-day OHLCV for a single equity ticker.

    Polygon supports US equities (no suffix), forex (C:EURUSD), and crypto
    (X:BTCUSD). For non-US asset classes we map the symbol; FX and crypto
    on free tier may return empty — caller handles via fallback.
    """
    key = _key()
    if not key:
        return pd.DataFrame(columns=[*_OHLCV, "ticker", "asset_class", "source"])

    sym = _polygon_symbol(asset_class, ticker)
    if sym is None:
        return pd.DataFrame(columns=[*_OHLCV, "ticker", "asset_class", "source"])

    end = date.today()
    start = end - timedelta(days=lookback_days)
    url = (
        f"{_BASE}/v2/aggs/ticker/{sym}/range/1/day/"
        f"{start.isoformat()}/{end.isoformat()}"
    )
    _wait_for_slot()
    try:
        r = requests.get(url, params={"adjusted": "true", "sort": "asc",
                                       "limit": 50000, "apiKey": key}, timeout=30)
        if r.status_code == 429:
            log.warning("polygon 429; retry after 60s")
            time.sleep(60)
            r = requests.get(url, params={"adjusted": "true", "sort": "asc",
                                           "limit": 50000, "apiKey": key}, timeout=30)
        if r.status_code != 200:
            log.warning(f"polygon {r.status_code}: {r.text[:200]}")
            return pd.DataFrame(columns=[*_OHLCV, "ticker", "asset_class", "source"])
        results = r.json().get("results") or []
    except Exception as e:
        log.warning(f"polygon fetch failed for {sym}: {e}")
        return pd.DataFrame(columns=[*_OHLCV, "ticker", "asset_class", "source"])

    if not results:
        return pd.DataFrame(columns=[*_OHLCV, "ticker", "asset_class", "source"])

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume"})
    df["adj_close"] = df["close"]  # polygon adjusted=true already gave adjusted
    df = df[_OHLCV].copy()
    df["ticker"] = ticker
    df["asset_class"] = asset_class
    df["source"] = "polygon"
    return df


def _polygon_symbol(asset_class: str, ticker: str) -> str | None:
    if asset_class == "us":
        return ticker.replace("-", ".")  # BRK-B -> BRK.B
    if asset_class == "crypto":
        return f"X:{ticker}USD"
    if asset_class == "fx":
        return f"C:{ticker}"  # EURUSD -> C:EURUSD
    if asset_class == "india":
        return None  # not supported on free tier
    return None
