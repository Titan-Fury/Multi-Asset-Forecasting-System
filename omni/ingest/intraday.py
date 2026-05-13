"""Intraday (sub-daily) bar fetcher.

Stateless, cached at the news TTL (30 min). Returns 5-minute bars for the
last `days` trading sessions, asset-class aware:

  india  -> yfinance .NS, period=Nd, interval=5m
  us     -> yfinance, period=Nd, interval=5m
  crypto -> ccxt 5m candles (binance)
  fx     -> yfinance =X, period=Nd, interval=5m

Output schema: [ts, open, high, low, close, volume, vwap]
where vwap is the cumulative session VWAP recomputed within each trading day.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from omni.ingest.prices import _yf_symbol
from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log


_COLS = ["ts", "open", "high", "low", "close", "volume", "vwap"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLS)


def _add_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative VWAP within each calendar date."""
    if df.empty:
        df["vwap"] = pd.Series(dtype="float64")
        return df
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    day = df["ts"].dt.normalize()
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    df["vwap"] = (cum_pv / cum_vol).fillna(df["close"])
    return df


@cached(ttl=TTL.news, key="intraday:{asset_class}:{ticker}:{interval}:{days}")
def fetch_intraday_bars(
    asset_class: str,
    ticker: str,
    *,
    interval: str = "5m",
    days: int = 5,
) -> pd.DataFrame:
    """Fetch the last `days` trading sessions of intraday bars.

    Returns DataFrame with [ts, open, high, low, close, volume, vwap].
    Empty DataFrame on any failure (off-hours, weekends, network errors).
    """
    try:
        if asset_class == "crypto":
            return _fetch_ccxt_intraday(ticker, interval=interval, days=days)
        return _fetch_yf_intraday(asset_class, ticker, interval=interval, days=days)
    except KeyError as e:
        # yfinance raises bare KeyError(symbol) when a ticker has no intraday
        # data — common for BSE-only stocks, indices like ^NSEI, weekends.
        # Expected, not actionable; downgrade to DEBUG to keep logs clean.
        log.debug(f"intraday no data {asset_class}/{ticker}: {e}")
        return _empty()
    except Exception as e:
        log.warning(f"intraday fetch failed {asset_class}/{ticker}: {e}")
        return _empty()


def _yf_history_with_bo_fallback(asset_class: str, ticker: str,
                                  *, period: str, interval: str):
    """yfinance Ticker.history() with the same NSE -> BSE fallback used by
    `omni.ingest.prices.fetch_prices`. India small/mid-caps like CIANAGRO
    and OMAXAUTO trade only on BSE; if the .NS suffix returns empty (Yahoo
    emits a 404 + 'possibly delisted' for them), retry once with .BO.
    Returns (df, sym_used) so the caller can log which symbol actually
    produced data. Empty df + original sym on hard failure."""
    import yfinance as yf
    sym = _yf_symbol(asset_class, ticker)
    df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
    if (df is None or df.empty) and asset_class == "india" and "." not in ticker:
        sym_bo = f"{ticker}.BO"
        df_bo = yf.Ticker(sym_bo).history(period=period, interval=interval, auto_adjust=False)
        if df_bo is not None and not df_bo.empty:
            return df_bo, sym_bo
    return df, sym


def _fetch_yf_intraday(asset_class: str, ticker: str, *, interval: str, days: int) -> pd.DataFrame:
    period = f"{max(1, min(days, 60))}d"  # yf 5m max 60d
    df, _sym = _yf_history_with_bo_fallback(
        asset_class, ticker, period=period, interval=interval,
    )
    if df is None or df.empty:
        return _empty()
    df = df.reset_index().rename(columns={
        "Datetime": "ts", "Date": "ts",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).reset_index(drop=True)
    if df.empty:
        return _empty()
    # Strip tz to keep things simple (we only care about ordering / day boundary)
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_convert(None)
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df = df.sort_values("ts").reset_index(drop=True)
    df = _add_session_vwap(df)
    return df[_COLS]


def _fetch_ccxt_intraday(ticker: str, *, interval: str, days: int) -> pd.DataFrame:
    try:
        import ccxt  # type: ignore
    except ImportError:
        log.warning("ccxt not installed; falling back to yfinance for crypto intraday")
        return _fetch_yf_intraday("crypto", ticker, interval=interval, days=days)
    ex = ccxt.binance({"enableRateLimit": True})
    tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h"}
    tf = tf_map.get(interval, "5m")
    symbol = f"{ticker}/USDT"
    since = ex.milliseconds() - days * 24 * 60 * 60 * 1000
    rows = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=1000)
    if not rows:
        return _empty()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)
    df = _add_session_vwap(df)
    return df[_COLS]


def current_session_bars(asset_class: str, ticker: str) -> pd.DataFrame:
    """Today's bars only (last calendar day in the fetched data)."""
    df = fetch_intraday_bars(asset_class, ticker, days=2)
    if df.empty:
        return df
    last_day = df["ts"].dt.normalize().max()
    return df[df["ts"].dt.normalize() == last_day].reset_index(drop=True)
