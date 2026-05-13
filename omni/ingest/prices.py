"""Price ingestion.

Routes by asset_class:
  india  -> yfinance (with .NS suffix), nsepython fallback
  us     -> yfinance
  crypto -> ccxt (binance) primary, yfinance fallback
  fx     -> yfinance (=X suffix)

Output schema (parquet, raw tier):
  date (datetime64[ns]), open, high, low, close, adj_close, volume, ticker, asset_class, source
Dedup key: (ticker, date).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from omni.storage import tiers
from omni.utils.env import universe
from omni.utils.logger import log

_OHLCV_COLS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


# India equities that trade ONLY on BSE (not NSE). yfinance's ".NS" fetch
# returns 404 + writes to stderr for these symbols on every attempt — and
# tenacity retries amplify the noise — so we route them straight to ".BO".
# Maintained manually; add a name when you see repeated 404s for it in the
# logs. Keeping it a small explicit set is preferable to a runtime probe
# (which would itself emit a 404 to learn that fact).
_BSE_ONLY_INDIA: frozenset[str] = frozenset({
    "CIANAGRO",
    "OMAXAUTO",
    "HINDMOTORS",
})


def is_bse_only(ticker: str) -> bool:
    """True if `ticker` is an India equity that trades only on BSE.
    Bare ticker only (no suffix) — caller normalises before checking."""
    return ticker.upper() in _BSE_ONLY_INDIA


def _yf_symbol(asset_class: str, ticker: str) -> str:
    # Indexes (e.g. ^NSEI, ^GSPC) and already-suffixed tickers pass through.
    if ticker.startswith("^") or "=" in ticker or "-" in ticker or "." in ticker:
        return ticker
    # Known BSE-only India names → .BO directly, skipping the .NS 404 hop.
    if asset_class == "india" and is_bse_only(ticker):
        return f"{ticker}.BO"
    suffix_map = {"india": ".NS", "us": "", "crypto": "-USD", "fx": "=X"}
    return f"{ticker}{suffix_map[asset_class]}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def _fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=_OHLCV_COLS)
    # Flatten possible MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    return df[_OHLCV_COLS]


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5), reraise=True)
def _fetch_ccxt(ticker: str, start: str, end: str, exchange: str = "binance") -> pd.DataFrame:
    import ccxt
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    sym = f"{ticker}/USDT"
    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    out: list[list] = []
    cur = since_ms
    while cur < end_ms:
        batch = ex.fetch_ohlcv(sym, timeframe="1d", since=cur, limit=1000)
        if not batch:
            break
        out.extend(batch)
        cur = batch[-1][0] + 86_400_000
        if len(batch) < 1000:
            break
    if not out:
        return pd.DataFrame(columns=_OHLCV_COLS)
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.tz_localize(None)
    df["adj_close"] = df["close"]
    return df[_OHLCV_COLS]


def fetch_prices(asset_class: str, ticker: str, *, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV for a single (asset_class, ticker). Tries primary, falls back on empty/error."""
    df: pd.DataFrame
    if asset_class == "crypto":
        try:
            df = _fetch_ccxt(ticker, start, end)
        except Exception as e:
            log.warning(f"ccxt failed for {ticker}: {e}; falling back to yfinance")
            df = _fetch_yfinance(_yf_symbol(asset_class, ticker), start, end)
    else:
        df = _fetch_yfinance(_yf_symbol(asset_class, ticker), start, end)
        # India small/mid caps can be listed only on BSE in Yahoo. If the
        # NSE suffix returns no rows OR a suspiciously thin result (<30
        # trading days, the typical "delisted phantom" Yahoo emits for
        # invalid NSE symbols), retry once with .BO.
        if (asset_class == "india" and "." not in ticker
                and (df.empty or len(df) < 30)):
            df_bo = _fetch_yfinance(f"{ticker}.BO", start, end)
            if not df_bo.empty and len(df_bo) > len(df):
                df = df_bo

    if df.empty:
        log.warning(f"No price rows fetched for {asset_class}/{ticker}")
        return df

    df["ticker"] = ticker
    df["asset_class"] = asset_class
    df["source"] = "yfinance" if asset_class != "crypto" else "ccxt"
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    # yfinance inserts a placeholder row for *today* with NaN OHLC until the
    # session closes. Drop any trailing NaN-close rows so feature engineering
    # uses only finalized bars.
    df = df.dropna(subset=["adj_close"]).reset_index(drop=True)
    return df


def latest_intraday(asset_class: str, ticker: str) -> dict:
    """Best-effort fetch of the latest 5-minute intraday bar.

    Used by the dashboard to surface a 'live' price even when the day's
    daily bar hasn't finalized yet. Never raises — returns an empty dict
    on any failure.

    NSE -> BSE fallback for India small/mid-caps (e.g. CIANAGRO, OMAXAUTO
    are BSE-only on Yahoo); mirrors the fallback already in fetch_prices().
    """
    try:
        import yfinance as yf
        sym = _yf_symbol(asset_class, ticker)
        df = yf.Ticker(sym).history(period="1d", interval="5m")
        if (df is None or df.empty) and asset_class == "india" and "." not in ticker:
            df = yf.Ticker(f"{ticker}.BO").history(period="1d", interval="5m")
        if df is None or df.empty:
            return {}
        last = df.dropna(subset=["Close"]).tail(1)
        if last.empty:
            return {}
        ts = last.index[-1]
        return {
            "last_intraday_price": float(last["Close"].iloc[-1]),
            "last_intraday_at": str(pd.Timestamp(ts).isoformat()),
            "last_intraday_volume": int(last["Volume"].iloc[-1] or 0),
        }
    except Exception as e:
        log.warning(f"intraday fetch failed {asset_class}/{ticker}: {e}")
        return {}


def ingest_universe(*, lookback_years: int | None = None, end: str | None = None) -> dict[str, int]:
    """Ingest prices for the entire configured universe and persist via tiers.save_raw."""
    cfg = universe()
    end_dt = datetime.now(timezone.utc).date() if end is None else pd.Timestamp(end).date()
    years = lookback_years or cfg.get("lookback_years", 8)
    start_dt = end_dt - timedelta(days=int(years * 365.25))
    start, end = start_dt.isoformat(), end_dt.isoformat()

    counts: dict[str, int] = {}
    for asset_class, spec in cfg["asset_classes"].items():
        for ticker in spec["tickers"]:
            try:
                df = fetch_prices(asset_class, ticker, start=start, end=end)
                if df.empty:
                    counts[f"{asset_class}/{ticker}"] = 0
                    continue
                tiers.save_raw(df, source="prices", asset_class=asset_class,
                               ticker=ticker, dedup_cols=["ticker", "date"])
                # Hot tier: cleaned OHLCV (same shape for v1)
                tiers.save_hot(df, asset_class=asset_class, ticker=ticker)
                counts[f"{asset_class}/{ticker}"] = len(df)
                log.info(f"ingested {asset_class}/{ticker}: {len(df)} rows")
            except Exception as e:
                log.error(f"ingest failed {asset_class}/{ticker}: {e}")
                counts[f"{asset_class}/{ticker}"] = -1
    return counts


def ingest_macro(*, lookback_years: int | None = None) -> dict[str, int]:
    """Ingest macro reference series from yfinance."""
    cfg = universe()
    end_dt = datetime.now(timezone.utc).date()
    years = lookback_years or cfg.get("lookback_years", 8)
    start_dt = end_dt - timedelta(days=int(years * 365.25))

    counts: dict[str, int] = {}
    for name, sym in cfg.get("macro_tickers", {}).items():
        try:
            df = _fetch_yfinance(sym, start_dt.isoformat(), end_dt.isoformat())
            if df.empty:
                counts[name] = 0
                continue
            df["ticker"] = name
            df["asset_class"] = "macro"
            df["source"] = "yfinance"
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            tiers.save_raw(df, source="prices", asset_class="macro",
                           ticker=name, dedup_cols=["ticker", "date"])
            counts[name] = len(df)
        except Exception as e:
            log.error(f"macro ingest failed {name}: {e}")
            counts[name] = -1
    return counts
