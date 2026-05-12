"""Fundamentals ingestion via yfinance Ticker.info.

Captures point-in-time snapshot per run_date. Schema:
  run_date, ticker, asset_class,
  market_cap, trailing_pe, forward_pe, peg_ratio, price_to_book,
  ev_to_ebitda, profit_margin, operating_margin, return_on_equity,
  revenue_growth, earnings_growth, debt_to_equity, current_ratio,
  dividend_yield, beta, source

Note: yfinance.info returns inconsistent keys across regions; we
defensively .get() everything and store NaN for missing fields.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from omni.ingest.prices import _yf_symbol
from omni.storage import tiers
from omni.utils.env import universe
from omni.utils.logger import log

_FIELDS = {
    "market_cap": "marketCap",
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "peg_ratio": "pegRatio",
    "price_to_book": "priceToBook",
    "ev_to_ebitda": "enterpriseToEbitda",
    "profit_margin": "profitMargins",
    "operating_margin": "operatingMargins",
    "return_on_equity": "returnOnEquity",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "debt_to_equity": "debtToEquity",
    "current_ratio": "currentRatio",
    "dividend_yield": "dividendYield",
    "beta": "beta",
}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5), reraise=True)
def _fetch_one(asset_class: str, ticker: str) -> dict:
    import yfinance as yf
    sym = _yf_symbol(asset_class, ticker)
    info = yf.Ticker(sym).info or {}
    row: dict[str, object] = {
        "run_date": datetime.now(timezone.utc).date().isoformat(),
        "ticker": ticker,
        "asset_class": asset_class,
        "source": "yfinance",
    }
    for our_key, yf_key in _FIELDS.items():
        v = info.get(yf_key)
        row[our_key] = float(v) if isinstance(v, (int, float)) else None

    # Phase 24.16 — Screener.in fallback for India tickers when yfinance
    # left the essential fields empty. Common for NSE SME / mid-cap names
    # like CIANAGRO, OMAXAUTO. We only call Screener if the most-used
    # fields are missing, to avoid an extra HTTP roundtrip when yfinance
    # already returned a full row.
    if asset_class == "india":
        essentials = ("trailing_pe", "return_on_equity",
                       "profit_margin", "debt_to_equity")
        n_missing = sum(1 for k in essentials if row.get(k) is None)
        if n_missing >= 2:
            try:
                from omni.ingest.sources import screener
                aug = screener.fetch(ticker)
                if aug:
                    # pe_ratio -> trailing_pe (alias)
                    if "pe_ratio" in aug and row.get("trailing_pe") is None:
                        row["trailing_pe"] = float(aug["pe_ratio"])
                    if "return_on_equity" in aug and row.get("return_on_equity") is None:
                        row["return_on_equity"] = float(aug["return_on_equity"])
                    if "debt_to_equity" in aug and row.get("debt_to_equity") is None:
                        row["debt_to_equity"] = float(aug["debt_to_equity"])
                    if "dividend_yield" in aug and row.get("dividend_yield") is None:
                        row["dividend_yield"] = float(aug["dividend_yield"])
                    if "market_cap" in aug and row.get("market_cap") is None:
                        row["market_cap"] = float(aug["market_cap"])
                    # Mark provenance so we know which row came from where
                    row["source"] = "yfinance+screener"
                    log.debug(f"screener fallback augmented {ticker}: "
                              f"{sorted(aug.keys())}")
            except Exception as e:  # noqa: BLE001
                log.debug(f"screener fallback failed for {ticker}: {e}")
    return row


def fetch_fundamentals(asset_class: str, ticker: str) -> pd.DataFrame:
    try:
        return pd.DataFrame([_fetch_one(asset_class, ticker)])
    except Exception as e:
        log.warning(f"fundamentals fetch failed for {asset_class}/{ticker}: {e}")
        return pd.DataFrame()


def ingest_universe() -> dict[str, int]:
    """Snapshot fundamentals across all equity tickers (skip crypto/fx)."""
    cfg = universe()
    counts: dict[str, int] = {}
    for asset_class, spec in cfg["asset_classes"].items():
        if asset_class in ("crypto", "fx"):
            continue
        for ticker in spec["tickers"]:
            df = fetch_fundamentals(asset_class, ticker)
            if df.empty:
                counts[f"{asset_class}/{ticker}"] = 0
                continue
            tiers.save_raw(df, source="fundamentals", asset_class=asset_class,
                           ticker=ticker, dedup_cols=["ticker", "run_date"])
            counts[f"{asset_class}/{ticker}"] = len(df)
            log.info(f"fundamentals {asset_class}/{ticker}: {len(df)} rows")
    return counts
