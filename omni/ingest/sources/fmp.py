"""Financial Modeling Prep fundamentals client.

Free tier: 250 calls/day. Better fundamentals coverage than yfinance
for US equities (revenue growth, ROE, margins are more reliable).

Returns the same 15-field schema as omni.ingest.fundamentals.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests

from omni.utils.logger import log

_FIELDS = [
    "market_cap", "trailing_pe", "forward_pe", "peg_ratio", "price_to_book",
    "ev_to_ebitda", "profit_margin", "operating_margin", "return_on_equity",
    "revenue_growth", "earnings_growth", "debt_to_equity", "current_ratio",
    "dividend_yield", "beta",
]
_BASE = "https://financialmodelingprep.com/api/v3"


def _key() -> str | None:
    k = os.getenv("FMP_API_KEY", "").strip()
    return k or None


def fundamentals(ticker: str, asset_class: str = "us") -> pd.DataFrame:
    """Fetch fundamentals for a US equity ticker.

    Returns a 1-row DataFrame with the standard schema, or empty if no
    key / no data / non-US asset class.
    """
    if asset_class != "us":
        return pd.DataFrame()
    key = _key()
    if not key:
        return pd.DataFrame()

    row: dict[str, object] = {
        "run_date": datetime.now(timezone.utc).date().isoformat(),
        "ticker": ticker, "asset_class": asset_class, "source": "fmp",
    }
    try:
        # 1) profile -> market cap, beta
        r1 = requests.get(f"{_BASE}/profile/{ticker}",
                          params={"apikey": key}, timeout=30)
        prof = (r1.json() or [{}])[0] if r1.ok else {}
        # 2) ratios-ttm -> all the ratio fields
        r2 = requests.get(f"{_BASE}/ratios-ttm/{ticker}",
                          params={"apikey": key}, timeout=30)
        ratios = (r2.json() or [{}])[0] if r2.ok else {}
        # 3) key-metrics-ttm -> ev/ebitda, peg
        r3 = requests.get(f"{_BASE}/key-metrics-ttm/{ticker}",
                          params={"apikey": key}, timeout=30)
        km = (r3.json() or [{}])[0] if r3.ok else {}
        # 4) income-statement-growth -> revenue/earnings growth
        r4 = requests.get(f"{_BASE}/income-statement-growth/{ticker}",
                          params={"limit": 1, "apikey": key}, timeout=30)
        growth = (r4.json() or [{}])[0] if r4.ok else {}
    except Exception as e:
        log.warning(f"fmp fetch failed for {ticker}: {e}")
        return pd.DataFrame()

    def _f(d: dict, k: str) -> float | None:
        v = d.get(k)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    row["market_cap"] = _f(prof, "mktCap")
    row["beta"] = _f(prof, "beta")
    row["trailing_pe"] = _f(ratios, "peRatioTTM")
    row["forward_pe"] = _f(km, "peRatioTTM")  # FMP no separate forward; reuse
    row["peg_ratio"] = _f(km, "pegRatioTTM")
    row["price_to_book"] = _f(ratios, "priceToBookRatioTTM")
    row["ev_to_ebitda"] = _f(km, "enterpriseValueOverEBITDATTM")
    row["profit_margin"] = _f(ratios, "netProfitMarginTTM")
    row["operating_margin"] = _f(ratios, "operatingProfitMarginTTM")
    row["return_on_equity"] = _f(ratios, "returnOnEquityTTM")
    row["revenue_growth"] = _f(growth, "growthRevenue")
    row["earnings_growth"] = _f(growth, "growthNetIncome")
    row["debt_to_equity"] = _f(ratios, "debtEquityRatioTTM")
    row["current_ratio"] = _f(ratios, "currentRatioTTM")
    row["dividend_yield"] = _f(ratios, "dividendYieldTTM")

    if all(row.get(f) is None for f in _FIELDS):
        return pd.DataFrame()
    return pd.DataFrame([row])
