"""Top movers section: top N gainers / losers across windows.

Uses on-demand price fetching for the curated universe (top-50 India
+ US large caps + crypto majors). Computes returns at multiple
horizons: 1d / 5d / 21d (week / month). Cached at TTL.prices_4h.

Used by the dashboard's "Top movers" tab.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from omni.ondemand.cache import TTL, cached
from omni.ondemand.universe import _CURATED
from omni.ingest.prices import fetch_prices
from omni.utils.logger import log


def _curated_for_scope(scope: str) -> list[tuple[str, str, str]]:
    """Subset curated universe by scope = 'india' | 'us' | 'crypto' | 'all'."""
    rows = list(_CURATED)
    if scope == "all":
        return rows
    return [r for r in rows if r[1] == scope]


@cached(ttl=TTL.prices, key="movers_returns:{scope}")
def _returns_panel(scope: str = "india") -> pd.DataFrame:
    """For each (ticker, asset_class) in scope, fetch ~40 trading days
    and compute 1d / 5d / 21d returns + last close + volume.

    Returns DataFrame: [ticker, asset_class, name, last_close, vol,
                          ret_1d, ret_5d, ret_21d, last_date]
    """
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=70)).isoformat()
    rows = []
    for ticker, ac, name in _curated_for_scope(scope):
        try:
            df = fetch_prices(ac, ticker, start=start, end=end)
            if df is None or df.empty or len(df) < 6:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            last = df.iloc[-1]
            close = float(last["close"])
            vol = float(last.get("volume", 0) or 0)

            def _ret(n: int) -> float:
                if len(df) <= n:
                    return float("nan")
                prev = float(df.iloc[-(n + 1)]["close"])
                if prev <= 0:
                    return float("nan")
                return (close / prev) - 1.0

            rows.append({
                "ticker": ticker,
                "asset_class": ac,
                "name": name,
                "last_close": round(close, 2),
                "vol": int(vol),
                "ret_1d": round(_ret(1) * 100, 2),
                "ret_5d": round(_ret(5) * 100, 2),
                "ret_21d": round(_ret(21) * 100, 2),
                "last_date": str(pd.Timestamp(last["date"]).date()),
            })
        except Exception as e:  # noqa: BLE001
            log.debug(f"movers fetch failed {ticker}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def top_movers(scope: str = "india", *, horizon: str = "ret_1d",
                top_k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top_gainers, top_losers) DataFrames sorted by `horizon`."""
    df = _returns_panel(scope)
    if df is None or df.empty or horizon not in df.columns:
        empty = pd.DataFrame(columns=["ticker", "name", "last_close", horizon])
        return empty, empty
    sub = df.dropna(subset=[horizon]).copy()
    cols = ["ticker", "name", "asset_class", "last_close", horizon, "last_date"]
    gainers = sub.nlargest(top_k, horizon)[cols].reset_index(drop=True)
    losers = sub.nsmallest(top_k, horizon)[cols].reset_index(drop=True)
    return gainers, losers
