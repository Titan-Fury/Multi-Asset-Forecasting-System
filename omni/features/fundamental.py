"""Fundamental features.

Inputs: the raw fundamentals snapshot from `omni.ingest.fundamentals`
(one row per (ticker, run_date)). Outputs cleaned + transformed features
plus z-scored versions within asset_class for cross-sectional comparison.

For asset_class in ('crypto', 'fx') -- fundamentals are not applicable;
the build pipeline simply skips them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Raw columns from ingest.fundamentals
_NUMERIC = [
    "market_cap", "trailing_pe", "forward_pe", "peg_ratio", "price_to_book",
    "ev_to_ebitda", "profit_margin", "operating_margin", "return_on_equity",
    "revenue_growth", "earnings_growth", "debt_to_equity", "current_ratio",
    "dividend_yield", "beta",
]


def _winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def compute(snapshots: pd.DataFrame) -> pd.DataFrame:
    """`snapshots` is the long-form table of fundamentals across the universe
    for one or more run_dates. Returns the same long form augmented with:
      log_market_cap
      earnings_yield (1/trailing_pe)
      <col>_z   -- z-score within (asset_class, run_date)
    """
    if snapshots.empty:
        return snapshots
    df = snapshots.copy()

    for c in _NUMERIC:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["log_market_cap"] = np.log(df["market_cap"].where(df["market_cap"] > 0))
    df["earnings_yield"] = 1.0 / df["trailing_pe"].replace(0, np.nan)

    # Winsorize within (asset_class, run_date) before z-scoring
    out_chunks = []
    for (ac, rd), grp in df.groupby(["asset_class", "run_date"], dropna=False):
        g = grp.copy()
        for c in _NUMERIC + ["log_market_cap", "earnings_yield"]:
            if c not in g.columns:
                continue
            w = _winsorize(g[c])
            mu = w.mean()
            sd = w.std(ddof=0)
            g[f"{c}_z"] = (w - mu) / sd if sd and not np.isnan(sd) else np.nan
        out_chunks.append(g)
    return pd.concat(out_chunks, ignore_index=True) if out_chunks else df


def latest_per_ticker(snapshots: pd.DataFrame, *, asof: str | None = None) -> pd.DataFrame:
    """Pick the most recent snapshot per ticker on/before `asof` (ISO date)."""
    if snapshots.empty:
        return snapshots
    df = snapshots.copy()
    df["run_date"] = pd.to_datetime(df["run_date"]).dt.date
    if asof:
        cutoff = pd.to_datetime(asof).date()
        df = df[df["run_date"] <= cutoff]
    return (df.sort_values("run_date")
              .groupby("ticker", as_index=False)
              .tail(1)
              .reset_index(drop=True))
