"""Cross-sectional rank features.

Stocks move RELATIVE to peers, so absolute values like PE=20 mean nothing
without context. These features convert raw values into per-date percentile
ranks within each asset class — robust to outliers and regime shifts.

Public API:
    add_cross_sectional_ranks(df) -> DataFrame
        Adds *_pct columns: percentile (0-1) of the value within (date, asset_class).
"""
from __future__ import annotations

import pandas as pd


_RANK_COLS = (
    "ret_20d", "ret_60d",                 # momentum
    "rsi_14", "macd_hist", "vol_ratio",   # technical
    "trailing_pe", "profit_margin",       # fundamental
    "return_on_equity", "revenue_growth",
)


def add_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-(date, asset_class) percentile rank columns.

    Output columns are named ``{col}_pct`` and lie in [0, 1]. NaN inputs
    produce NaN ranks. The function is leak-free because ranks are computed
    only across rows with the same date.
    """
    if df is None or df.empty:
        return df
    if "date" not in df.columns or "asset_class" not in df.columns:
        return df

    out = df.copy()
    for col in _RANK_COLS:
        if col not in out.columns:
            continue
        out[f"{col}_pct"] = (
            out.groupby(["date", "asset_class"])[col]
                .rank(pct=True, method="average")
        )
    return out
