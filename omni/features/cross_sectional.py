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
    "ret_20d",                                # momentum
    "ret_60d",                                # momentum (60d, Phase 28.4)
    "rsi_14", "macd_hist_pct", "vol_ratio",   # technical (Phase 24.17: macd_hist_pct)
    "trailing_pe", "profit_margin",           # fundamental
    "return_on_equity", "revenue_growth",
    # Phase 24.18 — narrative panic / news-volume features. Cross-sectional
    # ranks of these are MORE useful than absolute values: a 0.95 rank of
    # panic_keyword_ratio means "this stock has the noisiest news today
    # within its asset class" which is a much cleaner regime signal than
    # the raw ratio.
    "news_volume_z_60d", "panic_keyword_ratio",
    # Phase 28.4 — eight obvious cross-sectional candidates that were
    # previously absolute-value-only despite being naturally rank-friendly:
    #   vol_20             — 20-day annualised vol (which stock is hot today)
    #   adx_14             — trend strength (which stock is trending hardest)
    #   atr_pct_14         — daily-range / close (intraday wildness)
    #   dist_from_52w_high — how far from yearly peak (recovery vs distressed)
    #   dist_from_52w_low  — how far from yearly trough (oversold)
    #   obv_z_60d          — accumulation/distribution flow z-score
    #   piotroski_f_score  — fundamental quality composite (0-9)
    #   qmj_score          — Quality-Minus-Junk composite
    "vol_20", "adx_14", "atr_pct_14",
    "dist_from_52w_high", "dist_from_52w_low",
    "obv_z_60d",
    "piotroski_f_score", "qmj_score",
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
