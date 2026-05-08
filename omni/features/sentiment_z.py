"""Sentiment z-score and momentum features.

Differentiates 'always bullish' tickers from 'newly bullish' ones by
z-scoring today's sentiment against the ticker's own 30-day rolling mean
and standard deviation. Adds short-term sentiment acceleration too.

Public API:
    add_sentiment_z(df) -> DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_sentiment_z(df: pd.DataFrame, *, window: int = 30) -> pd.DataFrame:
    """Add ``sent_z_30d`` and ``sent_accel_3d`` features.

    Requires ``sent_mean`` and ``ticker`` columns (already present from the
    sentiment merge). Computed per-ticker so each stock has its own baseline.
    """
    if df is None or df.empty or "sent_mean" not in df.columns:
        return df

    out = df.copy().sort_values(["ticker", "date"])
    grp = out.groupby("ticker", observed=True)["sent_mean"]
    rolling_mean = grp.transform(lambda s: s.shift(1).rolling(window, min_periods=5).mean())
    rolling_std = grp.transform(lambda s: s.shift(1).rolling(window, min_periods=5).std())
    rolling_std = rolling_std.replace(0, np.nan)
    out["sent_z_30d"] = (out["sent_mean"] - rolling_mean) / rolling_std

    out["sent_accel_3d"] = (
        out.groupby("ticker", observed=True)["sent_mean"]
           .transform(lambda s: s - s.shift(3))
    )
    return out
