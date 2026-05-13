"""Sentiment z-score and momentum features.

Differentiates 'always bullish' tickers from 'newly bullish' ones by
z-scoring today's sentiment against the ticker's own 30-day rolling mean
and standard deviation. Adds short-term sentiment acceleration too.

Phase 24.18 — adds three "narrative" features that consume the keyword
counts produced by `omni.features.sentiment.aggregate`:
    news_volume_z_60d     — abnormal news volume vs 60d mean
    panic_keyword_ratio   — fraction of recent articles using crash/circuit
                            language; high values flag retail-panic regimes
    promoter_news_flag    — 1 if any insider/promoter mention in last 7d
    sebi_action_flag      — 1 if any SEBI/investigation mention in last 7d

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

    # ---- Phase 24.18 narrative features ----------------------------------
    # All three require columns produced by sentiment.aggregate(). If an
    # older sentiment frame is passed in (without the keyword columns) we
    # silently no-op the new features rather than crashing — graceful
    # degradation is the same pattern used elsewhere in the build.

    # news_volume_z_60d: 7d sum of n_articles vs 60d trailing mean of that 7d sum.
    if "n_articles" in out.columns:
        n7 = (
            out.groupby("ticker", observed=True)["n_articles"]
               .transform(lambda s: s.fillna(0).rolling(7, min_periods=1).sum())
        )
        n7_mean = (
            n7.groupby(out["ticker"], observed=True)
              .transform(lambda s: s.shift(1).rolling(60, min_periods=10).mean())
        )
        n7_std = (
            n7.groupby(out["ticker"], observed=True)
              .transform(lambda s: s.shift(1).rolling(60, min_periods=10).std())
        )
        n7_std = n7_std.replace(0, np.nan)
        out["news_volume_z_60d"] = (n7 - n7_mean) / n7_std

    # panic_keyword_ratio: panic_kw_count_7d / max(n_articles_7d, 1).
    # Bounded in [0, 1]. High values = "lots of recent crash language"
    if {"panic_kw_count", "n_articles"}.issubset(out.columns):
        panic7 = (
            out.groupby("ticker", observed=True)["panic_kw_count"]
               .transform(lambda s: s.fillna(0).rolling(7, min_periods=1).sum())
        )
        n7 = (
            out.groupby("ticker", observed=True)["n_articles"]
               .transform(lambda s: s.fillna(0).rolling(7, min_periods=1).sum())
        )
        out["panic_keyword_ratio"] = panic7 / n7.where(n7 > 0)

    # promoter_news_flag: 1 if any insider/promoter mention in last 7d
    if "promoter_kw_count" in out.columns:
        promoter7 = (
            out.groupby("ticker", observed=True)["promoter_kw_count"]
               .transform(lambda s: s.fillna(0).rolling(7, min_periods=1).sum())
        )
        out["promoter_news_flag"] = (promoter7 > 0).astype(float)

    # sebi_action_flag: 1 if any SEBI/probe mention in last 7d
    if "sebi_kw_count" in out.columns:
        sebi7 = (
            out.groupby("ticker", observed=True)["sebi_kw_count"]
               .transform(lambda s: s.fillna(0).rolling(7, min_periods=1).sum())
        )
        out["sebi_action_flag"] = (sebi7 > 0).astype(float)

    return out
