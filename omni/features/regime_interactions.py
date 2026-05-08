"""Regime-interaction features.

Same RSI=70 means VERY different things in a bull regime (continuation likely)
vs bear regime (reversion likely). XGBoost CAN learn this from the regime
column alone, but explicit interaction features make the signal much easier
to pick up — material lift in low-data regimes.

Public API:
    add_regime_interactions(df) -> DataFrame
"""
from __future__ import annotations

import pandas as pd


_INTERACT_COLS = (
    "rsi_14", "macd_hist", "vol_ratio", "ret_20d",
    "mom_vol_adj_20d", "sent_mean", "sent_z_30d",
)


def add_regime_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``{col}_x_{regime}`` interaction columns for top features.

    Uses the existing ``regime`` column (from regime_live or rubric input).
    The interaction is column-value × indicator(regime=='target_regime').
    Generates one column per (top_col × regime) pair where the regime is
    one of bull/bear/chop. Crisis kept rare so we skip explicit interactions
    for it (XGBoost can still learn the bool from the raw regime column).
    """
    if df is None or df.empty or "regime" not in df.columns:
        return df
    out = df.copy()
    target_regimes = ("bull", "bear", "chop")
    for col in _INTERACT_COLS:
        if col not in out.columns:
            continue
        for rg in target_regimes:
            out[f"{col}_x_{rg}"] = out[col] * (out["regime"] == rg).astype(int)
    return out
