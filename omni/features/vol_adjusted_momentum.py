"""Volatility-adjusted momentum features.

Raw 20-day return is noisy because high-vol stocks always look like big
movers. Dividing by realised volatility gives a risk-adjusted trend signal
that's far more predictive of future direction.

Public API:
    add_vol_adjusted_momentum(df) -> DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_vol_adjusted_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``mom_vol_adj_20d`` and ``mom_vol_adj_60d`` features.

    Definition: ``ret_Hd / realized_vol_Hd`` — sometimes called the t-stat
    of recent return. Higher means a steadier uptrend, not just a lucky one.
    Robust to NaN: inputs missing -> output NaN.
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    # Resolve the vol column — `omni.features.technical.compute` produces
    # ``vol_20`` (annualised); ``realized_vol_20d`` is also accepted.
    vol20_col = ("realized_vol_20d" if "realized_vol_20d" in out.columns
                  else "vol_20" if "vol_20" in out.columns else None)

    if "ret_20d" in out.columns and vol20_col:
        denom = out[vol20_col].replace(0, np.nan)
        out["mom_vol_adj_20d"] = out["ret_20d"] / denom

    if "ret_60d" in out.columns:
        if "realized_vol_60d" in out.columns:
            denom = out["realized_vol_60d"].replace(0, np.nan)
        elif vol20_col:
            denom = (out[vol20_col] * np.sqrt(3.0)).replace(0, np.nan)
        else:
            denom = None
        if denom is not None:
            out["mom_vol_adj_60d"] = out["ret_60d"] / denom

    return out
