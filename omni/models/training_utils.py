"""Sample-weighting helpers for time-series ML training.

- recency_weights:    exponential decay so recent rows count more
- magnitude_weights:  emphasize big absolute moves so noise is suppressed
- top_quartile_target: relabel direction as 'top quartile of recent returns'

All functions are pure (return numpy arrays, take pandas inputs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def recency_weights(dates: pd.Series, *, half_life_years: float = 5.0,
                    ref_date: pd.Timestamp | None = None) -> np.ndarray:
    """Exponential half-life weights — recent rows weighted more.

    weight = 0.5 ** (years_ago / half_life_years)
    Normalised to mean=1 so the loss scale is preserved.
    """
    if dates is None or len(dates) == 0:
        return np.array([], dtype=float)
    d = pd.to_datetime(dates)
    if ref_date is None:
        ref_date = d.max()
    years_ago = (ref_date - d).dt.total_seconds() / (365.25 * 86400)
    w = 0.5 ** (years_ago.values / float(half_life_years))
    if w.mean() > 0:
        w = w / w.mean()
    return w.astype(float)


def magnitude_weights(fwd_returns: pd.Series, *, floor: float = 0.05,
                       cap_z: float = 3.0) -> np.ndarray:
    """Weight by |fwd_ret| so big moves matter more, clipped to ±cap_z sigma.

    A small ``floor`` keeps even tiny moves from getting zero weight.
    """
    r = pd.Series(fwd_returns).fillna(0.0).abs().values.astype(float)
    s = r.std() if r.std() > 0 else 1.0
    r = np.clip(r, 0.0, cap_z * s)
    w = r + floor * s
    w = w / w.mean() if w.mean() > 0 else w
    return w


def top_quartile_target(fwd_returns: pd.Series, *,
                         lookback_window: int = 252) -> np.ndarray:
    """Reframe direction target as 'in top quartile of recent rolling returns'.

    Less noisy than raw +1/-1 sign; better matches the alpha most strategies
    actually capture (the right tail of the return distribution).
    """
    r = pd.Series(fwd_returns).copy()
    rolling_q75 = r.shift(1).rolling(lookback_window, min_periods=30).quantile(0.75)
    out = (r > rolling_q75).astype(int).fillna(0).values
    return out
