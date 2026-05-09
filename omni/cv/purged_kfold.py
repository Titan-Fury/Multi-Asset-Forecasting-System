"""Purged K-Fold cross-validation with embargo (de Prado, AFML §7).

Standard K-fold leaks because financial labels span multiple bars (a 5-day
forward return at t spans [t, t+5]). Two fixes:

  PURGE: drop training samples whose label window overlaps the validation
         set's time range.
  EMBARGO: after the validation set, drop the next `embargo_days` from
           training to prevent the model from memorising the just-seen
           validation signal.

Usage:
    folds = purged_kfold_indices(dates, n_folds=5, horizon_days=5, embargo_days=5)
    for tr, va in folds:
        ...
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class _Fold:
    train_idx: np.ndarray
    val_idx: np.ndarray


def purged_kfold_indices(
    dates: pd.Series | pd.DatetimeIndex,
    *,
    n_folds: int = 5,
    horizon_days: int = 5,
    embargo_days: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, val_idx) numpy arrays for each fold.

    `dates` is the timestamp of each row's prediction (NOT the label realisation).
    Validation folds are contiguous time windows; training is everything outside
    [val_start - horizon, val_end + embargo].
    """
    if not isinstance(dates, pd.Series):
        dates = pd.Series(pd.to_datetime(dates))
    else:
        dates = pd.to_datetime(dates).reset_index(drop=True)

    n = len(dates)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    # Sort by time for clean contiguous splits, but keep original positional indices
    order = np.argsort(dates.values)
    sorted_dates = dates.values[order]
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    edges = np.linspace(0, n, n_folds + 1).astype(int)
    horizon_td = pd.Timedelta(days=horizon_days)
    embargo_td = pd.Timedelta(days=embargo_days)

    for k in range(n_folds):
        v_lo, v_hi = edges[k], edges[k + 1]
        if v_lo == v_hi:
            continue
        val_pos = order[v_lo:v_hi]
        val_start = sorted_dates[v_lo]
        val_end = sorted_dates[v_hi - 1]

        purge_lo = pd.Timestamp(val_start) - horizon_td
        purge_hi = pd.Timestamp(val_end) + embargo_td

        train_mask = np.ones(n, dtype=bool)
        train_mask[val_pos] = False
        # Purge any training row whose label window overlaps the val window
        ts = dates.values
        overlap = (ts >= np.datetime64(purge_lo)) & (ts <= np.datetime64(purge_hi))
        train_mask[overlap] = False
        train_pos = np.where(train_mask)[0]

        if len(train_pos) == 0 or len(val_pos) == 0:
            continue
        folds.append((train_pos, val_pos))

    return folds


def assert_no_leakage(
    folds: list[tuple[np.ndarray, np.ndarray]],
    dates: pd.Series,
    *,
    horizon_days: int = 5,
    embargo_days: int = 5,
) -> None:
    """Cheap sanity check: no train sample's [t, t+horizon] window touches val,
    and no train sample within `embargo_days` after the val window."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    h = pd.Timedelta(days=horizon_days)
    e = pd.Timedelta(days=embargo_days)
    for k, (tr, va) in enumerate(folds):
        v_start = dates.iloc[va].min()
        v_end = dates.iloc[va].max()
        tr_dates = dates.iloc[tr]
        # train label-end must not enter val window
        leak1 = ((tr_dates + h) >= v_start) & (tr_dates <= v_end)
        leak2 = (tr_dates > v_end) & (tr_dates <= v_end + e)
        leaks = int(leak1.sum() + leak2.sum())
        if leaks:
            raise AssertionError(f"fold {k}: {leaks} leaked training samples")
