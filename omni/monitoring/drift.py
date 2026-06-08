"""Population Stability Index (PSI) computation.

PSI is the standard data-drift / covariate-shift metric in production ML
pipelines (Siddiqi 2006, originally for credit-scoring). Given a baseline
distribution and a current distribution of the same feature, PSI is

    PSI = Σ (cur_pct - base_pct) * ln(cur_pct / base_pct)

Rules of thumb:
  PSI < 0.10                        : negligible shift
  0.10 <= PSI < 0.25                : moderate shift, watch
  PSI >= 0.25                       : material shift, action required
                                       (retrain, recalibrate, or investigate)

Discretisation: the baseline is binned by deciles, and the same bin
edges are applied to the current sample. Empty bins are floored at a
small epsilon (`min_pct = 1e-4`) so the log term stays finite.

API:
    psi = compute_psi(baseline, current)
    drift = psi_per_feature(baseline_df, current_df)  # DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MIN_PCT = 1e-4


def compute_psi(baseline: np.ndarray | pd.Series,
                current: np.ndarray | pd.Series,
                *, n_bins: int = 10) -> float:
    """PSI between two 1-D numeric samples. NaNs are dropped on both sides."""
    b = pd.to_numeric(pd.Series(baseline), errors="coerce").dropna().to_numpy()
    c = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy()
    if b.size == 0 or c.size == 0:
        return float("nan")
    # If the baseline has near-zero variance, PSI is undefined — return 0 so
    # we don't fire false alarms on quasi-constant features.
    if np.std(b) < 1e-12:
        return 0.0
    # Build equal-frequency bins from baseline
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(b, quantiles))
    if len(edges) < 3:
        # not enough variation in baseline for meaningful binning
        return 0.0
    # Extend extreme edges so out-of-range current values still bin
    edges[0] = -np.inf
    edges[-1] = np.inf
    base_counts, _ = np.histogram(b, bins=edges)
    cur_counts, _ = np.histogram(c, bins=edges)
    base_pct = np.maximum(base_counts / base_counts.sum(), _MIN_PCT)
    cur_pct = np.maximum(cur_counts / cur_counts.sum(), _MIN_PCT)
    psi = np.sum((cur_pct - base_pct) * np.log(cur_pct / base_pct))
    return float(psi)


def severity(psi: float) -> str:
    """Standard PSI severity bucket."""
    if not np.isfinite(psi):
        return "unknown"
    if psi < 0.10:
        return "ok"
    if psi < 0.25:
        return "watch"
    return "alarm"


@dataclass(frozen=True)
class FeaturePSI:
    feature: str
    psi: float
    severity: str
    n_baseline: int
    n_current: int


def psi_per_feature(baseline_df: pd.DataFrame,
                     current_df: pd.DataFrame,
                     *,
                     n_bins: int = 10,
                     numeric_only: bool = True,
                     features: list[str] | None = None,
                     ) -> pd.DataFrame:
    """Compute per-feature PSI between two dataframes.

    Returns a DataFrame sorted by descending PSI with columns:
        feature, psi, severity, n_baseline, n_current
    """
    if features is None:
        cols = [c for c in baseline_df.columns
                 if c in current_df.columns
                 and (not numeric_only
                       or pd.api.types.is_numeric_dtype(baseline_df[c]))]
    else:
        cols = [c for c in features
                 if c in baseline_df.columns and c in current_df.columns]

    rows: list[FeaturePSI] = []
    for c in cols:
        try:
            psi = compute_psi(baseline_df[c], current_df[c], n_bins=n_bins)
        except Exception:  # noqa: BLE001
            psi = float("nan")
        rows.append(FeaturePSI(
            feature=c, psi=psi, severity=severity(psi),
            n_baseline=int(baseline_df[c].dropna().shape[0]),
            n_current=int(current_df[c].dropna().shape[0]),
        ))
    if not rows:
        return pd.DataFrame(columns=["feature", "psi", "severity",
                                       "n_baseline", "n_current"])
    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)
    return df
