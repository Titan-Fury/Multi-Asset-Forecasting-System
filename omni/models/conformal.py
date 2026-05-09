"""Split conformal prediction for calibrated probability intervals.

Given a held-out calibration set (true labels + model probabilities), we
compute non-conformity scores and use the (1-α) empirical quantile to
produce prediction sets / intervals with coverage guarantees:

  P( y_test ∈ C(x_test) ) >= 1 - α    (asymptotically, exchangeable data)

For binary classification we wrap a calibrated probability into a binary
prediction set; for return-magnitude intervals we use absolute residuals
to widen the ±σ envelope until empirical coverage hits the nominal level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalIntervalCalibrator:
    quantile: float       # the (1-α) quantile of |residuals| from calibration
    alpha: float          # nominal miscoverage rate (e.g. 0.20 for 80% coverage)

    def widen(self, mu: float, sigma: float) -> tuple[float, float]:
        """Return (low, high) interval that achieves the calibrated coverage."""
        half = self.quantile * sigma
        return mu - half, mu + half


def fit_interval_calibrator(
    cal_residuals: np.ndarray,
    cal_sigma: np.ndarray,
    *,
    alpha: float = 0.20,
) -> ConformalIntervalCalibrator:
    """`cal_residuals` = (y_true - mu_pred); `cal_sigma` = predicted scale.

    Returns calibrator whose `widen()` produces intervals with empirical
    coverage >= 1-alpha on data drawn from the same distribution.
    """
    cal_residuals = np.asarray(cal_residuals, dtype=float)
    cal_sigma = np.asarray(cal_sigma, dtype=float)
    if len(cal_residuals) == 0 or len(cal_sigma) == 0 or len(cal_residuals) != len(cal_sigma):
        return ConformalIntervalCalibrator(quantile=1.0, alpha=alpha)
    s = np.where(cal_sigma > 0, cal_sigma, 1e-9)
    scores = np.abs(cal_residuals) / s
    n = len(scores)
    # Finite-sample correction: ceil((n+1)*(1-alpha))/n quantile
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(scores, q_level, method="higher"))
    return ConformalIntervalCalibrator(quantile=q, alpha=alpha)


def empirical_coverage(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    cal: ConformalIntervalCalibrator,
) -> float:
    """Fraction of y_true that falls inside the conformal interval."""
    half = cal.quantile * np.where(sigma > 0, sigma, 1e-9)
    inside = (y_true >= (mu - half)) & (y_true <= (mu + half))
    return float(inside.mean()) if len(y_true) else 0.0
