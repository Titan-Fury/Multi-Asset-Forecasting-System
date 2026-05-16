"""Calibrator factory — env-controlled selection between isotonic / beta.

Set `OMNI_CAL_METHOD=beta` to opt into beta calibration. Default
remains isotonic regression so the existing trained models keep
behaving identically.

Returns a CalibratorWrapper that exposes a uniform `predict_proba`
2-column interface, regardless of the underlying method.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .beta_cal import BetaCalibrator


class CalibratorWrapper:
    """Uniform interface so callers don't branch on method.

    `inner` is either:
      - a fitted IsotonicRegression       (-> .predict gives P(y=1))
      - a fitted BetaCalibrator           (-> .predict gives P(y=1))
      - a fitted CalibratedClassifierCV   (-> .predict_proba gives [[p0,p1]])
    """

    __slots__ = ("inner", "method", "_kind")

    def __init__(self, inner: Any, method: str) -> None:
        self.inner = inner
        self.method = method
        # Detect call style once
        if hasattr(inner, "predict_proba") and not isinstance(inner, BetaCalibrator):
            self._kind = "predict_proba"
        else:
            self._kind = "predict"

    def predict_proba(self, X) -> np.ndarray:
        if self._kind == "predict_proba":
            return np.asarray(self.inner.predict_proba(X), dtype=float)
        # 1-D scores -> 2-column proba
        p1 = np.asarray(self.inner.predict(X), dtype=float).ravel()
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]


def get_method() -> str:
    """Return the calibration method from env. Defaults to 'isotonic'."""
    m = os.environ.get("OMNI_CAL_METHOD", "isotonic").strip().lower()
    if m not in ("isotonic", "beta"):
        return "isotonic"
    return m


def fit_calibrator(scores: np.ndarray, y: np.ndarray, *,
                    method: str | None = None) -> CalibratorWrapper:
    """Fit a 1-D score calibrator. Use when you have raw classifier
    probabilities (e.g. uncalibrated XGB output on a held-out fold)
    and want them mapped to well-calibrated probabilities.

    Use this in walk-forward training paths where you previously
    instantiated `IsotonicRegression()` directly.
    """
    if method is None:
        method = get_method()
    if method == "beta":
        cal = BetaCalibrator().fit(scores, y)
        return CalibratorWrapper(cal, "beta")
    # default: isotonic
    from sklearn.isotonic import IsotonicRegression
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(scores, y)
    return CalibratorWrapper(cal, "isotonic")
