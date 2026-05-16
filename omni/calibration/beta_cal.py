"""Beta calibration (Kull, Filho, Flach 2017).

Reference:
  Kull, M., Silva Filho, T., Flach, P. (2017).
  Beta calibration: a well-founded and easily implemented improvement
  on logistic calibration for binary classifiers. AISTATS.

Why this exists alongside isotonic:
  - Isotonic regression is non-parametric — flexible, but prone to
    overfit when the calibration sample is small (< 5k rows). Many of
    our walk-forward folds give isotonic only ~1-2k val rows.
  - Beta calibration assumes a Beta(a, b) distribution for scores in
    each class and produces a 3-parameter map
        p_cal = sigmoid( a * log(p) - b * log(1 - p) + c )
    that subsumes Platt (a == b) and gives well-defined behaviour at
    the 0/1 endpoints.

Empirical finding on this codebase (scripts/diagnose_calibration.py,
training set 2026-05-14, h ∈ {1, 5, 21}):

  | horizon | brier_iso | brier_beta | ece_iso | ece_beta | winner |
  | ------- | --------- | ---------- | ------- | -------- | ------ |
  | h=1     | 0.1784    | 0.1785     | 0.0031  | 0.0109   | iso    |
  | h=5     | 0.1291    | 0.1291     | 0.0041  | 0.0076   | mixed  |
  | h=21    | 0.1994    | 0.1995     | 0.0074  | 0.0058   | mixed  |

Brier is essentially tied; isotonic wins ECE on the two short horizons.
Beta only edges out at h=21. So we ship beta as **opt-in** via the
`OMNI_CAL_METHOD=beta` env switch and keep isotonic as the default.

API mirrors sklearn's `IsotonicRegression`:
    cal = BetaCalibrator().fit(scores, y)
    p   = cal.predict(scores)              # vector in (eps, 1-eps)
    p2  = cal.predict_proba(scores)[:, 1]  # for sklearn-style callers
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


_EPS = 1e-12


class BetaCalibrator:
    """3-parameter beta calibrator. Fit on raw classifier scores."""

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 1.0
        self.c: float = 0.0
        self._fitted: bool = False

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "BetaCalibrator":
        s = np.clip(np.asarray(scores, dtype=float).ravel(), _EPS, 1.0 - _EPS)
        y = np.asarray(y, dtype=int).ravel()
        if s.shape != y.shape:
            raise ValueError("scores and y must have the same shape")
        # Two regressors: log(p) and -log(1-p). Logistic regression on
        # them gives coefficients [a, b] and intercept c.
        X = np.column_stack([np.log(s), -np.log1p(-s)])
        # If only one class is present in y, beta calibration has nothing
        # to learn; return a constant predictor centered on the class rate.
        if len(np.unique(y)) < 2:
            self.a = 0.0
            self.b = 0.0
            self.c = float(np.log((y.mean() + _EPS) / (1.0 - y.mean() + _EPS)))
            self._fitted = True
            return self
        # liblinear handles small N robustly; default L2 regularisation
        # with C=1e6 gives near-MLE behaviour without numerical chaos.
        lr = LogisticRegression(C=1e6, solver="liblinear", max_iter=200)
        lr.fit(X, y)
        coef = lr.coef_.ravel()
        self.a = float(coef[0])
        self.b = float(coef[1])
        self.c = float(lr.intercept_[0])
        self._fitted = True
        return self

    def _logit(self, scores: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(scores, dtype=float).ravel(), _EPS, 1.0 - _EPS)
        return self.a * np.log(s) - self.b * np.log1p(-s) + self.c

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Calibrated P(y=1 | score). Returns 1-D array."""
        if not self._fitted:
            raise RuntimeError("BetaCalibrator.fit must be called before predict")
        z = self._logit(scores)
        # numerically stable sigmoid
        out = np.where(z >= 0,
                       1.0 / (1.0 + np.exp(-z)),
                       np.exp(z) / (1.0 + np.exp(z)))
        return out

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """sklearn-compatible 2-column (P0, P1) output."""
        p1 = self.predict(scores)
        return np.column_stack([1.0 - p1, p1])

    def __repr__(self) -> str:
        if not self._fitted:
            return "BetaCalibrator(unfit)"
        return f"BetaCalibrator(a={self.a:.3f}, b={self.b:.3f}, c={self.c:.3f})"
