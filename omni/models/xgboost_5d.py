"""Walk-forward XGBoost classifier for the 5d direction target.

Design:
  - Target column: `fwd_dir_5d` (mapped from {-1, 0, +1} to {0, 1} where +1 -> 1
    and {-1, 0} -> 0  — i.e. "is the 5d forward return positive").
  - Features: ALL columns except those flagged by `omni.features.build.is_target_col`,
    plus we drop identity / date columns and any all-NaN columns.
  - Walk-forward CV: time-ordered folds of (train_end, test_window). NEVER a
    random split.
  - Calibration: each fold trains XGBoost on the inner-train slice, then fits
    `CalibratedClassifierCV(method='isotonic', cv='prefit')` on a held-out
    inner-validation slice (last 20% of train window). The prefit calibrator
    is what produces the per-row probability used downstream.
  - Output: a dict[fold_id -> {model, calibrator, test_predictions_df}].

Public API:
  prepare_xy(df) -> (X, y, feature_cols)
  walkforward_fit(df, n_folds=..., test_size=...) -> WalkForwardResult
  predict_proba(model, calibrator, X) -> np.ndarray
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from omni.features.build import is_target_col
from omni.utils.logger import log

# Columns that are identity/metadata, never features
_DROP_ALWAYS = {
    "date", "ticker", "asset_class", "run_date", "source",
    # Phase 24.22 — raw OHLCV columns must NOT become features. They are
    # absolute price levels in each stock's own currency unit, so a tree
    # learns to split on price-level proxies which leak the ticker's
    # identity and destroy any chance of cross-ticker generalization.
    # The technical/diff_lagged feature builders already produce the
    # ticker-invariant transforms (returns, ratios, z-scores) we want the
    # model to consume.
    "open", "high", "low", "close", "adj_close", "volume",
}


def prepare_xy(df: pd.DataFrame, *, target: str = "fwd_dir_5d"
               ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Strip target columns and metadata, return (X, y, feature_cols).

    y is binary: 1 if fwd_dir_5d > 0 else 0. Rows where the target is NaN are
    dropped.
    """
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in frame")
    work = df.dropna(subset=[target]).copy()
    y = (work[target] > 0).astype(int)

    feat_cols = [c for c in work.columns
                 if c not in _DROP_ALWAYS
                 and not is_target_col(c)]
    X = work[feat_cols].apply(pd.to_numeric, errors="coerce")
    # Drop columns that are entirely NaN within this slice
    X = X.dropna(axis=1, how="all")
    # Replace ±inf with NaN so XGBoost's missing=NaN handling kicks in
    # (new vol-adjusted / z-score features can produce inf on near-zero denoms).
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y, list(X.columns)


def walkforward_splits(dates: pd.Series, *, n_folds: int = 5,
                       min_train: int = 252) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) tuples in chronological order.

    Each fold's test window is the next slab; train is everything before it.
    """
    order = np.argsort(dates.values)
    n = len(order)
    if n <= min_train + n_folds:
        raise ValueError(f"not enough rows ({n}) for {n_folds} folds")
    fold_size = (n - min_train) // n_folds
    for k in range(n_folds):
        train_end = min_train + k * fold_size
        test_start = train_end
        test_end = min(n, test_start + fold_size)
        if test_end <= test_start:
            break
        train_idx = order[:train_end]
        test_idx = order[test_start:test_end]
        yield train_idx, test_idx


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    test_dates: pd.Series
    test_tickers: pd.Series  # NEW: ticker per test row, parallel to test_dates
    proba: np.ndarray
    y_true: np.ndarray
    model: object = None
    calibrator: object = None


@dataclass
class WalkForwardResult:
    folds: list[FoldResult] = field(default_factory=list)
    feature_cols: list[str] = field(default_factory=list)

    def predictions_frame(self) -> pd.DataFrame:
        rows = []
        for fr in self.folds:
            df = pd.DataFrame({
                "fold": fr.fold,
                "date": fr.test_dates.values,
                "ticker": fr.test_tickers.values,
                "proba_up": fr.proba,
                "y_true": fr.y_true,
            })
            rows.append(df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def walkforward_fit(df: pd.DataFrame, *, target: str = "fwd_dir_5d",
                    n_folds: int = 5, min_train: int = 504,
                    inner_val_frac: float = 0.2,
                    xgb_params: dict | None = None,
                    keep_models: bool = False) -> WalkForwardResult:
    """Walk-forward fit. Requires `df` to have a `date` column."""
    try:
        from xgboost import XGBClassifier
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError as e:
        raise RuntimeError("xgboost / scikit-learn missing from env") from e

    # sklearn 1.6+ removed cv='prefit'; use FrozenEstimator if available.
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn>=1.6
        _HAS_FROZEN = True
    except ImportError:
        _HAS_FROZEN = False

    if "date" not in df.columns:
        raise KeyError("walkforward_fit requires a 'date' column")

    X, y, feat_cols = prepare_xy(df, target=target)
    dates = pd.to_datetime(df.loc[X.index, "date"])
    tickers = (df.loc[X.index, "ticker"]
               if "ticker" in df.columns
               else pd.Series(["__single__"] * len(X), index=X.index))

    params = dict(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42,
    )
    # Phase 22 — opt-in GPU: set OMNI_XGB_DEVICE=cuda to use the local GPU.
    # Defaults to CPU so existing CI / non-GPU envs keep working.
    import os as _os
    _dev = _os.environ.get("OMNI_XGB_DEVICE", "").strip().lower()
    if _dev in ("cuda", "gpu"):
        params["device"] = "cuda"
    if xgb_params:
        params.update(xgb_params)

    res = WalkForwardResult(feature_cols=feat_cols)
    for k, (tr_idx, te_idx) in enumerate(walkforward_splits(dates, n_folds=n_folds, min_train=min_train)):
        # Inner split: last 20% of train window for calibration (time-ordered)
        n_inner = max(1, int(len(tr_idx) * inner_val_frac))
        inner_train = tr_idx[:-n_inner]
        inner_val = tr_idx[-n_inner:]

        Xt, yt = X.iloc[inner_train], y.iloc[inner_train]
        Xv, yv = X.iloc[inner_val], y.iloc[inner_val]
        Xs, ys = X.iloc[te_idx], y.iloc[te_idx]

        clf = XGBClassifier(**params)
        clf.fit(Xt.values, yt.values, verbose=False)

        # Phase 25 — env-controlled calibration: OMNI_CAL_METHOD=beta opts
        # into Kull-Filho-Flach beta calibration. Default (isotonic) keeps
        # the existing CalibratedClassifierCV path so persisted models
        # behave identically.
        from omni.calibration.factory import get_method
        if get_method() == "beta":
            from omni.calibration.factory import fit_calibrator
            raw_v = clf.predict_proba(Xv.values)[:, 1]
            cal = fit_calibrator(raw_v, yv.values, method="beta")
            # Wrap so .predict_proba(X) keeps the original 2-D shape.
            # Build a small adapter that calls the underlying classifier
            # for raw scores, then runs the calibrator on them.

            class _BetaCalProxy:
                __slots__ = ("clf", "cal")

                def __init__(self, clf, cal):
                    self.clf = clf
                    self.cal = cal

                def predict_proba(self, X):
                    raw = self.clf.predict_proba(X)[:, 1]
                    return self.cal.predict_proba(raw)

            cal = _BetaCalProxy(clf, cal)
        else:
            if _HAS_FROZEN:
                cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
            else:
                cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
            cal.fit(Xv.values, yv.values)

        proba = cal.predict_proba(Xs.values)[:, 1]
        log.info(f"fold={k} train={len(inner_train)} val={len(inner_val)} "
                 f"test={len(te_idx)} mean_proba={proba.mean():.3f}")

        res.folds.append(FoldResult(
            fold=k,
            train_size=len(inner_train),
            test_size=len(te_idx),
            test_dates=dates.iloc[te_idx],
            test_tickers=tickers.iloc[te_idx],
            proba=proba,
            y_true=ys.values,
            model=clf if keep_models else None,
            calibrator=cal if keep_models else None,
        ))
    return res
