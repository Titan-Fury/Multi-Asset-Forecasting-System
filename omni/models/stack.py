"""Stacking ensemble for binary direction / rank classification.

Base learners (auto-detect what's installed):
  - XGBoost       (always; same hyperparams as omni.models.xgboost_5d)
  - LightGBM      (optional)
  - CatBoost      (optional)
  - sklearn HistGradientBoostingClassifier  (always — different bias from XGB)

Meta learner: scikit-learn LogisticRegression (L2, C=1.0).

Out-of-fold predictions are produced via PURGED K-fold (leakage-free); the
meta learner is trained on those OOF preds vs the true labels. Inference
runs all bases on the live row, stacks their predictions, then applies the
meta. Each base learner is also kept around so predictions can be hedged
on the inference path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from omni.cv.purged_kfold import purged_kfold_indices
from omni.utils.logger import log


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


lgb = _try_import("lightgbm")
cb = _try_import("catboost")
nf = _try_import("neuralforecast")


@dataclass
class StackResult:
    feature_cols: list[str]
    base_models: dict[str, Any]
    meta: Any
    medians: pd.Series
    oof_aucs: dict[str, float] = field(default_factory=dict)
    oof_meta_auc: float = 0.0
    horizon_days: int = 5

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xn = X.reindex(columns=self.feature_cols).fillna(self.medians).fillna(0.0)
        bases = []
        for name, m in self.base_models.items():
            try:
                if hasattr(m, "predict_proba"):
                    p = m.predict_proba(Xn.to_numpy(dtype=float))[:, 1]
                else:
                    p = m.predict(Xn.to_numpy(dtype=float))
            except Exception as e:  # noqa: BLE001
                log.warning(f"stack base {name} predict failed: {e}")
                p = np.full(len(Xn), 0.5)
            bases.append(p)
        Z = np.column_stack(bases)
        return self.meta.predict_proba(Z)[:, 1]


def _new_xgb():
    import xgboost as xgb
    import os
    kwargs = dict(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, tree_method="hist",
        eval_metric="auc", n_jobs=-1, verbosity=0,
    )
    if os.environ.get("OMNI_XGB_DEVICE", "").strip().lower() in ("cuda", "gpu"):
        kwargs["device"] = "cuda"
    return xgb.XGBClassifier(**kwargs)


def _new_hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=400, max_depth=6, learning_rate=0.05,
        l2_regularization=1.0, random_state=42,
    )


def _new_lgb():
    if lgb is None:
        return None
    import os
    kwargs = dict(
        n_estimators=500, max_depth=-1, learning_rate=0.04,
        num_leaves=63, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, n_jobs=-1, verbose=-1,
    )
    if os.environ.get("OMNI_LGB_DEVICE", "").strip().lower() in ("cuda", "gpu"):
        kwargs["device"] = "gpu"
        kwargs["gpu_platform_id"] = 0
        kwargs["gpu_device_id"] = 0
    return lgb.LGBMClassifier(**kwargs)


def _new_cb():
    if cb is None:
        return None
    return cb.CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        l2_leaf_reg=3.0, eval_metric="AUC", verbose=False,
    )


def _new_nhits(horizon_days: int = 5):
    if nf is None:
        return None
    from omni.models.nhits import NHiTSBinaryClassifier, _NF_AVAILABLE
    if not _NF_AVAILABLE:
        return None
    return NHiTSBinaryClassifier(
        horizon_days=horizon_days,
        input_size=64, max_steps=300,
    )


def fit_stack(
    X: pd.DataFrame,
    y: np.ndarray,
    dates: pd.Series,
    *,
    horizon_days: int = 5,
    n_folds: int = 5,
    embargo_days: int | None = None,
    tickers: pd.Series | None = None,
    closes: pd.Series | None = None,
) -> StackResult:
    """Train a stacking ensemble with purged K-fold OOF predictions.

    Phase 23: ``tickers`` + ``closes`` are optional panel inputs consumed only
    by the N-HiTS base. Other bases ignore them. When both are provided AND
    neuralforecast is installed, NHITS is added as a 5th base learner.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    embargo = horizon_days if embargo_days is None else embargo_days
    folds = purged_kfold_indices(dates, n_folds=n_folds,
                                 horizon_days=horizon_days, embargo_days=embargo)
    if not folds:
        raise RuntimeError("purged_kfold_indices produced no folds")

    # Define base learners once (fresh instances per fold)
    base_specs: dict[str, Any] = {"xgb": _new_xgb, "hgb": _new_hgb}
    if lgb is not None: base_specs["lgb"] = _new_lgb
    if cb is not None:  base_specs["cb"]  = _new_cb
    has_panel = (tickers is not None and closes is not None)
    use_nhits = (nf is not None and has_panel
                 and os.environ.get("OMNI_DISABLE_NHITS", "0") != "1")
    if use_nhits:
        base_specs["nhits"] = lambda: _new_nhits(horizon_days=horizon_days)

    medians = X.median(numeric_only=True).fillna(0.0)
    X_filled = X.fillna(medians).fillna(0.0)
    feat_cols = list(X.columns)
    n = len(X)
    oof = {name: np.full(n, np.nan) for name in base_specs}

    dates_arr = pd.to_datetime(pd.Series(dates).values) if has_panel else None
    tickers_arr = pd.Series(tickers).astype(str).values if has_panel else None
    closes_arr = (pd.to_numeric(pd.Series(closes), errors="coerce").to_numpy()
                  if has_panel else None)

    for k, (tr, va) in enumerate(folds):
        Xtr = X_filled.iloc[tr].to_numpy(dtype=float)
        Xva = X_filled.iloc[va].to_numpy(dtype=float)
        ytr = y[tr]; yva = y[va]
        for name, mk in base_specs.items():
            try:
                m = mk()
                if m is None:
                    continue
                if name == "nhits":
                    m.fit(
                        Xtr, ytr,
                        dates=dates_arr[tr], tickers=tickers_arr[tr],
                        closes=closes_arr[tr], horizon_days=horizon_days,
                    )
                    oof[name][va] = m.predict_panel(
                        dates=dates_arr[va], tickers=tickers_arr[va],
                        closes=closes_arr[va],
                    )
                else:
                    m.fit(Xtr, ytr)
                    if hasattr(m, "predict_proba"):
                        oof[name][va] = m.predict_proba(Xva)[:, 1]
                    else:
                        oof[name][va] = m.predict(Xva)
            except Exception as e:  # noqa: BLE001
                log.warning(f"fold {k} base {name} fit failed: {e}")

    oof_aucs: dict[str, float] = {}
    for name, p in oof.items():
        mask = ~np.isnan(p)
        try:
            if mask.sum() > 50 and len(set(y[mask])) == 2:
                oof_aucs[name] = float(roc_auc_score(y[mask], p[mask]))
        except Exception:  # noqa: BLE001
            # Single-class OOF window, sklearn version glitch, or numeric
            # edge case. AUC is informational; missing it doesn't change
            # the stacked model output.
            pass

    # Meta: train on OOF-stacked features (fill remaining NaNs with 0.5)
    Z = np.column_stack([np.where(np.isnan(p), 0.5, p) for p in oof.values()])
    meta = LogisticRegression(C=1.0, max_iter=200)
    meta.fit(Z, y)
    try:
        meta_oof_auc = float(roc_auc_score(y, meta.predict_proba(Z)[:, 1]))
    except Exception:
        meta_oof_auc = 0.0

    # Re-fit each base on FULL data for inference
    base_models: dict[str, Any] = {}
    for name, mk in base_specs.items():
        try:
            m = mk()
            if m is None:
                continue
            if name == "nhits":
                m.fit(
                    X_filled.to_numpy(dtype=float), y,
                    dates=dates_arr, tickers=tickers_arr, closes=closes_arr,
                    horizon_days=horizon_days,
                )
            else:
                m.fit(X_filled.to_numpy(dtype=float), y)
            base_models[name] = m
        except Exception as e:  # noqa: BLE001
            log.warning(f"final fit base {name} failed: {e}")

    log.info(f"stack: oof_aucs={oof_aucs} meta_auc={meta_oof_auc:.4f}")
    return StackResult(
        feature_cols=feat_cols,
        base_models=base_models,
        meta=meta,
        medians=medians,
        oof_aucs=oof_aucs,
        oof_meta_auc=meta_oof_auc,
        horizon_days=horizon_days,
    )
