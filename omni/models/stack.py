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
import warnings
from contextlib import contextmanager
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


@contextmanager
def _silence_lgb_feature_name_warning():
    """LightGBM's sklearn wrapper auto-generates synthetic feature names
    (Column_0, Column_1, ...) on fit() even when given a numpy array.
    sklearn >= 1.2 then warns at predict_proba() that "X does not have
    valid feature names" — a false positive since fit and predict both
    used arrays. Suppress just that specific warning so real warnings
    still surface."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*does not have valid feature names.*",
            category=UserWarning,
        )
        yield


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
        # NHITS needs the panel triple (date, ticker, close) — pull from the
        # original X if present so the base contributes real signal at
        # inference, not a constant global mean.
        nhits_panel = None
        if any(name == "nhits" for name in self.base_models):
            try:
                if {"date", "ticker"}.issubset(X.columns):
                    closes_col = (
                        "adj_close" if "adj_close" in X.columns
                        else ("close" if "close" in X.columns else None)
                    )
                    if closes_col is not None:
                        nhits_panel = (
                            X["date"].to_numpy(),
                            X["ticker"].astype(str).to_numpy(),
                            X[closes_col].astype(float).to_numpy(),
                        )
            except Exception as e:  # noqa: BLE001
                log.debug(f"stack: NHITS panel extraction failed: {e}")
                nhits_panel = None

        bases = []
        with _silence_lgb_feature_name_warning():
            for name, m in self.base_models.items():
                try:
                    if name == "nhits" and nhits_panel is not None and hasattr(m, "predict_panel"):
                        d, t, c = nhits_panel
                        p = m.predict_panel(dates=d, tickers=t, closes=c)
                    elif hasattr(m, "predict_proba"):
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
    # Phase 24.22 — same hyperparameter overhaul as xgboost_5d:
    # - learning_rate 0.05 -> 0.03 with n_estimators 400 -> 800
    # - min_child_weight 1 -> 5, reg_alpha 0 -> 0.5
    # The stack picks up the new circuit-history features from the
    # universal feature catalog automatically once retrained.
    kwargs = dict(
        n_estimators=800, max_depth=5, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=5, reg_alpha=0.5, reg_lambda=1.0,
        tree_method="hist",
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


def _new_patchtst(horizon_days: int = 5):
    """Phase 25.1 — PatchTST sequence base. Mirrors NHITS panel-shape contract.

    Default-disabled via OMNI_ENABLE_PATCHTST gate; opt in only after the
    diagnostic shows OOF AUC >= 0.51.
    """
    if nf is None:
        return None
    try:
        from omni.models.patchtst import PatchTSTBinaryClassifier, _PATCHTST_AVAILABLE
    except ImportError:
        return None
    if not _PATCHTST_AVAILABLE:
        return None
    return PatchTSTBinaryClassifier(
        horizon_days=horizon_days,
        input_size=96, max_steps=300,
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
    # Phase 24.18 — pass `tickers` so purging is per-ticker (de Prado AFML §7).
    # Cross-ticker train rows in the val window aren't actually leaks; the
    # global purge throws them out unnecessarily. Per-ticker keeps them.
    folds = purged_kfold_indices(
        dates, n_folds=n_folds,
        horizon_days=horizon_days, embargo_days=embargo,
        tickers=tickers,
    )
    if not folds:
        raise RuntimeError("purged_kfold_indices produced no folds")

    # Define base learners once (fresh instances per fold)
    base_specs: dict[str, Any] = {"xgb": _new_xgb, "hgb": _new_hgb}
    if lgb is not None: base_specs["lgb"] = _new_lgb
    if cb is not None:  base_specs["cb"]  = _new_cb
    has_panel = (tickers is not None and closes is not None)
    # Phase 24.18 — NHITS DEFAULT-DISABLED. Diagnostic showed its OOF AUC was
    # 0.4974 (worse than coin flip) across all three horizons, dragging the
    # ensemble meta-AUC down ~0.005-0.010. Re-enable explicitly with
    # OMNI_ENABLE_NHITS=1 once the wrapper is fixed (see omni/models/nhits.py).
    use_nhits = (nf is not None and has_panel
                 and os.environ.get("OMNI_ENABLE_NHITS", "0") == "1"
                 and os.environ.get("OMNI_DISABLE_NHITS", "0") != "1")
    if use_nhits:
        base_specs["nhits"] = lambda: _new_nhits(horizon_days=horizon_days)
        log.info("stack: NHITS enabled via OMNI_ENABLE_NHITS=1")
    else:
        log.info("stack: NHITS disabled (anti-predictive in diagnostics; "
                 "set OMNI_ENABLE_NHITS=1 to re-enable)")
    # Phase 25.1 — PatchTST sequence base. Default off; opt-in via
    # OMNI_ENABLE_PATCHTST=1 once the diagnostic clears OOF AUC >= 0.51.
    use_patchtst = (nf is not None and has_panel
                     and os.environ.get("OMNI_ENABLE_PATCHTST", "0") == "1")
    if use_patchtst:
        base_specs["patchtst"] = lambda: _new_patchtst(horizon_days=horizon_days)
        log.info("stack: PatchTST enabled via OMNI_ENABLE_PATCHTST=1")

    # Phase 24.22 — DO NOT precompute medians from the entire frame: those
    # medians leak future-fold data into the imputation values used during
    # OOF training of earlier folds. We still need a "global" set of medians
    # for the FINAL fit (computed below from the full X) and as a fallback,
    # but inside the OOF loop each fold gets fold-train medians instead.
    # The mid-pipeline replace([+/-inf, -inf], NaN) stays here so we don't
    # have to repeat it per fold.
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0.0)  # used for inference + as fallback
    feat_cols = list(X.columns)
    n = len(X)
    oof = {name: np.full(n, np.nan) for name in base_specs}

    dates_arr = pd.to_datetime(pd.Series(dates).values) if has_panel else None
    tickers_arr = pd.Series(tickers).astype(str).values if has_panel else None
    closes_arr = (pd.to_numeric(pd.Series(closes), errors="coerce").to_numpy()
                  if has_panel else None)

    for k, (tr, va) in enumerate(folds):
        # Phase 24.22 — fold-train medians, computed only from training rows
        # of THIS fold so validation-fold values never leak into the impute
        # values. Falls back to the global medians for any column that's
        # entirely NaN within this fold (e.g. a feature that hadn't yet been
        # backfilled at the start of the time window).
        fold_med = X.iloc[tr].median(numeric_only=True)
        fold_med = fold_med.fillna(medians).fillna(0.0)
        Xtr = X.iloc[tr].fillna(fold_med).fillna(0.0).to_numpy(dtype=float)
        Xva = X.iloc[va].fillna(fold_med).fillna(0.0).to_numpy(dtype=float)
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
                elif name == "patchtst":
                    # Same panel-shape contract as NHITS.
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
                    with _silence_lgb_feature_name_warning():
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

    # Honest OOF evaluation of the meta-learner: refit a meta on each fold's
    # OOF predictions only (using the same purged folds) so we don't report
    # in-sample AUC. Falls back to predicting 0.5 for the rare case where a
    # fold is degenerate (single-class y_tr).
    meta_oof = np.full(len(y), np.nan)
    for tr, va in folds:
        Ztr, Zva = Z[tr], Z[va]
        ytr = y[tr]
        if len(set(ytr)) < 2:
            meta_oof[va] = float(np.mean(ytr)) if len(ytr) else 0.5
            continue
        try:
            mm = LogisticRegression(C=1.0, max_iter=200)
            mm.fit(Ztr, ytr)
            meta_oof[va] = mm.predict_proba(Zva)[:, 1]
        except Exception as e:  # noqa: BLE001
            log.warning(f"meta OOF fold fit failed: {e}")
            meta_oof[va] = 0.5
    try:
        m_mask = ~np.isnan(meta_oof)
        if m_mask.sum() > 50 and len(set(y[m_mask])) == 2:
            meta_oof_auc = float(roc_auc_score(y[m_mask], meta_oof[m_mask]))
        else:
            meta_oof_auc = 0.0
    except Exception:
        meta_oof_auc = 0.0

    # Re-fit each base on FULL data for inference
    # Phase 24.23 — define X_filled (impute with global medians, then 0 for
    # any column that was all-NaN). Previously this variable was referenced
    # below without being assigned, causing the final base-fit step to raise
    # NameError and silently leave base_models empty (regime stacks shipped
    # with no base learners, which broke inference).
    X_filled = X.fillna(medians).fillna(0.0)
    base_models: dict[str, Any] = {}
    for name, mk in base_specs.items():
        try:
            m = mk()
            if m is None:
                continue
            if name in ("nhits", "patchtst"):
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
