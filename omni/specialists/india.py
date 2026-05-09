"""India-specialist model head.

Trains an XGBoost classifier on India-only rows of the lake training set,
with hyperparameters tuned for the smaller universe and the specific
characteristics of NSE/BSE data:

  - smaller learning_rate (more shrinkage, less variance)
  - shallower trees (fewer rows → simpler model generalises better)
  - higher reg_lambda (India fundamentals are noisier)
  - calibrated isotonic regression on a 20% holdout

Saves to data/models/india_specialist.pkl. Loadable via load() — exposes
predict_proba(X) compatible with the universal-model interface.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omni.utils.logger import log


@dataclass
class IndiaSpecialist:
    model: Any
    calibrator: Any
    feature_cols: list[str]
    medians: pd.Series
    horizon_days: int
    oof_auc: float
    n_train_rows: int
    trained_at: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xn = X.reindex(columns=self.feature_cols).fillna(self.medians).fillna(0.0)
        Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = self.model.predict_proba(Xn.to_numpy(dtype=float))[:, 1]
        return self.calibrator.predict(raw)


_PATH = Path("data/models/india_specialist.pkl")


def fit(df_full: pd.DataFrame, *, horizon: int = 5, n_folds: int = 5) -> IndiaSpecialist:
    """Train an India-specialist on the india slice of the training set."""
    from sklearn.calibration import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    from omni.cv.purged_kfold import purged_kfold_indices
    from omni.features.build import is_target_col
    from omni.models.xgboost_5d import _DROP_ALWAYS

    if "asset_class" not in df_full.columns:
        raise ValueError("training frame missing 'asset_class' column")
    df = df_full[df_full["asset_class"].astype(str).str.lower() == "india"].copy()
    target = f"fwd_dir_{horizon}d"
    if target not in df.columns:
        raise ValueError(f"target {target!r} missing from training frame")
    df = df.dropna(subset=[target])
    if len(df) < n_folds * 200:
        raise ValueError(f"India slice too small for training: {len(df)} rows")

    feat_cols = [c for c in df.columns
                  if c not in _DROP_ALWAYS and not is_target_col(c)]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0.0)
    X_filled = X.fillna(medians).fillna(0.0)
    y = (df[target].astype(int) > 0).astype(int).to_numpy()
    dates = pd.to_datetime(df["date"])

    # OOF predictions via purged K-fold + embargo
    folds = purged_kfold_indices(dates, n_folds=n_folds,
                                 horizon_days=horizon, embargo_days=horizon)
    oof = np.full(len(df), np.nan)
    for tr, va in folds:
        m = XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.80,
            reg_lambda=2.0, tree_method="hist",
            eval_metric="auc", n_jobs=-1, verbosity=0,
        )
        m.fit(X_filled.iloc[tr].to_numpy(dtype=float), y[tr])
        oof[va] = m.predict_proba(X_filled.iloc[va].to_numpy(dtype=float))[:, 1]

    mask = ~np.isnan(oof)
    try:
        oof_auc = float(roc_auc_score(y[mask], oof[mask])) if mask.sum() > 50 else 0.5
    except Exception:
        oof_auc = 0.5

    # Final model on full data
    final = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.80,
        reg_lambda=2.0, tree_method="hist",
        eval_metric="auc", n_jobs=-1, verbosity=0,
    )
    final.fit(X_filled.to_numpy(dtype=float), y)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof[mask], y[mask])

    spec = IndiaSpecialist(
        model=final, calibrator=iso, feature_cols=feat_cols,
        medians=medians, horizon_days=horizon, oof_auc=oof_auc,
        n_train_rows=int(len(df)),
        trained_at=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    log.info(f"india specialist: rows={len(df)} feats={len(feat_cols)} oof_auc={oof_auc:.4f}")
    return spec


def save(spec: IndiaSpecialist) -> Path:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    with _PATH.open("wb") as f:
        pickle.dump(spec, f)
    return _PATH


def load() -> IndiaSpecialist | None:
    if not _PATH.exists():
        return None
    try:
        with _PATH.open("rb") as f:
            return pickle.load(f)
    except Exception as e:  # noqa: BLE001
        log.warning(f"india_specialist load failed: {e}")
        return None


def is_available() -> bool:
    return _PATH.exists()
