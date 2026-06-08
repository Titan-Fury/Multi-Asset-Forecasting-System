"""Meta-labelling (de Prado, AFML §3.6).

The idea:
    1. Primary model -> direction (UP / DOWN / FLAT).
    2. Meta model    -> probability the primary is RIGHT.
    3. Final bet size = primary_direction * meta_prob * Kelly_fraction.

Why it works:
    A precision-tuned secondary classifier on the SAME features plus a
    few primary-model outputs ("primary_p", "primary_conviction",
    "regime", "vol_regime") learns to filter out the calls the primary
    is most uncertain about. Empirically this lifts realised precision
    by 5-10% with no recall hit on the high-conviction tail.

Schema:
    - X : feature frame already used by the primary stack.
    - y_primary : primary direction call ({-1, 0, +1}).
    - y_true : realised triple-barrier label or fwd_dir.
    - meta_y : 1 if (y_primary == y_true and y_primary != 0) else 0.
              i.e. "did the primary make a correct, non-flat call?".

Training signature is intentionally tiny: 30-LOC LightGBM-style
gradient-boosted tree with an early-stopping CV split. We default to
`sklearn.HistGradientBoostingClassifier` so we keep the existing
dependency footprint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class MetaResult:
    auc_oof: float
    n_train: int
    n_pos: int
    base_rate: float
    feature_cols: list[str]


def make_meta_target(y_primary: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """1 iff primary made a correct, non-flat call."""
    yp = np.asarray(y_primary)
    yt = np.asarray(y_true)
    return ((yp != 0) & (yp == yt)).astype(np.int8)


def fit_meta(features: pd.DataFrame,
             y_primary: np.ndarray,
             y_true: np.ndarray,
             *,
             n_splits: int = 5,
             random_state: int = 42) -> tuple[HistGradientBoostingClassifier, MetaResult]:
    """Fit a meta-labelling classifier with time-series CV.

    Args:
        features: numeric feature frame ALREADY aligned with y_primary/y_true.
                  Should include the primary model's own outputs as columns
                  (primary_p, primary_conviction, regime, vol_regime).
        y_primary: -1/0/+1 array, length == len(features).
        y_true: realised direction (-1/0/+1) or triple-barrier label.

    Returns:
        (fitted_model, MetaResult). The fitted model predicts P(meta=1).
    """
    meta_y = make_meta_target(y_primary, y_true)
    mask = ~features.isna().any(axis=1) & np.isfinite(meta_y)
    X = features.loc[mask].to_numpy()
    y = meta_y[mask.values if hasattr(mask, "values") else mask]

    # Out-of-fold AUC via TimeSeriesSplit (no leakage)
    oof = np.zeros_like(y, dtype=float)
    tss = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tss.split(X):
        m = HistGradientBoostingClassifier(
            max_iter=400, max_leaf_nodes=31, learning_rate=0.05,
            l2_regularization=1.0, random_state=random_state,
        )
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]

    try:
        auc = float(roc_auc_score(y, oof)) if y.sum() and (1 - y).sum() else float("nan")
    except Exception:
        auc = float("nan")

    # Final fit on full data
    final = HistGradientBoostingClassifier(
        max_iter=600, max_leaf_nodes=31, learning_rate=0.05,
        l2_regularization=1.0, random_state=random_state,
    )
    final.fit(X, y)

    return final, MetaResult(
        auc_oof=auc,
        n_train=int(mask.sum()),
        n_pos=int(y.sum()),
        base_rate=float(y.mean()) if len(y) else 0.0,
        feature_cols=list(features.columns),
    )


def save_meta(model, result: MetaResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "result": result}, path)


def load_meta(path: Path | str) -> tuple | None:
    path = Path(path)
    if not path.exists():
        return None
    obj = joblib.load(path)
    return obj["model"], obj["result"]


def apply_meta(model, features: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Run inference. Returns P(primary is correct) per row."""
    X = features.reindex(columns=feature_cols).to_numpy()
    # Replace NaNs with column medians (HGB tolerates NaN but we want stability)
    if np.isnan(X).any():
        col_med = np.nanmedian(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_med, idx[1])
    return model.predict_proba(X)[:, 1]
