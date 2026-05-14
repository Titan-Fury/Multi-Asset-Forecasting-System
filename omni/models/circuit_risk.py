"""Lower-circuit risk classifier (Phase 24.18).

Forward-direction models (universal_5d, stack_h5) are tuned for the *centre*
of the return distribution — they predict E[fwd_ret_5d] which averages over
small daily moves. They cannot catch tail events like a stock hitting the
NSE 10% lower circuit, because circuit-hits are rare (~1-3% of stock-days
for non-microcaps) and live in the deep left tail.

This module trains a *separate* small XGBoost binary classifier that
predicts P(min(fwd_5d_low / today_close - 1) <= -0.095). Sparse-positive
problem → use scale_pos_weight + Average-Precision (PR-AUC) for evaluation,
NOT ROC-AUC which is misleading for imbalanced labels.

Public API:
    fit(df, *, threshold_pct=0.095, horizon=5, n_folds=5) -> CircuitRiskModel
    CircuitRiskModel.predict_proba(X) -> ndarray[N, 2]   (sklearn-compatible)
    save(model, path)  /  load(path)

The model is India-only by design — circuit mechanics differ across markets
and the training script filters to asset_class=='india' before fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omni.cv.purged_kfold import purged_kfold_indices
from omni.utils.logger import log


# Features are price + microstructure + (optional) narrative sentiment.
# All ticker-invariant by Phase 24.17 normalization. The order here is
# also the inference-time alignment order — keep stable.
_DEFAULT_FEATS = (
    # Distance / volatility
    "dist_from_52w_low", "dist_from_52w_high", "atr_pct_14",
    "vol_5", "vol_20", "vol_ratio",
    # Recent returns
    "ret_1d", "ret_5d", "ret_20d",
    # Trend / momentum
    "dist_sma_10", "dist_sma_20", "dist_sma_50",
    "rsi_14", "macd_hist_pct",
    # Microstructure (India-specific)
    "delivery_pct_z_60d",
    # Narrative (Phase 24.18 — may be NaN if news pipeline is offline)
    "panic_keyword_ratio", "news_volume_z_60d",
    "promoter_news_flag", "sebi_action_flag",
)


@dataclass
class CircuitRiskModel:
    """Trained circuit-risk classifier + alignment metadata."""

    feature_cols: list[str]
    base: Any                              # XGBClassifier
    medians: pd.Series
    threshold_pct: float = 0.095
    horizon_days: int = 5
    pr_auc: float = 0.0                    # OOF Average Precision
    base_rate: float = 0.0                 # P(circuit hit) in training data
    n_train_rows: int = 0
    folds_pr_auc: list[float] = field(default_factory=list)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Sklearn-compatible: returns shape (N, 2) — column 1 is P(circuit)."""
        Xn = X.reindex(columns=self.feature_cols).fillna(self.medians).fillna(0.0)
        return self.base.predict_proba(Xn.to_numpy(dtype=float))


def _build_label(df: pd.DataFrame, *, horizon: int, threshold_pct: float
                  ) -> pd.Series:
    """Label = 1 iff the realized forward `horizon`-day return drops below
    -threshold_pct.

    Uses the existing `fwd_ret_<H>d` target column when present (the
    training set already strips raw close/adj_close). When absent, falls
    back to recomputing from the `low` series — useful for live inference
    on a live OHLCV frame which DOES carry low/close.

    Note: fwd_ret_5d is the cumulative 5d return (close/close - 1), which
    is a slightly LESS sensitive proxy than the running min over the
    window. A stock that hit -12% intraday and recovered to -3% at close
    would NOT be labelled here. That's a known limitation; the cumulative
    proxy still catches the bulk of true circuit-day samples and avoids
    needing intraday lows in training.
    """
    if "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError("circuit_risk: need ticker+date columns")

    fwd_col = f"fwd_ret_{horizon}d"
    if fwd_col in df.columns:
        # Preferred path: training-set frames already carry forward returns.
        fwd = pd.to_numeric(df[fwd_col], errors="coerce")
        # mask = NaN only where the forward window is undefined
        label = (fwd <= -threshold_pct).astype(int)
        label = label.where(fwd.notna(), other=np.nan)
        return label

    # Fallback: live OHLCV frame with low/close present.
    out = df.sort_values(["ticker", "date"]).copy()
    if "close" in out.columns:
        price_today = out["close"].astype(float)
    elif "adj_close" in out.columns:
        price_today = out["adj_close"].astype(float)
    else:
        raise ValueError(
            "circuit_risk: no fwd_ret_{H}d, close, or adj_close column "
            "available to compute label"
        )
    low_col = "low" if "low" in out.columns else (
        "close" if "close" in out.columns else "adj_close"
    )
    fwd_low = out.groupby("ticker", observed=True)[low_col].transform(
        lambda s: s.shift(-1).rolling(horizon, min_periods=1).min()
    )
    drawdown = fwd_low / price_today - 1.0
    label = (drawdown <= -threshold_pct).astype(int)
    valid = out.groupby("ticker", observed=True)[low_col].transform(
        lambda s: s.shift(-1).rolling(horizon, min_periods=1).count()
    ) >= 1
    label = label.where(valid, other=np.nan)
    return label


def fit(df_full: pd.DataFrame, *,
        threshold_pct: float = 0.095,
        horizon: int = 5,
        n_folds: int = 5,
        feature_cols: tuple[str, ...] | None = None,
        ) -> CircuitRiskModel:
    """Train a sparse-positive XGBoost on circuit-hit labels.

    Filters to asset_class=='india' if that column is present (other markets
    have different circuit mechanics → train separately if needed).
    """
    import xgboost as xgb
    from sklearn.metrics import average_precision_score

    df = df_full.copy()
    if "asset_class" in df.columns:
        df = df[df["asset_class"].astype(str).str.lower() == "india"].copy()

    feats = list(feature_cols) if feature_cols else [
        c for c in _DEFAULT_FEATS if c in df.columns
    ]
    if not feats:
        raise ValueError("circuit_risk: zero feature columns matched the input frame")

    y = _build_label(df, horizon=horizon, threshold_pct=threshold_pct)
    df = df[y.notna()].copy()
    y = y[y.notna()].astype(int).to_numpy()
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X_filled = X.fillna(medians).fillna(0.0)
    dates = pd.to_datetime(df["date"])

    base_rate = float(y.mean()) if len(y) else 0.0
    if base_rate <= 0:
        log.warning(
            f"circuit_risk: 0 positives in training set "
            f"(threshold={threshold_pct}, horizon={horizon}d). Aborting fit."
        )
        return CircuitRiskModel(
            feature_cols=feats, base=None, medians=medians,
            threshold_pct=threshold_pct, horizon_days=horizon,
            pr_auc=0.0, base_rate=0.0, n_train_rows=len(y),
        )

    spw = float((1 - base_rate) / base_rate) if base_rate < 0.5 else 1.0
    log.info(
        f"circuit_risk: rows={len(y):,} positives={int(y.sum())} "
        f"base_rate={base_rate:.4f} scale_pos_weight={spw:.2f} feats={len(feats)}"
    )

    # OOF eval via purged K-fold + embargo (same as stack)
    folds = purged_kfold_indices(
        dates, n_folds=n_folds, horizon_days=horizon, embargo_days=horizon * 2,
    )
    oof = np.full(len(y), np.nan)
    fold_pr: list[float] = []
    for k, (tr, va) in enumerate(folds):
        m = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            tree_method="hist", eval_metric="aucpr",
            scale_pos_weight=spw, n_jobs=-1, verbosity=0,
        )
        try:
            m.fit(X_filled.iloc[tr].to_numpy(dtype=float), y[tr])
            p = m.predict_proba(X_filled.iloc[va].to_numpy(dtype=float))[:, 1]
            oof[va] = p
            if y[va].sum() > 0:
                fold_pr.append(float(average_precision_score(y[va], p)))
        except Exception as exc:  # noqa: BLE001
            log.warning(f"circuit_risk fold {k}: fit failed: {exc}")

    mask = ~np.isnan(oof)
    if mask.sum() > 50 and y[mask].sum() > 0:
        pr_auc = float(average_precision_score(y[mask], oof[mask]))
    else:
        pr_auc = 0.0

    # Final fit on full data
    final = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        tree_method="hist", eval_metric="aucpr",
        scale_pos_weight=spw, n_jobs=-1, verbosity=0,
    )
    final.fit(X_filled.to_numpy(dtype=float), y)

    log.info(
        f"circuit_risk: OOF PR-AUC={pr_auc:.4f} (vs base_rate={base_rate:.4f}, "
        f"lift={pr_auc/max(base_rate,1e-6):.2f}x) per-fold={fold_pr}"
    )

    return CircuitRiskModel(
        feature_cols=feats, base=final, medians=medians,
        threshold_pct=threshold_pct, horizon_days=horizon,
        pr_auc=pr_auc, base_rate=base_rate,
        n_train_rows=len(y), folds_pr_auc=fold_pr,
    )


def _default_path() -> Path:
    from omni.utils.paths import models_dir
    return models_dir() / "circuit_risk_india.joblib"


# Legacy artifact (pre-fix) was a pickle. Loader still falls back to it
# read-only so an existing trained model isn't lost — but new saves go
# through joblib.
def _legacy_pickle_path() -> Path:
    from omni.utils.paths import models_dir
    return models_dir() / "circuit_risk_india.pkl"


def save(model: CircuitRiskModel, path: Path | None = None) -> Path:
    import joblib
    p = path or _default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    log.info(f"circuit_risk model saved -> {p}")
    return p


def load(path: Path | None = None) -> CircuitRiskModel | None:
    import joblib
    p = path or _default_path()
    if p.exists():
        try:
            return joblib.load(p)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"circuit_risk load failed: {exc}")
            return None
    # Fallback: read legacy pickle if present (one-time migration path).
    legacy = _legacy_pickle_path()
    if legacy.exists():
        try:
            import pickle  # nosec - legacy artifact migration only
            with legacy.open("rb") as f:
                m = pickle.load(f)  # nosec
            log.warning(
                f"circuit_risk: loaded legacy pickle {legacy}; "
                "re-run training to migrate to joblib."
            )
            return m
        except Exception as exc:  # noqa: BLE001
            log.warning(f"circuit_risk legacy pickle load failed: {exc}")
    return None


def band(prob: float | None) -> str | None:
    """Map raw probability to a UI band: low/elevated/high."""
    if prob is None or not np.isfinite(prob):
        return None
    if prob < 0.10:
        return "low"
    if prob < 0.25:
        return "elevated"
    return "high"
