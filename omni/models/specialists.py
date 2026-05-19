"""Specialist models — Step 6 ensemble blending.

Trains 4 narrow XGBoost+isotonic models on disjoint feature subsets:
- technical only      (works in trending markets)
- fundamental only    (works in stable markets)
- sentiment only      (works in news-driven moves)
- macro only          (works at regime transitions)

Each is then blended at predict time using regime-conditional weights from
``config/regime_weights.yaml``.

Public API:
    train_specialists(df, target, *, n_folds, min_train) -> dict[str, Path]
    blend(predictions: dict, regime: str, regime_proba: dict|None=None) -> float
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from omni.utils.logger import log
from omni.utils.paths import models_dir


# Feature group prefixes (loose — extend as new feature modules land)
_TECHNICAL_PREFIXES = (
    "ret_", "sma_", "ema_", "rsi_", "macd_", "bb_", "atr_", "vol_",
    "realized_vol_", "trend_", "mom_vol_adj_", "stoch_", "log_ret_",
)
_FUNDAMENTAL_PREFIXES = (
    "trailing_pe", "forward_pe", "peg_", "price_to_", "ev_to_", "profit_margin",
    "operating_margin", "return_on_equity", "revenue_growth", "earnings_growth",
    "debt_to_equity", "current_ratio", "dividend_yield", "beta", "market_cap",
    "earnings_yield_z", "return_on_equity_z", "profit_margin_z",
    "revenue_growth_z", "debt_to_equity_z", "trailing_pe_pct", "profit_margin_pct",
    "return_on_equity_pct", "revenue_growth_pct", "days_until_earnings",
    "earnings_window",
)
_SENTIMENT_PREFIXES = (
    "sent_mean", "sent_std", "sent_pos_", "sent_neg_", "sent_momentum_",
    "n_articles", "sent_z_", "sent_accel_",
)
_MACRO_PREFIXES = (
    "vix_", "yield_curve_", "dxy_", "dgs", "dff", "cpi", "macro_",
)


def _select_columns(df: pd.DataFrame, prefixes: Iterable[str]) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


@dataclass
class SpecialistBundle:
    name: str
    feature_cols: list[str]
    model: object
    calibrator: object
    medians: pd.Series
    oof_auc: float | None


def _save_specialist(bundle: SpecialistBundle, target: str,
                       horizon: int) -> Path:
    base = models_dir()
    base.mkdir(parents=True, exist_ok=True)
    p = base / f"specialist_{bundle.name}_{horizon}d.joblib"
    joblib.dump({
        "name": bundle.name,
        "feature_cols": bundle.feature_cols,
        "model": bundle.model,
        "calibrator": bundle.calibrator,
        "medians": bundle.medians.to_dict(),
        "oof_auc": bundle.oof_auc,
        "target": target,
    }, p)
    log.info(f"specialist saved: {bundle.name}/{horizon}d auc={bundle.oof_auc}  -> {p}")
    return p


def train_specialists(df: pd.DataFrame, *, target: str = "fwd_dir_5d",
                        n_folds: int = 5, min_train: int = 2520,
                        horizon: int = 5) -> dict[str, Path]:
    """Train one specialist per feature group; persist each to disk."""
    from omni.models.xgboost_5d import walkforward_fit

    groups = {
        "tech":   _select_columns(df, _TECHNICAL_PREFIXES),
        "fund":   _select_columns(df, _FUNDAMENTAL_PREFIXES),
        "sent":   _select_columns(df, _SENTIMENT_PREFIXES),
        "macro":  _select_columns(df, _MACRO_PREFIXES),
    }
    saved: dict[str, Path] = {}
    for name, feat_cols in groups.items():
        if len(feat_cols) < 3:
            log.warning(f"specialist {name}: only {len(feat_cols)} features; skip")
            continue
        sub = df[["date", "ticker", "asset_class", target] + feat_cols].dropna(
            subset=[target]).copy()
        log.info(f"training specialist '{name}' on {len(feat_cols)} features, "
                  f"{len(sub):,} rows")
        res = walkforward_fit(sub, target=target,
                                  n_folds=n_folds, min_train=min_train,
                                  keep_models=True)
        if not res.folds:
            log.warning(f"specialist {name}: no folds")
            continue
        last = res.folds[-1]
        pred = res.predictions_frame()
        try:
            from sklearn.metrics import roc_auc_score
            auc = (float(roc_auc_score(pred["y_true"], pred["proba_up"]))
                    if pred["y_true"].nunique() == 2 else None)
        except Exception:
            auc = None
        # Phase 24.22 — medians used at INFERENCE time should reflect the
        # most-recent training distribution, not include the OOF-evaluated
        # rows. The walk-forward fit's last fold trains on everything before
        # the final test slab, which is the strongest "live" approximation
        # we have. Compute medians from the rows BEFORE the last fold's test
        # window so the inference-time fillna value is leak-free relative
        # to the OOF metric we just reported.
        try:
            last_test_dates = pd.to_datetime(last.test_dates)
            train_cutoff = last_test_dates.min()
            sub_dates = pd.to_datetime(sub["date"])
            train_mask = sub_dates < train_cutoff
            train_sub = sub.loc[train_mask, res.feature_cols]
            if train_sub.empty:
                # Defensive fallback for a malformed last fold
                train_sub = sub[res.feature_cols]
            medians = train_sub.apply(pd.to_numeric, errors="coerce").median()
        except Exception as _exc:
            # Any indexing failure -> fall back to the previous (full-sub)
            # behaviour so the specialist still trains; just log it.
            log.debug(f"specialist {name} median pre-cut slice failed: {_exc}")
            medians = sub[res.feature_cols].apply(pd.to_numeric, errors="coerce").median()
        bundle = SpecialistBundle(
            name=name, feature_cols=res.feature_cols,
            model=last.model, calibrator=last.calibrator,
            medians=medians, oof_auc=auc,
        )
        saved[name] = _save_specialist(bundle, target, horizon)
    return saved


# ---- prediction-time blending --------------------------------------------

# Default regime → weights. Editable via config/regime_weights.yaml later.
_DEFAULT_REGIME_WEIGHTS = {
    "bull":   {"tech": 0.40, "sent": 0.30, "macro": 0.20, "fund": 0.10},
    "bear":   {"macro": 0.40, "fund": 0.30, "tech": 0.15, "sent": 0.15},
    "chop":   {"tech": 0.30, "fund": 0.30, "sent": 0.20, "macro": 0.20},
    "crisis": {"macro": 0.50, "fund": 0.25, "tech": 0.15, "sent": 0.10},
}


def blend(specialist_probas: dict[str, float], regime: str,
            regime_proba: dict[str, float] | None = None) -> float:
    """Soft-blend specialist probabilities by regime.

    Hard mode (``regime_proba=None``): use regime weights directly.
    Soft mode: weight each regime's preferred mix by its probability,
    so transitions are smooth.
    """
    if not specialist_probas:
        return 0.5
    if regime_proba:
        # Soft blend: sum_r P(r) * weighted(r)
        weighted_total = 0.0
        for rg, p_r in regime_proba.items():
            weights = _DEFAULT_REGIME_WEIGHTS.get(rg, _DEFAULT_REGIME_WEIGHTS["chop"])
            denom = sum(w for k, w in weights.items() if k in specialist_probas)
            if denom <= 0:
                continue
            r_blend = sum(specialist_probas[k] * w / denom
                            for k, w in weights.items() if k in specialist_probas)
            weighted_total += p_r * r_blend
        return float(np.clip(weighted_total, 0.0, 1.0))
    weights = _DEFAULT_REGIME_WEIGHTS.get(regime, _DEFAULT_REGIME_WEIGHTS["chop"])
    denom = sum(w for k, w in weights.items() if k in specialist_probas)
    if denom <= 0:
        return float(np.mean(list(specialist_probas.values())))
    out = sum(specialist_probas[k] * w / denom
                for k, w in weights.items() if k in specialist_probas)
    return float(np.clip(out, 0.0, 1.0))


def load_specialists(horizon: int = 5) -> dict[str, dict]:
    """Load all available specialist bundles for a horizon."""
    base = models_dir()
    out: dict[str, dict] = {}
    for f in base.glob(f"specialist_*_{horizon}d.joblib"):
        try:
            b = joblib.load(f)
            out[b["name"]] = b
        except Exception as exc:  # noqa: BLE001
            log.warning(f"specialist load failed {f}: {exc}")
    return out
