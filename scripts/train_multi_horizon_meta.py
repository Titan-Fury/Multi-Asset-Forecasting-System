"""Phase 24.18 — Train one meta-labelling head per horizon.

Earlier work shipped a single meta_h5.joblib head. h=1 has the strongest
primary stack (OOF AUC 0.69) so a dedicated meta head there should yield
the biggest secondary lift. h=21 is at coin-flip and won't benefit.

This script:
  1. Loads the TB-augmented training set
  2. For each horizon in {1, 5, 21}:
       a. Trains a primary XGB on tb_label_<H>d == 1
       b. Trains a meta head on (primary outputs + risk-state cols)
       c. Saves to data/models/meta_h<H>.joblib

Run after scripts/augment_triple_barrier.py.

Outputs:
  data/models/meta_h1.joblib
  data/models/meta_h5.joblib   (overwrites prior single-horizon meta)
  data/models/meta_h21.joblib
  data/models/meta_comparison.json  — per-horizon AUCs
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from omni.cv.purged_kfold import purged_kfold_indices
from omni.modeling.meta_label import fit_meta, save_meta
from omni.utils.logger import log

MODELS = Path("data/models")
TRAIN = Path("data/lake/training/training_set.parquet")


def _train_primary_oof(X_arr, y, dates, *, horizon: int, n_folds: int = 5):
    """Walk-forward OOF probas via purged k-fold + embargo."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    folds = purged_kfold_indices(dates, n_folds=n_folds,
                                  horizon_days=horizon, embargo_days=horizon)
    oof = np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(folds):
        m = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            tree_method="hist", eval_metric="auc",
            n_jobs=-1, verbosity=0,
        )
        m.fit(X_arr[tr], y[tr])
        oof[va] = m.predict_proba(X_arr[va])[:, 1]
        log.info(f"  [h={horizon}] primary fold {k+1}/{len(folds)}")

    mask = ~np.isnan(oof)
    auc = float("nan")
    if mask.sum() > 50 and len(set(y[mask])) == 2:
        try:
            auc = float(roc_auc_score(y[mask], oof[mask]))
        except Exception:
            pass
    return oof, auc


def _train_one(df: pd.DataFrame, X_filled: pd.DataFrame, X_arr,
               horizon: int, dates: pd.Series) -> dict:
    """Train primary + meta for one horizon. Returns metrics dict."""
    tb_col = f"tb_label_{horizon}d"
    if tb_col not in df.columns:
        log.warning(f"h={horizon}: {tb_col} missing; skip")
        return {"horizon": horizon, "skipped": True,
                "reason": f"{tb_col} missing"}

    sub_mask = df[tb_col].notna().to_numpy()
    if sub_mask.sum() < 1000:
        log.warning(f"h={horizon}: only {sub_mask.sum()} non-null; skip")
        return {"horizon": horizon, "skipped": True, "reason": "too few rows"}

    y = (df.loc[sub_mask, tb_col].astype(int) == 1).astype(int).to_numpy()
    X_arr_h = X_arr[sub_mask]
    dates_h = dates[sub_mask]

    log.info(f"h={horizon}: training primary (n={len(y):,}, "
             f"base_rate={y.mean():.4f})")
    t0 = time.time()
    primary_oof, primary_auc = _train_primary_oof(X_arr_h, y, dates_h,
                                                    horizon=horizon)
    log.info(f"h={horizon}: primary OOF AUC={primary_auc:.4f} "
             f"in {time.time()-t0:.1f}s")

    if primary_auc < 0.51:
        log.warning(f"h={horizon}: primary AUC too weak; skip meta head")
        return {"horizon": horizon, "primary_auc": primary_auc,
                "meta_auc": None, "skipped": "primary too weak"}

    # Meta head — same scaffolding as scripts/train_tb_and_meta.py
    primary_dir = np.where(primary_oof > 0.5, 1, -1)
    y_true_dir = np.where(df.loc[sub_mask, tb_col] > 0, 1,
                          np.where(df.loc[sub_mask, tb_col] < 0, -1, 0)
                          ).astype(np.int8)

    meta_features = pd.DataFrame({
        "primary_p": primary_oof,
        "primary_conviction": np.abs(primary_oof - 0.5) * 200,
        "primary_dir": primary_dir.astype(float),
    })
    risk_cols = ("rsi_14", "atr_pct_14", "vol_ratio",
                 "dist_from_52w_high", "dist_from_52w_low",
                 "macd_hist_pct", "ret_5d", "ret_20d")
    for c in risk_cols:
        if c in X_filled.columns:
            meta_features[c] = X_filled.loc[sub_mask, c].to_numpy()

    log.info(f"h={horizon}: fitting meta head ({meta_features.shape[1]} feats)")
    meta_model, meta_result = fit_meta(meta_features, primary_dir, y_true_dir,
                                        n_splits=5, random_state=42)
    log.info(f"h={horizon}: meta OOF AUC={meta_result.auc_oof:.4f} "
             f"base_rate={meta_result.base_rate:.4f}")

    out_path = MODELS / f"meta_h{horizon}.joblib"
    save_meta(meta_model, meta_result, out_path)
    log.info(f"h={horizon}: saved -> {out_path}")

    return {
        "horizon": horizon,
        "primary_auc": primary_auc,
        "meta_auc": meta_result.auc_oof,
        "meta_n_train": meta_result.n_train,
        "meta_base_rate": meta_result.base_rate,
        "meta_path": str(out_path),
    }


def main() -> int:
    if not TRAIN.exists():
        log.error(f"no training set at {TRAIN}; "
                  "run scripts/augment_triple_barrier.py first")
        return 1

    log.info(f"loading {TRAIN}...")
    df = pd.read_parquet(TRAIN)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    log.info(f"after dedup: {df.shape[0]:,} rows")

    drop_cols = {"date", "ticker", "asset_class", "run_date", "asof_date"}
    drop_cols |= {c for c in df.columns
                  if c.startswith(("fwd_", "tb_label_", "tb_first_", "fwd_rank_"))}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0.0)
    X_filled = X.fillna(medians).fillna(0.0)
    X_arr = X_filled.to_numpy(dtype=float)
    dates = pd.to_datetime(df["date"])
    log.info(f"feature matrix: {X_arr.shape}")

    results = []
    for h in (1, 5, 21):
        r = _train_one(df, X_filled, X_arr, h, dates)
        results.append(r)

    MODELS.mkdir(parents=True, exist_ok=True)
    out = {"horizons": results}
    (MODELS / "meta_comparison.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print("  RESULTS — multi-horizon meta heads")
    print("=" * 78)
    for r in results:
        if r.get("skipped"):
            print(f"  h={r['horizon']:>2}: SKIPPED ({r.get('reason')})")
        else:
            print(f"  h={r['horizon']:>2}: primary AUC={r['primary_auc']:.4f}  "
                  f"meta AUC={r['meta_auc']:.4f}  "
                  f"-> {r['meta_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
