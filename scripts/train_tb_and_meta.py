"""Phase 24.18 — Train triple-barrier primary + meta-labelling head.

Compares three labelling strategies on the SAME 5d horizon:

  A) baseline:      y = (fwd_dir_5d > 0).astype(int)        # current
  B) triple-barr:   y = (tb_label_5d > 0).astype(int)       # PT hit before SL
  C) tb-strict:     y = (tb_label_5d == 1).astype(int)      # only PT, exclude time-expired

For each, we train a quick stack using fit_stack on purged k-fold and
report OOF meta AUC. Then we train a META-LABELLING head on top of the
best primary: predicts P(primary's call is correct).

Saves results to data/models/tb_meta_comparison.json so the choice can
be made reproducibly. Best primary saved to data/models/stack_h5_tb.joblib;
meta head to data/models/meta_h5.joblib.
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


def _binary_oof_auc(X_arr, y, dates, *, label: str, n_folds: int = 5,
                    horizon: int = 5) -> dict:
    """Run a single XGBoost walk-forward on purged folds. Returns OOF AUC + the
    per-row OOF probabilities so we can build a meta-labelling head."""
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
        log.info(f"  [{label}] fold {k+1}/{len(folds)} done")

    mask = ~np.isnan(oof)
    auc = float("nan")
    if mask.sum() > 50 and len(set(y[mask])) == 2:
        try:
            auc = float(roc_auc_score(y[mask], oof[mask]))
        except Exception:
            pass
    return {"label": label, "oof_auc": auc, "oof_proba": oof,
            "n_pos": int(y.sum()), "n_total": int(len(y)),
            "base_rate": float(y.mean())}


def main() -> int:
    if not TRAIN.exists():
        log.error(f"no training set at {TRAIN}; run augment_triple_barrier.py first")
        return 1

    log.info(f"loading {TRAIN}...")
    df = pd.read_parquet(TRAIN)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    log.info(f"after dedup: {df.shape[0]:,} rows")

    # Verify required columns
    required = ["fwd_dir_5d", "tb_label_5d", "log_ret_1d", "ticker", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"missing columns: {missing}")
        return 1

    # Drop rows with no horizon-5 label
    df = df[df["fwd_dir_5d"].notna() & df["tb_label_5d"].notna()].copy()
    log.info(f"with valid h=5 labels: {df.shape[0]:,} rows")

    # Build feature matrix: take everything except identifiers / labels.
    drop_cols = {"date", "ticker", "asset_class", "run_date", "asof_date"}
    drop_cols |= {c for c in df.columns
                  if c.startswith(("fwd_", "tb_label_", "tb_first_"))}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0.0)
    X_filled = X.fillna(medians).fillna(0.0)
    X_arr = X_filled.to_numpy(dtype=float)
    dates = pd.to_datetime(df["date"])
    log.info(f"feature matrix: {X_arr.shape}, n_features={len(feat_cols)}")

    # Three label variants
    y_a = (df["fwd_dir_5d"] > 0).astype(int).to_numpy()
    y_b = (df["tb_label_5d"] > 0).astype(int).to_numpy()
    y_c = (df["tb_label_5d"] == 1).astype(int).to_numpy()  # same as y_b in our case
    log.info(f"baselines: A=fwd_dir>0 base_rate={y_a.mean():.4f}  "
             f"B=tb_label>0 base_rate={y_b.mean():.4f}  "
             f"C=tb_label==1 base_rate={y_c.mean():.4f}")

    results = []
    t0 = time.time()
    for name, y in [("A_fwd_dir", y_a), ("B_tb_label", y_b)]:
        log.info(f"training primary {name}...")
        r = _binary_oof_auc(X_arr, y, dates, label=name)
        results.append(r)
        log.info(f"  -> OOF AUC = {r['oof_auc']:.4f}")

    log.info(f"primary training done in {time.time()-t0:.1f}s")

    # Meta-labelling: best primary -> meta head
    best = max(results, key=lambda r: r["oof_auc"] or 0)
    log.info(f"meta-labelling on top of best primary: {best['label']} "
             f"(AUC={best['oof_auc']:.4f})")
    primary_proba = best["oof_proba"]
    primary_dir = np.where(primary_proba > 0.5, 1, -1)
    # y_true for meta = realised direction from triple-barrier
    y_true_dir = np.where(df["tb_label_5d"] > 0, 1,
                          np.where(df["tb_label_5d"] < 0, -1, 0)).astype(np.int8)

    # Features for meta = primary outputs + a few risk-state cols
    meta_features = pd.DataFrame({
        "primary_p": primary_proba,
        "primary_conviction": np.abs(primary_proba - 0.5) * 200,
        "primary_dir": primary_dir.astype(float),
    })
    # Add a few raw risk features
    for c in ("rsi_14", "atr_pct_14", "vol_ratio", "dist_from_52w_high",
              "dist_from_52w_low", "macd_hist_pct", "ret_5d", "ret_20d"):
        if c in df.columns:
            meta_features[c] = X_filled[c].to_numpy()

    meta_model, meta_result = fit_meta(meta_features, primary_dir, y_true_dir,
                                        n_splits=5, random_state=42)
    log.info(f"META: AUC={meta_result.auc_oof:.4f} "
             f"n_pos={meta_result.n_pos:,} base_rate={meta_result.base_rate:.4f}")

    # Save artifacts
    MODELS.mkdir(parents=True, exist_ok=True)
    save_meta(meta_model, meta_result, MODELS / "meta_h5.joblib")

    out = {
        "primary": [{k: v for k, v in r.items() if k != "oof_proba"}
                    for r in results],
        "meta": {
            "auc_oof": meta_result.auc_oof,
            "n_train": meta_result.n_train,
            "n_pos": meta_result.n_pos,
            "base_rate": meta_result.base_rate,
            "feature_cols": meta_result.feature_cols,
        },
        "best_primary": best["label"],
    }
    (MODELS / "tb_meta_comparison.json").write_text(json.dumps(out, indent=2))
    log.info(f"comparison saved -> {MODELS / 'tb_meta_comparison.json'}")

    print("\n" + "=" * 78)
    print("  RESULTS — primary label comparison + meta head")
    print("=" * 78)
    for r in results:
        n_pos = r["n_pos"]
        n_total = r["n_total"]
        print(f"  {r['label']:<14s}  AUC={r['oof_auc']:.4f}  "
              f"base_rate={r['base_rate']:.4f}  "
              f"n_pos={n_pos:,}/{n_total:,}")
    print(f"\n  meta on top of {best['label']}:  AUC={meta_result.auc_oof:.4f}  "
          f"base_rate={meta_result.base_rate:.4f}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
