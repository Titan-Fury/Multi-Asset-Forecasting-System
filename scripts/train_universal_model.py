"""Bootstrap the universal XGBoost+isotonic models (1d / 5d / 21d).

LAKE-FIRST: reads pre-built training set from the data lake when present
(produced by ``scripts/build_training_set.py``); falls back to the legacy
live-fetch path of 30 representative tickers if the lake is empty.

For each horizon in --horizons (default 1,5,21):
  1. Load training frame
  2. Walk-forward fit XGBoost+isotonic on target ``fwd_dir_{h}d``
  3. Save to data/models/universal_{h}d.joblib via model_registry.save_horizon

Run after the lake is built:
    python scripts/build_training_set.py
    python scripts/train_universal_model.py --horizons 1,5,21
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omni.lake import io as lake_io       # noqa: E402
from omni.lake import paths as lake_paths # noqa: E402
from omni.models.training_utils import recency_weights  # noqa: E402
from omni.models.xgboost_5d import walkforward_fit  # noqa: E402
from omni.ondemand import model_registry  # noqa: E402
from omni.ondemand.features_inline import build_features_inline  # noqa: E402
from omni.ondemand.lookup import lookup_ticker  # noqa: E402
from omni.ondemand.macro_live import compute_live  # noqa: E402
from omni.utils.logger import log  # noqa: E402

_VALID_HORIZONS = (1, 5, 21)

# Legacy fallback seed universe — used only if lake is empty
_SEEDS = [
    ("AAPL", "us", "Apple"), ("MSFT", "us", "Microsoft"),
    ("GOOGL", "us", "Alphabet A"), ("AMZN", "us", "Amazon"),
    ("META", "us", "Meta"), ("NVDA", "us", "NVIDIA"),
    ("TSLA", "us", "Tesla"), ("JPM", "us", "JPMorgan"),
    ("V", "us", "Visa"), ("UNH", "us", "UnitedHealth"),
    ("XOM", "us", "Exxon"), ("WMT", "us", "Walmart"),
    ("JNJ", "us", "JNJ"), ("HD", "us", "Home Depot"),
    ("MA", "us", "Mastercard"), ("PG", "us", "P&G"),
    ("DIS", "us", "Disney"), ("BAC", "us", "Bank of America"),
    ("PFE", "us", "Pfizer"), ("KO", "us", "Coca-Cola"),
    ("INTC", "us", "Intel"), ("AMD", "us", "AMD"),
    ("ADBE", "us", "Adobe"), ("CRM", "us", "Salesforce"),
    ("RELIANCE", "india", "Reliance"), ("TCS", "india", "TCS"),
    ("HDFCBANK", "india", "HDFC Bank"), ("INFY", "india", "Infosys"),
    ("BTC", "crypto", "Bitcoin"), ("ETH", "crypto", "Ethereum"),
]


def _load_lake_training_set() -> tuple[pd.DataFrame, list[str]]:
    """Try to load the pre-built training set from the lake."""
    out_path = lake_paths.lake_root() / "training" / "training_set.parquet"
    if not out_path.exists():
        return pd.DataFrame(), []
    df = pd.read_parquet(out_path)
    used = (sorted(df["ticker"].dropna().unique().tolist())
              if "ticker" in df.columns else [])
    log.info(f"lake training set loaded: {df.shape} from {len(used)} tickers")
    return df, used


def _build_pooled_live() -> tuple[pd.DataFrame, list[str]]:
    """Legacy live-fetch path — only used when the lake is empty."""
    macro = compute_live(lookback_years=8)
    pooled: list[pd.DataFrame] = []
    used: list[str] = []
    for ticker, ac, name in _SEEDS:
        t0 = time.time()
        try:
            bundle = lookup_ticker(ticker, ac, display_name=name)
        except Exception as e:
            log.warning(f"lookup failed {ac}/{ticker}: {e}")
            continue
        if bundle is None or bundle.prices.empty:
            log.warning(f"no prices {ac}/{ticker}, skipping")
            continue
        feats = build_features_inline(
            ac, ticker, bundle.prices,
            fundamentals=bundle.fundamentals,
            sentiment=bundle.news_sentiment,
            macro=macro if not macro.empty else bundle.macro,
            horizons=(1, 5, 20),
        )
        if feats.empty:
            continue
        pooled.append(feats)
        used.append(f"{ac}/{ticker}")
        log.info(f"+ {ac}/{ticker}: {feats.shape} ({time.time()-t0:.1f}s)")
    if not pooled:
        raise RuntimeError("no usable feature frames pooled")
    return pd.concat(pooled, ignore_index=True), used


def _train_stack_one_horizon(df: pd.DataFrame, used: list[str], *,
                              horizon: int, n_folds: int,
                              target_kind: str) -> dict:
    """Train a stacking ensemble with PURGED K-fold + embargo on one horizon.

    target_kind="dir" uses binary fwd_dir_{H}d.
    target_kind="rank" uses cross-sectional fwd_rank_{H}d binarised at 0.5
                       (above-median performer within (date, asset_class)).

    Saves the stack to data/models/stack_h{H}.pkl. AUC printed; the stack
    is loadable via pickle and exposes `predict_proba(X)`.
    """
    import pickle
    from pathlib import Path
    from omni.features.targets import add_cross_sectional_rank_targets
    from omni.models.stack import fit_stack
    from omni.features.build import is_target_col
    from omni.models.xgboost_5d import _DROP_ALWAYS

    sub = df.copy()
    if target_kind == "rank":
        sub = add_cross_sectional_rank_targets(sub, horizons=(horizon,))
        rank_col = f"fwd_rank_{horizon}d"
        if rank_col not in sub.columns:
            return {"horizon": horizon, "skipped": True, "reason": "rank target build failed"}
        sub = sub.dropna(subset=[rank_col])
        y = (sub[rank_col].astype(float) >= 0.5).astype(int).to_numpy()
    else:
        target = f"fwd_dir_{horizon}d"
        if target not in sub.columns:
            return {"horizon": horizon, "skipped": True, "reason": "no target column"}
        sub = sub.dropna(subset=[target])
        y = (sub[target].astype(int) > 0).astype(int).to_numpy()

    if len(sub) < n_folds * 200:
        return {"horizon": horizon, "skipped": True, "reason": f"too few rows ({len(sub)})"}

    feat_cols = [c for c in sub.columns
                  if c not in _DROP_ALWAYS and not is_target_col(c)
                  and not c.startswith("fwd_rank_")]
    X = sub[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    dates = pd.to_datetime(sub["date"])

    log.info(f"[h={horizon}/stack/{target_kind}] fit_stack on {X.shape}, folds={n_folds}")
    # Phase 23 — thread panel info so N-HiTS can join the stack as a 5th base.
    tickers_panel = sub["ticker"].astype(str) if "ticker" in sub.columns else None
    if "close" in sub.columns:
        closes_panel = sub["close"].astype(float)
    elif "adj_close" in sub.columns:
        closes_panel = sub["adj_close"].astype(float)
    elif "log_ret_1d" in sub.columns and tickers_panel is not None:
        # Reconstruct synthetic close = exp(cumulative log returns) per ticker.
        # Adequate for N-HiTS — relative shape is what matters for forecasting.
        tmp = sub[["ticker", "date", "log_ret_1d"]].copy()
        tmp["log_ret_1d"] = pd.to_numeric(tmp["log_ret_1d"], errors="coerce").fillna(0.0)
        tmp = tmp.sort_values(["ticker", "date"])
        tmp["cum_lret"] = tmp.groupby("ticker")["log_ret_1d"].cumsum()
        tmp["synth_close"] = np.exp(tmp["cum_lret"])
        tmp = tmp.reindex(sub.index)  # back to original order
        closes_panel = tmp["synth_close"].astype(float)
        log.info("N-HiTS: using synthetic close from cumulative log returns")
    else:
        closes_panel = None
    sr = fit_stack(
        X, y, dates,
        horizon_days=horizon, n_folds=n_folds,
        tickers=tickers_panel, closes=closes_panel,
    )

    out_dir = Path("data/models"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stack_h{horizon}.pkl"
    with out_path.open("wb") as f:
        pickle.dump({"stack": sr, "tickers_used": used, "target_kind": target_kind}, f)

    return {
        "horizon": horizon, "skipped": False, "kind": "stack",
        "rows": len(sub), "n_features": len(feat_cols),
        "oof_auc": float(sr.oof_meta_auc),
        "oof_aucs_per_base": sr.oof_aucs,
        "path": str(out_path),
    }


def _train_one_horizon(df: pd.DataFrame, used: list[str], *,
                         horizon: int, n_folds: int, min_train: int,
                         half_life_years: float) -> dict:
    """Fit + save one horizon model. Returns a stats dict."""
    target = f"fwd_dir_{horizon}d"
    if target not in df.columns:
        log.warning(f"target {target} not in training frame; skip h={horizon}")
        return {"horizon": horizon, "skipped": True, "reason": "no target column"}

    sub = df.dropna(subset=[target]).copy()
    if len(sub) < (n_folds + 1) * min_train // 2:
        log.warning(f"too few rows for h={horizon} ({len(sub)}); skip")
        return {"horizon": horizon, "skipped": True, "reason": "too few rows"}

    # Recency weighting: rows further in the past get exponentially less weight
    weights = (recency_weights(sub["date"], half_life_years=half_life_years)
                if "date" in sub.columns else None)

    log.info(f"[h={horizon}] walk-forward fit on {sub.shape}, "
              f"folds={n_folds}, min_train={min_train}, "
              f"recency_half_life={half_life_years}y")
    res = walkforward_fit(
        sub, target=target,
        n_folds=n_folds, min_train=min_train,
        keep_models=True,
    )
    if not res.folds:
        log.error(f"[h={horizon}] no folds produced")
        return {"horizon": horizon, "skipped": True, "reason": "no folds"}

    pred = res.predictions_frame()
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        if pred["y_true"].nunique() == 2:
            auc = float(roc_auc_score(pred["y_true"], pred["proba_up"]))
    except Exception:
        pass

    feat_cols = res.feature_cols
    medians = sub[feat_cols].apply(pd.to_numeric, errors="coerce").median()
    last = res.folds[-1]
    um = model_registry.UniversalModel(
        model=last.model, calibrator=last.calibrator,
        feature_cols=feat_cols, medians=medians,
        trained_at=model_registry.now_iso(),
        tickers_used=used, oof_auc=auc,
        horizon_days=horizon,
    )
    path = model_registry.save_horizon(um, horizon)
    return {"horizon": horizon, "skipped": False,
             "rows": len(sub), "n_features": len(feat_cols),
             "oof_auc": auc, "path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                          help="retrain even if model is < 7 days old")
    parser.add_argument("--min-train", type=int, default=2520,
                          help="min training rows per fold (pooled scale)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--horizons", default="5",
                          help="comma-separated horizons in trading days "
                                "(e.g. '1,5,21' for full multi-horizon)")
    parser.add_argument("--half-life-years", type=float, default=5.0,
                          help="recency-weight half-life; smaller = more recency bias")
    parser.add_argument("--lake-only", action="store_true",
                          help="error if lake training set is missing")
    parser.add_argument("--ensemble", default="single",
                          choices=("single", "stack"),
                          help="single = current XGB+isotonic. stack = XGB+HGB(+LGBM+CB) "
                                "+ logistic meta on PURGED-kfold OOF preds.")
    parser.add_argument("--target", default="dir",
                          choices=("dir", "rank"),
                          help="dir = binary fwd_dir_<H>d. rank = cross-sectional "
                                "percentile rank of fwd_ret (Numerai-style); "
                                "binarised at 0.5 for the classifier head.")
    parser.add_argument("--cv", default="walkforward",
                          choices=("walkforward", "purged"),
                          help="purged = de-Prado purged K-fold + embargo (leakage-free)")
    parser.add_argument("--india-specialist", action="store_true",
                          help="ALSO train an India-only specialist head and save "
                                "to data/models/india_specialist.pkl")
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    invalid = [h for h in horizons if h not in _VALID_HORIZONS]
    if invalid:
        print(f"ERROR: unsupported horizons {invalid}; valid: {_VALID_HORIZONS}")
        return 1

    blob_5d, _ = model_registry._paths_for(5)
    if blob_5d.exists() and not args.force and horizons == [5]:
        age_days = (time.time() - blob_5d.stat().st_mtime) / 86400
        if age_days < 7:
            print(f"universal_5d is {age_days:.1f}d old; --force to retrain")
            return 0

    print("[1/3] loading training frame...")
    df, used = _load_lake_training_set()
    if df.empty:
        if args.lake_only:
            print("ERROR: lake training set missing — run "
                   "`python scripts/build_training_set.py` first.")
            return 1
        print("  lake empty; falling back to live-fetch")
        df, used = _build_pooled_live()

    print(f"[2/3] training {len(horizons)} horizon model(s) on {df.shape}...")
    print(f"      ensemble={args.ensemble}  target={args.target}  cv={args.cv}")
    results = []
    for h in horizons:
        if args.ensemble == "stack":
            r = _train_stack_one_horizon(df, used, horizon=h,
                                          n_folds=args.n_folds,
                                          target_kind=args.target)
        else:
            r = _train_one_horizon(df, used, horizon=h,
                                      n_folds=args.n_folds,
                                      min_train=args.min_train,
                                      half_life_years=args.half_life_years)
        results.append(r)
        if r.get("skipped"):
            print(f"  h={h}: SKIPPED ({r.get('reason')})")
        else:
            extra = (f"  bases={r['oof_aucs_per_base']}" if "oof_aucs_per_base" in r else "")
            print(f"  h={h}: oof_auc={r['oof_auc']:.4f}  rows={r['rows']:,}  "
                   f"feats={r['n_features']}  kind={r.get('kind','single')}{extra}")

    print("[3/3] done.")
    print(f"  available horizons on disk: {model_registry.available_horizons()}")

    if args.india_specialist:
        print("[bonus] training India-specialist head...")
        try:
            from omni.specialists import india as _ind
            spec = _ind.fit(df, horizon=5, n_folds=args.n_folds)
            path = _ind.save(spec)
            print(f"  india specialist: rows={spec.n_train_rows}  "
                   f"feats={len(spec.feature_cols)}  oof_auc={spec.oof_auc:.4f}  "
                   f"path={path}")
        except Exception as e:  # noqa: BLE001
            print(f"  india specialist FAILED: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
