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

from omni.lake import paths as lake_paths  # noqa: E402
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

    Saves the stack to data/models/stack_h{H}.joblib. AUC printed; the stack
    is loadable via joblib and exposes `predict_proba(X)`.
    """
    from pathlib import Path

    import joblib

    from omni.features.build import is_target_col
    from omni.features.targets import add_cross_sectional_rank_targets
    from omni.models.stack import fit_stack
    from omni.models.xgboost_5d import _DROP_ALWAYS

    sub = df.copy()
    if target_kind == "rank":
        sub = add_cross_sectional_rank_targets(sub, horizons=(horizon,))
        rank_col = f"fwd_rank_{horizon}d"
        if rank_col not in sub.columns:
            return {"horizon": horizon, "skipped": True, "reason": "rank target build failed"}
        sub = sub.dropna(subset=[rank_col])
        y = (sub[rank_col].astype(float) >= 0.5).astype(int).to_numpy()
    elif target_kind == "tb":
        # Phase 24.18 — triple-barrier label. Requires augment_triple_barrier
        # to have been run on the training set first. tb_label_<H>d is in
        # {-1, 0, +1}; we binarise as (== +1) so the positive class is
        # "PT hit before SL within H days" (de Prado AFML §3.4).
        # In a 622K-row diagnostic this lifted OOF AUC from 0.513 (fwd_dir)
        # to 0.602 — the single biggest target-redesign win in the project.
        tb_col = f"tb_label_{horizon}d"
        if tb_col not in sub.columns:
            # Phase 25 / Item 4: graceful fallback when tb columns are
            # missing. tb is now the default --target so we must not
            # hard-fail when callers haven't run augment_triple_barrier
            # yet — just log a warning and use fwd_dir.
            log.warning(
                f"[h={horizon}] tb_label missing; falling back to fwd_dir "
                f"(run scripts/augment_triple_barrier.py to use TB labels)"
            )
            target = f"fwd_dir_{horizon}d"
            if target not in sub.columns:
                return {"horizon": horizon, "skipped": True,
                        "reason": "no target column (tb_label and fwd_dir both missing)"}
            sub = sub.dropna(subset=[target])
            y = (sub[target].astype(int) > 0).astype(int).to_numpy()
        else:
            sub = sub.dropna(subset=[tb_col])
            y = (sub[tb_col].astype(int) == 1).astype(int).to_numpy()
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
    out_path = out_dir / f"stack_h{horizon}.joblib"
    joblib.dump({"stack": sr, "tickers_used": used, "target_kind": target_kind}, out_path)

    return {
        "horizon": horizon, "skipped": False, "kind": "stack",
        "rows": len(sub), "n_features": len(feat_cols),
        "oof_auc": float(sr.oof_meta_auc),
        "oof_aucs_per_base": sr.oof_aucs,
        "path": str(out_path),
    }


def _parse_half_life(spec: str) -> dict[int, float] | float:
    """Parse the --half-life-years value into either a float (uniform across
    horizons, legacy) or a {horizon: half_life_years} dict.

    Per-horizon half-life is the principled choice: a 1-day fwd_dir label
    is mostly explained by the *current* market regime, so a 5y half-life
    dilutes the recent signal. A 21-day label is broader and benefits from
    a longer recall. Default mapping {1: 1.5, 5: 3.0, 21: 5.0} encodes that.
    """
    if "," in spec or ":" in spec:
        out: dict[int, float] = {}
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            h_s, y_s = tok.split(":")
            out[int(h_s)] = float(y_s)
        return out
    return float(spec)


def _half_life_for(spec: dict[int, float] | float, horizon: int) -> float:
    """Resolve effective half-life for one horizon, with safe default."""
    if isinstance(spec, dict):
        if horizon in spec:
            return spec[horizon]
        # Pick the closest known horizon by absolute distance
        nearest = min(spec.keys(), key=lambda k: abs(k - horizon))
        return spec[nearest]
    return float(spec)


def _train_one_horizon(df: pd.DataFrame, used: list[str], *,
                         horizon: int, n_folds: int, min_train: int,
                         half_life_years: float,
                         target_kind: str = "dir") -> dict:
    """Fit + save one horizon model. Returns a stats dict.

    Phase 28.3 — added `target_kind` parameter so single-mode honours
    `--target tb` (was previously hardcoded to `fwd_dir_<H>d`, silently
    ignoring the CLI flag). Mirrors the dispatch logic from the
    stack-mode handler:
        "dir" -> fwd_dir_<H>d, binary > 0  (legacy default)
        "tb"  -> tb_label_<H>d, binary == 1; falls back to fwd_dir
                 with a warning if augment_triple_barrier hasn't been run.
    Recency weights are now passed through to walkforward_fit (was
    silently dropped before this fix, making --half-life-years a no-op
    in single mode).
    """
    sub = df.copy()

    if target_kind == "tb":
        tb_col = f"tb_label_{horizon}d"
        if tb_col in sub.columns:
            target = tb_col
            sub = sub.dropna(subset=[target])
            log.info(f"[h={horizon}] using TB labels: {target}, "
                      f"binarising as (== +1)")
            # Binarise tb_label ∈ {-1, 0, +1} into {0, 1} for the classifier.
            sub[target] = (sub[target].astype(int) == 1).astype(int)
        else:
            log.warning(
                f"[h={horizon}] tb_label missing; falling back to fwd_dir "
                f"(run scripts/augment_triple_barrier.py to use TB labels)"
            )
            target = f"fwd_dir_{horizon}d"
            if target not in sub.columns:
                return {"horizon": horizon, "skipped": True,
                          "reason": "no target column (tb and fwd_dir both missing)"}
            sub = sub.dropna(subset=[target])
    else:
        target = f"fwd_dir_{horizon}d"
        if target not in sub.columns:
            log.warning(f"target {target} not in training frame; skip h={horizon}")
            return {"horizon": horizon, "skipped": True, "reason": "no target column"}
        sub = sub.dropna(subset=[target])

    if len(sub) < (n_folds + 1) * min_train // 2:
        log.warning(f"too few rows for h={horizon} ({len(sub)}); skip")
        return {"horizon": horizon, "skipped": True, "reason": "too few rows"}

    # Recency weighting: rows further in the past get exponentially less weight.
    # Phase 28.3 — these are now passed through to walkforward_fit instead of
    # being silently dropped.
    weights = (recency_weights(sub["date"], half_life_years=half_life_years)
                if "date" in sub.columns else None)

    log.info(f"[h={horizon}] walk-forward fit on {sub.shape}, "
              f"folds={n_folds}, min_train={min_train}, "
              f"recency_half_life={half_life_years}y, target={target}")
    res = walkforward_fit(
        sub, target=target,
        n_folds=n_folds, min_train=min_train,
        keep_models=True,
        sample_weight=weights,
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
    except Exception as exc:  # noqa: BLE001
        # AUC computation is informational on the train script — a missing
        # value just shows as N/A in the report, doesn't affect the saved
        # model. Print at INFO so it's not totally invisible.
        print(f"[train] OOS AUC unavailable: {exc}")

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
             "oof_auc": auc, "path": str(path),
             "target": target, "target_kind": target_kind}


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
    parser.add_argument("--half-life-years", type=str, default="1:1.5,5:3.0,21:5.0",
                          help="recency-weight half-life. Either a single float "
                               "(applied to every horizon) or a comma-separated "
                               "'h:y' map (e.g. '1:1.5,5:3.0,21:5.0'). "
                               "Short horizons get tighter half-lives — h=1 day "
                               "labels are dominated by recent regime, so a 5y "
                               "half-life dilutes the signal. Default is the "
                               "per-horizon map; pass a single float for the "
                               "legacy uniform behaviour.")
    parser.add_argument("--lake-only", action="store_true",
                          help="error if lake training set is missing")
    parser.add_argument("--ensemble", default="single",
                          choices=("single", "stack"),
                          help="single = current XGB+isotonic. stack = XGB+HGB(+LGBM+CB) "
                                "+ logistic meta on PURGED-kfold OOF preds.")
    parser.add_argument("--target", default="tb",
                          choices=("dir", "rank", "tb"),
                          help="dir = binary fwd_dir_<H>d. rank = cross-sectional "
                                "percentile rank of fwd_ret (Numerai-style); "
                                "binarised at 0.5 for the classifier head. "
                                "tb = triple-barrier (de Prado): label=1 iff price "
                                "hits +2-sigma before -1-sigma within the horizon. "
                                "Phase 25 / Item 4: tb is now the default because it "
                                "lifts h=5 OOF AUC ~0.09 vs dir on the same features. "
                                "If tb_label_<H>d columns are missing, the trainer "
                                "automatically falls back to dir with a warning. "
                                "Run scripts/augment_triple_barrier.py to add tb columns "
                                "to an existing training set.")
    parser.add_argument("--cv", default="walkforward",
                          choices=("walkforward", "purged"),
                          help="purged = de-Prado purged K-fold + embargo (leakage-free)")
    parser.add_argument("--india-specialist", action="store_true",
                          help="ALSO train an India-only specialist head and save "
                                "to data/models/india_specialist.pkl")
    parser.add_argument("--asset-classes", default=None,
                          help="comma-separated asset_class filter for training "
                                "(e.g. 'india' or 'us,india'). Default: all classes "
                                "in the training set. Use this to train a market-"
                                "focused universal model — pairs well with the new "
                                "ticker-invariant feature set (Phase 24.17) so a "
                                "model trained only on Indian equities doesn't have "
                                "the cross-currency scale problem to solve.")
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    invalid = [h for h in horizons if h not in _VALID_HORIZONS]
    if invalid:
        print(f"ERROR: unsupported horizons {invalid}; valid: {_VALID_HORIZONS}")
        return 1

    half_life_spec = _parse_half_life(args.half_life_years)
    if isinstance(half_life_spec, dict):
        print(f"  recency half-life per horizon: {half_life_spec}")
    else:
        print(f"  recency half-life (uniform): {half_life_spec}y")

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

    # Phase 24.17 — optional asset-class filter so an "india-only" universal
    # model can be trained from the same pooled lake. Done before the row
    # count check so an India-only run doesn't error on min_train when the
    # full pool would have been fine.
    if args.asset_classes:
        wanted = {ac.strip().lower() for ac in args.asset_classes.split(",") if ac.strip()}
        if "asset_class" not in df.columns:
            print("ERROR: --asset-classes requested but training set has no "
                   "'asset_class' column")
            return 1
        before = len(df)
        df = df[df["asset_class"].astype(str).str.lower().isin(wanted)].reset_index(drop=True)
        used = [t for t in used
                  if any(t.lower().startswith(f"{ac}/") for ac in wanted)
                  or t.lower() in wanted
                  or "/" not in t]   # keep bare-ticker entries (lake mode)
        print(f"  asset-class filter ({sorted(wanted)}): {before:,} -> {len(df):,} rows")
        if df.empty:
            print(f"ERROR: 0 rows left after filtering to {sorted(wanted)}")
            return 1

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
                                      half_life_years=_half_life_for(half_life_spec, h),
                                      target_kind=args.target)
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
