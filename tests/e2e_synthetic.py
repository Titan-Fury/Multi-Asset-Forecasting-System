"""End-to-end synthetic-data smoke test.

Generates fake OHLCV for a handful of tickers, then runs:
   technical features -> targets -> XGBoost walk-forward -> rubric+blend+gate
   -> backtest -> persist -> dashboard cache load.

Skips ingest stages and macro/news (those are I/O bound and tested separately).
The point is to exercise EVERY in-process code path with a controlled dataset.

Run with:
    .venv\\Scripts\\python.exe tests\\e2e_synthetic.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def make_synthetic_ohlcv(ticker: str, n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Geometric Brownian motion with mild drift + volatility clustering."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n)
    # Drift varies per ticker so the model has signal to find
    drift = rng.uniform(-0.0003, 0.0008)
    vol = rng.uniform(0.008, 0.022)
    rets = rng.normal(drift, vol, size=n)
    # Inject a regime shift halfway: vol doubles for a 60-day stretch
    rets[400:460] *= 2.5
    price = 100.0 * np.exp(np.cumsum(rets))
    high = price * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = price * (1 + rng.normal(0, 0.002, n))
    close = price
    adj_close = price
    volume = rng.integers(1_000_000, 10_000_000, n)
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "adj_close": adj_close, "volume": volume,
        "ticker": ticker,
    })


def make_synthetic_macro(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n)
    vix = 18 + rng.normal(0, 4, n).cumsum() * 0.05
    vix = np.clip(vix, 10, 60)
    vix[400:460] += 15  # spike during the stress window
    return pd.DataFrame({
        "date": dates,
        "vix_level": vix,
        "yield_curve_2_10": rng.normal(0.005, 0.002, n),
        "dxy_chg_5d": rng.normal(0, 0.005, n),
    })


def make_synthetic_spx(n: int = 800) -> pd.DataFrame:
    """SPX-like proxy for regime feature builder."""
    return make_synthetic_ohlcv("^GSPC", n=n, seed=42)


def step(name: str, fn):
    print(f"\n--- {name} ---")
    try:
        out = fn()
        print(f"OK: {name}")
        return out
    except Exception as e:
        print(f"FAIL: {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


def main() -> int:
    from omni.features import technical as tech
    from omni.features import targets as tgt
    from omni.models.xgboost_5d import walkforward_fit
    from omni.regime.hmm import build_regime_features, fit as hmm_fit
    from omni.rubric.scorer import score_frame as rubric_score
    from omni.ensemble.combine import blend, BlendConfig
    from omni.ensemble.gate import apply as gate_apply, GateConfig
    from omni.backtest.runner import run_backtest

    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]

    # 1) Build per-ticker features
    def build_features():
        macro = make_synthetic_macro()
        spx = make_synthetic_spx()
        frames = []
        for i, t in enumerate(tickers):
            prices = make_synthetic_ohlcv(t, seed=i + 1)
            tdf = tech.compute(prices)
            tg = tgt.compute(prices, horizons=(1, 5, 20))
            df = tdf.merge(tg, on="date", how="left")
            df["ticker"] = t
            df["asset_class"] = "us"
            df = df.merge(macro, on="date", how="left")
            frames.append(df)
        feat = pd.concat(frames, ignore_index=True).dropna(subset=["fwd_dir_5d"])
        feat.attrs["macro"] = macro
        feat.attrs["spx"] = spx
        print(f"features: {len(feat)} rows, {feat.shape[1]} cols, "
              f"tickers={feat['ticker'].nunique()}")
        return feat

    features = step("build features", build_features)

    # 2) Walk-forward XGBoost
    def fit_model():
        wf = walkforward_fit(features, target="fwd_dir_5d",
                             n_folds=4, min_train=400)
        preds = wf.predictions_frame()
        print(f"predictions: {len(preds)} rows, cols={list(preds.columns)}")
        assert "ticker" in preds.columns, "ticker column missing from predictions_frame!"
        return preds

    preds = step("walk-forward XGBoost", fit_model)

    # 3) Regime HMM
    def fit_regime():
        macro = features.attrs["macro"]
        spx = features.attrs["spx"]
        rfeat = build_regime_features(macro, spx=spx)
        if len(rfeat) < 100:
            print(f"WARN: only {len(rfeat)} regime rows, skipping HMM")
            return None
        model = hmm_fit(rfeat, n_states=4)
        labels = model.predict(rfeat)
        proba = model.predict_proba(rfeat)
        print(f"regime: {len(labels)} labels, unique={set(labels)}, "
              f"proba shape={proba.shape}")
        return pd.DataFrame({"date": pd.to_datetime(rfeat["date"]),
                             "regime": labels})

    regime_df = step("HMM regime", fit_regime)

    # 4) Rubric + blend + gate
    def score_blend_gate():
        # Merge predictions back into features by (date, ticker)
        preds["date"] = pd.to_datetime(preds["date"])
        feat = features.copy()
        feat["date"] = pd.to_datetime(feat["date"])
        merged = feat.merge(preds[["date", "ticker", "proba_up"]],
                            on=["date", "ticker"], how="inner")
        if regime_df is not None:
            rmap = pd.Series(regime_df["regime"].values,
                             index=pd.to_datetime(regime_df["date"]))
            merged["regime"] = pd.to_datetime(merged["date"]).map(rmap).fillna("chop")
        else:
            merged["regime"] = "chop"
        # Use a tiny sample for speed since iterrows is slow
        sample = merged.sample(min(500, len(merged)), random_state=0)
        scored = rubric_score(sample, regimes=sample["regime"])
        blended = blend(scored, cfg=BlendConfig())
        gated = gate_apply(blended, cfg=GateConfig(min_conviction=55.0))
        n_pass = int(gated["passes_gate"].sum())
        print(f"rubric+blend+gate: scored={len(scored)} pass={n_pass}")
        return gated

    signals = step("rubric + blend + gate", score_blend_gate)

    # 5) Backtest
    def backtest():
        # Use a smaller universe for speed
        small = features[features["ticker"].isin(tickers[:3])].copy()
        rep = run_backtest(small, n_folds=3, min_train=300,
                           run_shuffled_baseline=False)
        print(f"backtest summary keys: {list(rep.summary.keys())}")
        print(f"  hit_rate={rep.summary.get('hit_rate'):.3f}, "
              f"auc={rep.summary.get('auc')}")
        return rep

    bt = step("backtest", backtest)

    # 6) Persist
    def persist():
        from omni.orchestrator.persist import persist_run
        res = persist_run(
            predictions=preds.head(200),
            signals=signals.head(200),
            regime_labels=regime_df,
            backtest_summary=bt.summary,
            fold_metrics=bt.fold_metrics,
            features_latest=features.tail(50),
            run_date="2024-12-31",
        )
        print(f"persist: {res}")
        return res

    pr = step("persist", persist)

    # 7) Dashboard cache load
    def dashboard_load():
        from omni.dashboard import cache as dc
        preds_loaded = dc.load_predictions()
        sigs_loaded = dc.load_signals()
        regime_loaded = dc.load_regime()
        bt_loaded = dc.load_backtest_summary()
        print(f"dashboard load: predictions={len(preds_loaded)}, "
              f"signals={len(sigs_loaded)}, regime={len(regime_loaded)}, "
              f"backtest_keys={list(bt_loaded.keys()) if bt_loaded else None}")
        return True

    step("dashboard cache load", dashboard_load)

    print("\n=== E2E synthetic test PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
