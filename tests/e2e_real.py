"""End-to-end real-data smoke test using yfinance.

Fetches AAPL daily bars over 5 years and runs the in-process pipeline:
  technical features -> targets -> walk-forward XGBoost
  -> rubric + blend + gate (a sample) -> backtest

Skips macro/regime since regime needs SPX+VIX joined and is well-tested by
the synthetic harness already; the point here is to confirm the real OHLCV
schema flows through cleanly.

Run:
    .venv\\Scripts\\python.exe tests\\e2e_real.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def fetch_aapl(years: int = 5) -> pd.DataFrame:
    import yfinance as yf
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    df = yf.download("AAPL", start=start, end=end, auto_adjust=False,
                     progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned no data for AAPL")
    # yfinance returns multi-index columns when single ticker on newer versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # Normalize to omni schema
    rename = {"adj_close": "adj_close", "date": "date"}
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "adj_close": df.get("adj_close", df["close"]).astype(float),
        "volume": df["volume"].astype(float),
    })
    return out


def main() -> int:
    from omni.features import technical as tech
    from omni.features import targets as tgt
    from omni.models.xgboost_5d import walkforward_fit
    from omni.rubric.scorer import score_frame as rubric_score
    from omni.ensemble.combine import blend, BlendConfig
    from omni.ensemble.gate import apply as gate_apply, GateConfig
    from omni.backtest.runner import run_backtest

    print("--- fetching AAPL via yfinance ---")
    try:
        prices = fetch_aapl(years=5)
    except Exception as e:
        print(f"SKIP: yfinance unavailable or rate-limited: {e}")
        traceback.print_exc()
        return 0  # don't fail CI on a transient network issue
    print(f"AAPL rows: {len(prices)}, range: {prices['date'].min().date()} -> "
          f"{prices['date'].max().date()}")

    print("\n--- technical + targets ---")
    tdf = tech.compute(prices)
    tg = tgt.compute(prices, horizons=(1, 5, 20))
    feat = tdf.merge(tg, on="date", how="left")
    feat["ticker"] = "AAPL"
    feat["asset_class"] = "us"
    feat = feat.dropna(subset=["fwd_dir_5d"])
    print(f"feature frame: {len(feat)} rows, {feat.shape[1]} cols")

    print("\n--- walk-forward XGBoost ---")
    wf = walkforward_fit(feat, target="fwd_dir_5d", n_folds=4, min_train=400)
    preds = wf.predictions_frame()
    print(f"predictions: {len(preds)} rows, mean_proba_up={preds['proba_up'].mean():.3f}")
    assert "ticker" in preds.columns

    print("\n--- rubric + blend + gate (last 200 rows) ---")
    feat["date"] = pd.to_datetime(feat["date"])
    preds["date"] = pd.to_datetime(preds["date"])
    merged = feat.merge(preds[["date", "ticker", "proba_up"]],
                        on=["date", "ticker"], how="inner")
    sample = merged.tail(200).copy()
    sample["regime"] = "chop"
    scored = rubric_score(sample, regimes=sample["regime"])
    blended = blend(scored, cfg=BlendConfig())
    gated = gate_apply(blended, cfg=GateConfig(min_conviction=55))
    print(f"sample rubric+blend+gate: scored={len(scored)} "
          f"pass={int(gated['passes_gate'].sum())}")

    print("\n--- backtest ---")
    rep = run_backtest(feat, n_folds=3, min_train=400, run_shuffled_baseline=False)
    print(f"backtest hit_rate={rep.summary.get('hit_rate'):.3f} "
          f"auc={rep.summary.get('auc')}")
    print("\n=== E2E real-data test PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
