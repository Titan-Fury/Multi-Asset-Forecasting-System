"""Cached data loaders for the dashboard.

Streamlit's @st.cache_data is used so the dashboard never refits a model and
never re-reads parquet from disk on every interaction. The cache key is the
parquet mtime so an updated daily run will invalidate.

All loaders here are READ-ONLY. The dashboard must never trigger model fit /
walk-forward / rubric scoring -- those happen in the orchestrator (Phase 9).
What the dashboard reads:

  features tier  -- latest snapshot of feature frame (for sidebar filters)
  predictions    -- latest WalkForwardResult.predictions_frame() persisted as parquet
  signals        -- latest gated signals frame
  regime         -- latest regime label series
  backtest       -- latest BacktestReport summary (json) + fold_metrics

Falls back to synthetic demo data when files are missing so the dashboard
runs out-of-the-box before the orchestrator has been wired up.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:  # for unit/import tests
    class _Stub:
        def cache_data(self, *a, **kw):
            def deco(fn): return fn
            return deco
    st = _Stub()


# Locations (kept as relative roots — wire to omni.utils.paths later if exposed)
_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "data"
PRED_PATH = _DATA / "predictions" / "latest.parquet"
SIGNAL_PATH = _DATA / "signals" / "latest.parquet"
REGIME_PATH = _DATA / "regime" / "latest.parquet"
BACKTEST_PATH = _DATA / "backtest" / "summary.json"
FEATURE_PATH = _DATA / "features" / "latest.parquet"


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _looks_like_synth_test(df: pd.DataFrame) -> bool:
    if df is None or df.empty or "ticker" not in df.columns:
        return False
    tickers = {str(t).upper() for t in df["ticker"].dropna().unique().tolist()}
    return bool(tickers) and tickers.issubset({"AAA", "BBB", "CCC", "DDD", "EEE"})


@st.cache_data(show_spinner=False)
def load_predictions(_mt: float = 0.0) -> pd.DataFrame:
    if PRED_PATH.exists():
        df = pd.read_parquet(PRED_PATH)
        if _looks_like_synth_test(df):
            return pd.DataFrame()
        return df
    return _demo_predictions()


@st.cache_data(show_spinner=False)
def load_signals(_mt: float = 0.0) -> pd.DataFrame:
    if SIGNAL_PATH.exists():
        df = pd.read_parquet(SIGNAL_PATH)
        if _looks_like_synth_test(df):
            return pd.DataFrame()
        return df
    return _demo_signals()


@st.cache_data(show_spinner=False)
def load_regime(_mt: float = 0.0) -> pd.DataFrame:
    if REGIME_PATH.exists():
        return pd.read_parquet(REGIME_PATH)
    return _demo_regime()


@st.cache_data(show_spinner=False)
def load_backtest_summary(_mt: float = 0.0) -> dict:
    import json
    if BACKTEST_PATH.exists():
        return json.loads(BACKTEST_PATH.read_text())
    return _demo_backtest()


def reload_keys() -> dict:
    """Return mtime tuple to use as cache-busting param if the caller wants
    fresh data after a daily run, e.g. load_predictions(reload_keys()['pred'])."""
    return {
        "pred": _mtime(PRED_PATH),
        "signals": _mtime(SIGNAL_PATH),
        "regime": _mtime(REGIME_PATH),
        "backtest": _mtime(BACKTEST_PATH),
        "features": _mtime(FEATURE_PATH),
    }


# ---------- demo fallback data ----------

_UNIVERSE = (
    # ---- Top 50 Indian stocks (NSE) — primary focus ----
    [(t, "india") for t in [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "LT",
        "KOTAKBANK", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
        "TITAN", "BAJFINANCE", "WIPRO", "HCLTECH", "ULTRACEMCO",
        "ADANIENT", "ADANIPORTS", "ONGC", "NTPC", "POWERGRID",
        "COALINDIA", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO",
        "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "DIVISLAB", "DRREDDY",
        "CIPLA", "APOLLOHOSP", "BRITANNIA", "NESTLEIND", "BAJAJ-AUTO",
        "HEROMOTOCO", "EICHERMOT", "GRASIM", "SHRIRAMFIN", "INDUSINDBK",
        "BPCL", "TECHM", "GRSE", "MAZDOCK", "HAL",
    ]] +
    # ---- US large-cap (smaller weight) ----
    [(t, "us") for t in ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]] +
    # ---- Crypto top 5 ----
    [(t, "crypto") for t in ["BTC", "ETH", "SOL", "BNB", "XRP"]]
)


def _demo_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=120)
    rows = []
    for d in dates:
        for tk, ac in _UNIVERSE:
            rows.append({
                "date": d, "ticker": tk, "asset_class": ac,
                "proba_up": float(np.clip(rng.normal(0.5, 0.1), 0.05, 0.95)),
                "y_true": int(rng.integers(0, 2)),
                "fwd_ret_5d": float(rng.normal(0.001, 0.02)),
                "fold": int(rng.integers(0, 5)),
            })
    return pd.DataFrame(rows)


def _demo_signals() -> pd.DataFrame:
    preds = _demo_predictions()
    last = preds[preds["date"] == preds["date"].max()].copy()
    rng = np.random.default_rng(11)
    last["rubric_total"] = rng.uniform(20, 95, size=len(last))
    # Widen demo proba so the UI shows realistic high-conviction signals.
    # Without this, model AUC ~0.51 keeps every row under the 70-gate and
    # the heatmap / cosmos / "Top signals" tabs look empty in demo mode.
    spread = rng.normal(0.0, 0.30, size=len(last))
    last["blended_p"] = (0.5 + spread).clip(0.05, 0.95)
    last["conviction"] = (last["blended_p"].sub(0.5).abs() * 200).clip(0, 100)
    last["direction"] = np.where(last["blended_p"] > 0.52, 1,
                          np.where(last["blended_p"] < 0.48, -1, 0))
    last["regime"] = rng.choice(["bull", "chop", "bear", "crisis"],
                                 size=len(last), p=[0.4, 0.4, 0.15, 0.05])
    last["passes_gate"] = (last["conviction"] >= 70) & (last["direction"] != 0)
    last["reasons"] = np.where(last["passes_gate"], "ok", "conviction<70")
    # rubric breakdown for explainability
    for pillar in ("technical", "fundamental", "sentiment", "macro"):
        last[f"pillar_{pillar}"] = rng.uniform(20, 90, size=len(last))
    return last.reset_index(drop=True)


def _demo_regime() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(3)
    labels = rng.choice(["bull", "chop", "bear", "crisis"], size=len(dates),
                        p=[0.45, 0.35, 0.15, 0.05])
    return pd.DataFrame({"date": dates, "regime": labels})


def _demo_backtest() -> dict:
    return {
        "n": 1500, "hit_rate": 0.563, "auc": 0.582, "brier": 0.244,
        "log_loss": 0.681,
        "top_decile": {"precision": 0.682, "n": 150, "mean_fwd_ret": 0.011},
        "bottom_decile": {"precision_short": 0.654, "n": 150, "mean_fwd_ret": -0.009},
        "gated": {"n_emitted": 84, "emit_rate": 0.056,
                  "directional_hit_rate": 0.690, "mean_signed_fwd_ret": 0.012},
        "shuffled_baseline": {"hit_rate": 0.503, "auc": 0.504, "brier": 0.250, "n": 1500},
        "edge": {"edge_hit_rate_pp": 6.0, "edge_auc_pp": 7.8, "passes_min_edge": True},
    }
