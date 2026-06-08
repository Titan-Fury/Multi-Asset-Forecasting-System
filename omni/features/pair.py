"""Pair / cross-asset features.

Computes rolling beta, alpha, residual, correlation between a stock and a
benchmark series. All inputs are ``date`` indexed price frames with
``adj_close`` columns. Outputs are numeric Series aligned to the stock.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from omni.utils.logger import log

_BENCH_CFG: dict | None = None


def _load_bench_cfg() -> dict:
    global _BENCH_CFG
    if _BENCH_CFG is None:
        cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "benchmarks.yaml"
        if not cfg_path.exists():
            log.warning(f"benchmarks.yaml missing at {cfg_path}; pair signals disabled")
            _BENCH_CFG = {}
        else:
            with cfg_path.open("r", encoding="utf-8") as f:
                _BENCH_CFG = yaml.safe_load(f) or {}
    return _BENCH_CFG


def benchmark_for(ticker: str, asset_class: str) -> str | None:
    """Return the benchmark ticker for (ticker, asset_class), or None."""
    cfg = _load_bench_cfg().get(asset_class) or {}
    by_ticker = cfg.get("tickers") or {}
    if ticker in by_ticker:
        return str(by_ticker[ticker])
    default = cfg.get("default")
    return str(default) if default else None


def _align(stock: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    s = stock[["date", "adj_close"]].rename(columns={"adj_close": "s_close"})
    b = bench[["date", "adj_close"]].rename(columns={"adj_close": "b_close"})
    s["date"] = pd.to_datetime(s["date"])
    b["date"] = pd.to_datetime(b["date"])
    df = pd.merge(s, b, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["s_ret"] = df["s_close"].pct_change()
    df["b_ret"] = df["b_close"].pct_change()
    return df


def compute_pair_features(
    stock_prices: pd.DataFrame,
    bench_prices: pd.DataFrame,
    *,
    beta_window: int = 60,
    z_window: int = 20,
) -> pd.DataFrame:
    """Compute rolling pair features. Returns DataFrame indexed by date with:

    [beta_60d, alpha_60d, residual_20d, residual_z_20d, correlation_60d,
     residual_vol_20d_pct]
    """
    if stock_prices is None or stock_prices.empty or bench_prices is None or bench_prices.empty:
        return pd.DataFrame()
    df = _align(stock_prices, bench_prices)
    if len(df) < beta_window + 5:
        return pd.DataFrame()

    cov = df["s_ret"].rolling(beta_window).cov(df["b_ret"])
    var = df["b_ret"].rolling(beta_window).var()
    df["beta_60d"] = cov / var.replace(0, np.nan)
    df["alpha_60d"] = (
        df["s_ret"].rolling(beta_window).mean()
        - df["beta_60d"] * df["b_ret"].rolling(beta_window).mean()
    )
    df["residual"] = df["s_ret"] - df["beta_60d"].fillna(1.0) * df["b_ret"]
    rmean = df["residual"].rolling(z_window).mean()
    rstd = df["residual"].rolling(z_window).std()
    df["residual_z_20d"] = (df["residual"] - rmean) / rstd.replace(0, np.nan)
    df["correlation_60d"] = df["s_ret"].rolling(beta_window).corr(df["b_ret"])
    df["residual_vol_20d_pct"] = (
        df["residual"].rolling(z_window).std().fillna(0) * 100.0
    )

    out = df[[
        "date", "beta_60d", "alpha_60d",
        "residual_z_20d", "correlation_60d", "residual_vol_20d_pct",
    ]].copy()
    return out
