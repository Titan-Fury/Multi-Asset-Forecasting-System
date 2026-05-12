"""Runtime pair-signal scorer.

Computes a pair / hedging suggestion at prediction time WITHOUT requiring
retraining. Uses:
  - existing 5d universal-model probability for the stock and the benchmark
  - 60d rolling beta + 20d residual vol from pair features
  - alpha = move_5d_pct(stock) - move_5d_pct(bench)
  - pair_sharpe ~= alpha / residual_vol_20d_pct

Returns a dict; never raises (returns {} on any failure).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from omni.features.pair import benchmark_for, compute_pair_features
from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log


def _bench_asset_class(bench_ticker: str) -> str:
    if bench_ticker.endswith("-USD"):
        return "crypto"
    if bench_ticker.endswith(".NS") or bench_ticker.startswith("^NS") or bench_ticker.startswith("^CNX"):
        return "india"
    if bench_ticker.startswith("^") or bench_ticker in ("DX-Y.NYB",) or "=" in bench_ticker:
        return "us"
    return "us"


@cached(ttl=TTL.news, key="bench_card:{bench_ticker}:{bench_asset_class}")
def _bench_card_cached(bench_ticker: str, bench_asset_class: str) -> dict | None:
    """Lightweight cached prediction for the benchmark, used by many tickers."""
    try:
        # Lazy import to avoid circular dep on predict.py
        from omni.ondemand.predict import predict_ticker
        c = predict_ticker(bench_ticker, bench_asset_class, display_name=bench_ticker)
        if c is None:
            return None
        return c.to_dict()
    except Exception as exc:  # noqa: BLE001
        log.warning(f"bench predict failed {bench_asset_class}/{bench_ticker}: {exc}")
        return None


@cached(ttl=TTL.news, key="bench_prices:{bench_ticker}:{bench_asset_class}")
def _cached_bench_prices(bench_ticker: str, bench_asset_class: str) -> "pd.DataFrame | None":
    """Phase 24.16 — restored. Earlier reference at line 85 called this helper
    but it had been removed in a refactor, raising NameError per request.
    Cached benchmark price fetch so pair_signal doesn't refetch yfinance for
    every ticker in a screen sweep that shares the same benchmark.
    """
    try:
        from omni.ondemand.lookup import lookup_ticker
        b = lookup_ticker(bench_ticker, bench_asset_class, display_name=bench_ticker)
        return b.prices if b is not None else None
    except Exception as exc:  # noqa: BLE001
        log.warning(f"bench prices fetch failed {bench_asset_class}/{bench_ticker}: {exc}")
        return None


def compute_pair_signal(
    stock_card: dict,
    *,
    stock_prices: pd.DataFrame | None = None,
    bench_prices: pd.DataFrame | None = None,
) -> dict:
    """Build the pair-signal block for a given stock prediction card.

    `stock_prices` and `bench_prices` are optional; when omitted we'll fetch
    the benchmark prices via the lake / yfinance lookup.
    Returns {} on any failure or when benchmark == self.
    """
    try:
        ticker = stock_card.get("ticker")
        asset_class = stock_card.get("asset_class")
        if not ticker or not asset_class:
            return {}
        bench = benchmark_for(ticker, asset_class)
        if not bench or bench == ticker:
            return {}

        bench_ac = _bench_asset_class(bench)

        # Get benchmark prediction (cached)
        b_card = _bench_card_cached(bench, bench_ac)
        if not b_card:
            return {}

        # Pair features need both price series
        if stock_prices is None or bench_prices is None:
            try:
                from omni.ondemand.lookup import lookup_ticker
                if stock_prices is None:
                    sb = lookup_ticker(ticker, asset_class, display_name=ticker)
                    stock_prices = sb.prices if sb else None
                if bench_prices is None:
                    # Phase 24 — cached fetch: same benchmark requested
                    # 100x in a screen sweep should hit memory, not yfinance.
                    bench_prices = _cached_bench_prices(bench, bench_ac)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"pair price lookup failed: {exc}")
                return {}

        if stock_prices is None or bench_prices is None or stock_prices.empty or bench_prices.empty:
            return {}

        pf = compute_pair_features(stock_prices, bench_prices)
        if pf.empty:
            return {}
        last = pf.iloc[-1]

        s_alpha = float(stock_card.get("move_1w_pct", 0.0))
        b_alpha = float(b_card.get("move_1w_pct", 0.0))
        predicted_alpha_pct = s_alpha - b_alpha
        residual_vol = float(last.get("residual_vol_20d_pct", 0.0))
        if residual_vol > 0:
            pair_sharpe = predicted_alpha_pct / residual_vol
        else:
            pair_sharpe = 0.0

        beta = float(last.get("beta_60d", 1.0)) if pd.notna(last.get("beta_60d")) else 1.0
        corr = float(last.get("correlation_60d", 0.0)) if pd.notna(last.get("correlation_60d")) else 0.0
        resid_z = float(last.get("residual_z_20d", 0.0)) if pd.notna(last.get("residual_z_20d")) else 0.0

        # Decide suggestion
        suggestion = "no edge"
        if abs(predicted_alpha_pct) >= 0.5 and abs(pair_sharpe) >= 0.3:
            if predicted_alpha_pct > 0:
                suggestion = f"long {ticker} / short {bench}"
            else:
                suggestion = f"short {ticker} / long {bench}"

        return {
            "benchmark": bench,
            "benchmark_asset_class": bench_ac,
            "beta_60d": round(beta, 3),
            "correlation_60d": round(corr, 3),
            "residual_z_20d": round(resid_z, 3),
            "residual_vol_20d_pct": round(residual_vol, 3),
            "stock_move_5d_pct": round(s_alpha, 3),
            "bench_move_5d_pct": round(b_alpha, 3),
            "predicted_alpha_pct": round(predicted_alpha_pct, 3),
            "pair_sharpe": round(pair_sharpe, 3),
            "suggestion": suggestion,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning(f"compute_pair_signal failed: {exc}")
        return {}
