"""Portfolio-level, cost-aware backtest metrics.

The existing `metrics.py` scores predictions point-wise (AUC, hit-rate, decile
returns). That answers "is the ranking any good?" but NOT "would trading this
make money after costs?". This module turns signals into a long/short book and
reports the metrics a desk actually cares about: annualized return, **Sharpe**,
Sortino, max drawdown, turnover, profit factor.

Key methodology note (why this is non-trivial): we only have `horizon`-day
forward returns (`fwd_ret_5d`), not daily mark-to-market. Naively amortizing a
5-day trade into 5 equal daily legs fabricates an artificially smooth daily P&L
series, which collapses the volatility and inflates Sharpe to absurd values
(an early version produced Sharpe ~58, barely below a perfect-foresight score).
The honest fix used here: aggregate into **non-overlapping `horizon`-day
periods** and annualize by `sqrt(periods_per_year / horizon)`. Each period is a
distinct, non-overlapping holding window, so the period returns are not
artificially autocorrelated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_KEYS = ["n_trades", "n_long", "n_short", "ann_return", "ann_vol", "sharpe",
         "sortino", "max_drawdown", "turnover", "hit_rate", "profit_factor",
         "avg_trade_ret", "avg_open_positions", "n_periods"]


def _empty() -> dict:
    return {k: 0.0 for k in _KEYS}


def _perf_stats(period_returns: np.ndarray, periods_per_year: float) -> dict:
    """Annualized return / Sharpe / Sortino / max-DD from NON-overlapping
    period returns. `periods_per_year` is already horizon-adjusted."""
    r = np.asarray(period_returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n == 0:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "sortino": 0.0, "max_drawdown": 0.0, "n_periods": 0}
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1)) if n > 1 else 0.0
    ann_return = mu * periods_per_year
    ann_vol = sd * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0
    downside = r[r < 0]
    dvol = float(np.std(downside, ddof=1)) * np.sqrt(periods_per_year) if downside.size > 1 else 0.0
    sortino = ann_return / dvol if dvol > 1e-12 else 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0))
    return {"ann_return": round(ann_return, 4), "ann_vol": round(ann_vol, 4),
            "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
            "max_drawdown": round(max_dd, 4), "n_periods": int(n)}


def simulate_portfolio(sig: pd.DataFrame, *, horizon: int = 5,
                       cost_bps: float = 15.0,
                       periods_per_year: int = 252) -> dict:
    """Equal-weight directional long/short book from horizon-return signals.

    Signals are bucketed into non-overlapping `horizon`-day blocks; each block's
    return is the equal-weight mean net (post-cost) trade return of signals
    opened in that block. Trade-level stats (hit-rate, profit factor) are over
    all individual trades.
    """
    if sig is None or sig.empty:
        return _empty()
    s = sig.copy()
    if "passes_gate" in s.columns:
        s = s[s["passes_gate"].astype(bool)]
    s = s.dropna(subset=["fwd_ret_5d", "direction", "date"])
    s = s[s["direction"].astype(int) != 0]
    if s.empty:
        return _empty()

    s["date"] = pd.to_datetime(s["date"])
    cost = float(cost_bps) / 1e4
    dir_arr = s["direction"].astype(float).to_numpy()
    fwd = pd.to_numeric(s["fwd_ret_5d"], errors="coerce").to_numpy()
    gross_trade = dir_arr * fwd
    net_trade = gross_trade - cost                  # round-trip cost once

    # Non-overlapping horizon-day blocks over the trading calendar.
    dates = np.sort(s["date"].unique())
    didx = {d: i for i, d in enumerate(dates)}
    block = (s["date"].map(didx).to_numpy() // horizon)
    tmp = pd.DataFrame({"block": block, "net": net_trade})
    block_ret = tmp.groupby("block")["net"].mean().to_numpy()

    ppy = periods_per_year / horizon
    stats = _perf_stats(block_ret, ppy)

    wins = gross_trade[gross_trade > 0].sum()
    losses = gross_trade[gross_trade < 0].sum()
    if losses < 0:
        profit_factor = round(float(wins / abs(losses)), 3)
    elif wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    # avg concurrent open positions (for context): trades * horizon / n_days.
    avg_open = round(len(s) * horizon / max(1, len(dates)), 1)

    out = _empty()
    out.update(stats)
    out.update({
        "n_trades": int(len(s)),
        "n_long": int((dir_arr > 0).sum()),
        "n_short": int((dir_arr < 0).sum()),
        "turnover": round(len(s) / max(1, len(dates)), 3),
        "hit_rate": round(float(np.mean(gross_trade > 0)), 4),
        "profit_factor": profit_factor,
        "avg_trade_ret": round(float(np.mean(net_trade)), 5),
        "avg_open_positions": avg_open,
    })
    return out


def simulate_cross_sectional_ls(df: pd.DataFrame, *, score_col: str = "proba_up",
                                ret_col: str = "fwd_ret_5d",
                                top_q: float = 0.1, horizon: int = 5,
                                cost_bps: float = 15.0,
                                periods_per_year: int = 252,
                                min_names: int = 6) -> dict:
    """Market-neutral cross-sectional long/short book from a ranking score.

    Each REBALANCE (every `horizon` trading days, non-overlapping) rank names by
    `score_col`; long the top `top_q` fraction, short the bottom `top_q`
    fraction (equal weight, dollar-neutral). Period return =
    `mean(top fwd_ret) - mean(bottom fwd_ret)` less a round-trip cost on both
    legs. Market-neutral by construction, so far less sensitive to
    survivorship-driven beta than the directional book. Correct way to score a
    RANKING signal.
    """
    if df is None or df.empty:
        return _empty()
    s = df.dropna(subset=["date", score_col, ret_col]).copy()
    if s.empty:
        return _empty()
    s["date"] = pd.to_datetime(s["date"])

    all_dates = np.sort(s["date"].unique())
    rebal = set(all_dates[::horizon])           # non-overlapping rebalances
    cost = float(cost_bps) / 1e4

    periods, ks = [], []
    for d, g in s.groupby("date"):
        if d not in rebal or len(g) < min_names:
            continue
        k = max(1, int(round(len(g) * top_q)))
        gs = g.sort_values(score_col)
        bottom = float(gs.iloc[:k][ret_col].astype(float).mean())
        top = float(gs.iloc[-k:][ret_col].astype(float).mean())
        periods.append((top - bottom) - 2.0 * cost)   # both legs round-trip
        ks.append(k)
    if not periods:
        return _empty()
    period_ret = np.array(periods, dtype=float)

    ppy = periods_per_year / horizon
    stats = _perf_stats(period_ret, ppy)
    gross = period_ret + 2.0 * cost
    out = _empty()
    out.update(stats)
    out.update({
        "n_trades": int(len(period_ret)),          # rebalances
        "hit_rate": round(float(np.mean(gross > 0)), 4),
        "avg_trade_ret": round(float(np.mean(period_ret)), 5),
        "avg_open_positions": round(float(np.mean(ks)), 1),  # names per leg
        "turnover": 1.0,
    })
    return out
