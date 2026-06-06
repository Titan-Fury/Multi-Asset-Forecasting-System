"""Portfolio-level, cost-aware backtest metrics.

The existing `metrics.py` scores predictions point-wise (AUC, hit-rate, decile
returns). That answers "is the ranking any good?" but NOT "would trading this
make money after costs?". This module turns gated signals into a daily
long/short book and reports the metrics a desk actually cares about: annualized
return, **Sharpe**, Sortino, max drawdown, turnover, profit factor, and the
edge over a shuffled-direction baseline.

Model & assumptions (documented on purpose — this is a horizon-return backtest,
not a tick simulator):

  * Each emitted signal is a discrete `horizon`-day trade. Its gross return is
    `direction * fwd_ret_5d`; a round-trip transaction cost (`cost_bps`) is
    charged once per trade.
  * A trade's return is amortized evenly across its `horizon` holding days
    (linear accrual) so overlapping cohorts don't double-count capital.
  * The book is **equal-weight across whatever is open** on each day (fully
    invested, gross ~1), so the daily portfolio return is the mean of the active
    daily legs. This is the standard, assumption-light way to get a Sharpe from
    horizon returns; swap in conviction weighting later if desired.

Inputs: a frame with `date`, `direction` (-1/0/+1), `fwd_ret_5d`, and optionally
`passes_gate`. Only gated, non-zero-direction rows are traded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _empty() -> dict:
    keys = ["n_trades", "n_long", "n_short", "ann_return", "ann_vol", "sharpe",
            "sortino", "max_drawdown", "turnover", "hit_rate", "profit_factor",
            "avg_trade_ret", "avg_open_positions", "n_days"]
    return {k: 0.0 for k in keys}


def simulate_portfolio(sig: pd.DataFrame, *, horizon: int = 5,
                       cost_bps: float = 15.0,
                       periods_per_year: int = 252) -> dict:
    """Daily equal-weight long/short book from horizon-return signals.

    Returns a dict of portfolio metrics plus the daily return series under
    `_daily_returns` (np.ndarray) for plotting an equity curve.
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
    gross_trade = dir_arr * fwd                  # gross per-trade return
    net_trade = gross_trade - cost               # round-trip cost charged once
    leg = net_trade / horizon                    # amortized per holding day

    # Trading calendar = sorted unique signal dates.
    dates = np.sort(s["date"].unique())
    T = len(dates)
    didx = {d: i for i, d in enumerate(dates)}
    start = s["date"].map(didx).to_numpy()

    # Difference-array accumulation: a trade opened at day i is held on days
    # [i+1, i+horizon]. Add at open+1, remove at open+horizon+1, then cumsum.
    ret_diff = np.zeros(T + 1)
    cnt_diff = np.zeros(T + 1)
    a = np.minimum(start + 1, T)
    b = np.minimum(start + 1 + horizon, T)
    np.add.at(ret_diff, a, leg)
    np.add.at(ret_diff, b, -leg)
    np.add.at(cnt_diff, a, 1.0)
    np.add.at(cnt_diff, b, -1.0)
    daily_ret_sum = np.cumsum(ret_diff)[:T]
    daily_cnt = np.cumsum(cnt_diff)[:T]
    with np.errstate(invalid="ignore", divide="ignore"):
        port = np.where(daily_cnt > 0, daily_ret_sum / daily_cnt, 0.0)

    mu, sd = float(np.mean(port)), float(np.std(port, ddof=1)) if T > 1 else 0.0
    ann_return = mu * periods_per_year
    ann_vol = sd * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0
    downside = port[port < 0]
    dvol = float(np.std(downside, ddof=1)) * np.sqrt(periods_per_year) if downside.size > 1 else 0.0
    sortino = ann_return / dvol if dvol > 1e-12 else 0.0

    equity = np.cumprod(1.0 + port)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity / peak - 1.0)) if T else 0.0

    # Turnover: positions opened per day / avg book size (fraction of book replaced daily).
    avg_open = float(np.mean(daily_cnt[daily_cnt > 0])) if np.any(daily_cnt > 0) else 0.0
    opens_per_day = len(s) / max(1, T)
    turnover = opens_per_day / avg_open if avg_open > 0 else 0.0

    wins = gross_trade[gross_trade > 0].sum()
    losses = gross_trade[gross_trade < 0].sum()
    if losses < 0:
        profit_factor = float(wins / abs(losses))
    elif wins > 0:
        profit_factor = float("inf")   # all winners, no losers
    else:
        profit_factor = 0.0

    out = _empty()
    out.update({
        "n_trades": int(len(s)),
        "n_long": int((dir_arr > 0).sum()),
        "n_short": int((dir_arr < 0).sum()),
        "ann_return": round(ann_return, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(max_dd, 4),
        "turnover": round(turnover, 3),
        "hit_rate": round(float(np.mean(gross_trade > 0)), 4),
        "profit_factor": round(profit_factor, 3),
        "avg_trade_ret": round(float(np.mean(net_trade)), 5),
        "avg_open_positions": round(avg_open, 1),
        "n_days": int(T),
    })
    out["_daily_returns"] = port
    return out
