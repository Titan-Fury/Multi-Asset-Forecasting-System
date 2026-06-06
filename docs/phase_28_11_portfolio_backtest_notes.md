# Phase 28.11 — Tier-0 cost-aware portfolio backtest (Sharpe/maxDD/turnover)

## What & why

First item of `docs/prediction_efficiency_research.md`: the harness could only
score predictions point-wise (AUC/hit-rate/decile), never "would trading it make
money after costs?". `omni/backtest/portfolio.py::simulate_portfolio` adds the
desk-level view and is wired into `run_backtest` as `summary["portfolio"]`.

Model (documented, assumption-light): each gated signal is a `horizon`-day
trade returning `direction * fwd_ret_5d` minus a round-trip cost (default 15bps);
the trade's return is amortized across its holding days; the book is equal-weight
across whatever is open each day. Reports annualized return, Sharpe, Sortino,
max drawdown, turnover, hit-rate, profit factor, avg trade return.

## Result (india tb_label_5d, 205,283 rows, 15bps cost)

| metric | base-rate OFF (0.5 neutral) | base-rate ON (0.33) |
| ------ | --------------------------: | ------------------: |
| trades | 75,925 (216 long / 75,709 short) | 2,268 (1,664 long / 604 short) |
| ann return | -26.4% | +18.6% |
| Sharpe | **-4.02** | **+1.71** |
| Sortino | -5.92 | +2.29 |
| max drawdown | -98% | -37% |
| turnover | 0.20 | 0.20 |
| hit-rate | 0.470 | 0.515 |
| profit factor | 0.767 | 0.793 |
| **avg trade ret (post-cost)** | **-0.0059** | **-0.0079** |
| avg open positions | 102 | 11 |

## Honest read (the tool earning its keep)

- **OFF is catastrophic** (Sharpe -4.0, -98% drawdown) — independent
  confirmation of the 28.9/28.10 short-bias bug: a 99.7%-short book on a
  15-year rising index.
- **ON *looks* great (+1.71 Sharpe) but is NOT alpha.** The tell is in the same
  table: `avg_trade_ret` is still **negative** (-0.0079, post-cost), and
  `profit_factor < 1`. The positive portfolio return comes from (a) the long
  tilt (1,664/2,268) capturing the index's secular drift = **market beta**, and
  (b) **concentration** — with only ~11 open positions (sometimes 1-2 on sparse
  days), an equal-weight-all-open book over-weights a handful of lucky low-count
  days. Strip beta and concentration and the per-trade edge is negative. This is
  exactly the self-deception a point-wise AUC/hit-rate view hides and a
  cost-aware Sharpe backtest exposes.

## Takeaways / next

1. The Tier-0 measurement layer is in place; every future change can now be
   judged on post-cost Sharpe, not just AUC.
2. The headline number remains: **no standalone alpha** on liquid NIFTY
   large-caps from absolute direction (consistent with 28.8's ~0 decile spread).
   The positive ON Sharpe is beta + concentration, not skill.
3. Robustness refinements deferred (report trade-weighted return alongside
   day-weighted; market/beta-neutralize the book; min-open-positions floor;
   conviction weighting). The decisive lever is still Tier-1: switch the target
   to cross-sectional / market-neutral relative return so the book trades
   *alpha* instead of dressed-up beta.
