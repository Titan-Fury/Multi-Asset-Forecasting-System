# Backtest leakage audit (the four classic traps)

Audit of Omni_Stocks against the four canonical backtest-leakage failure modes.
Verdict first, evidence below.

| # | Trap | Verdict | Severity |
| - | ---- | ------- | -------- |
| 1 | Standardization leakage (global mean/std) | Clean (one negligible technicality) | none |
| 2 | Look-ahead in code | Clean at label level (1-bar fill note) | none |
| 3 | Bad splits / time-series overlap | Excellent — purged k-fold + embargo | low* |
| 4 | Survivorship bias | **Present — real** | medium |

## 1. Standardization leakage — CLEAN

No `StandardScaler`/`MinMaxScaler`/`fit_transform` anywhere (grep clean).
Feature normalization is either **rolling-causal** or **cross-sectional within a
date**, never global-over-time:
- `technical.py`, `macro.py`, `microstructure.py`: z-scores use
  `.rolling(N).mean()/.std()` (e.g. `vix_z_20d`, `usd_inr_z_20d`) — only past
  bars.
- `fundamental.py`: winsorize + z-score computed **within
  `(asset_class, run_date)`** groups (cross-section of one snapshot), so no
  future quarter leaks in.
- `cross_sectional.py`: percentile ranks within `(date, asset_class)` — same-day
  cross-section only.

Negligible technicality: `xgboost_5d._magnitude_weights` bins by
`|fwd_ret|` quantiles computed over the whole training panel (incl. future). It
only sets per-row **sample weights** (not features, not predictions), so it
cannot leak a test label into a prediction; a purist would compute the quantiles
per-fold on train-only.

## 2. Look-ahead in code — CLEAN (label level)

`targets.compute`: `fwd_ret = adj_close.shift(-h)/adj_close - 1` — strictly
forward. Features at day *t* use data through close[*t*]; sentiment is lagged one
trading day; the OHLCV sanitizer is causal (`shift(1)` robust band). No feature
consumes a future bar.

Execution-realism note (not a data leak): the gated backtest treats a signal as
entering at the **same close** used to compute its features (`direction *
fwd_ret_5d` from close[t]). A stricter live sim would enter at *t+1* open. This
is a ~1-bar optimistic fill, common in research backtests; worth a t+1-open
variant later.

## 3. Splits / overlap — EXCELLENT, with one trading caveat

`cv/purged_kfold.py` implements de Prado AFML §7: contiguous time-ordered
validation folds, **purge** of train rows whose label window overlaps the fold,
**embargo** of the following `horizon+1` days, **per-ticker** purge, and explicit
`assert_no_leakage` / `assert_no_leakage_per_ticker` checks (the runner calls
them). This is decisively NOT random k-fold — concern #3's classic failure is
fully avoided.

*Caveat (the asterisk):* it is **bidirectional purged k-fold**, not strict
expanding-window walk-forward. Training for an early fold can include rows dated
*after* that fold (just purged around the boundary). That is the gold standard
for estimating **generalization AUC** and is leak-safe at the label level, but it
is **optimistic for the trading/Sharpe metrics** (Phase 28.11): an early-period
trade used a model that had "seen" later regimes. For an honest *live-trading*
P&L simulation, predictions should come from strict past-only walk-forward.
The function is named `walkforward_fit` but implements purged k-fold — a naming
mismatch worth tightening.

## 4. Survivorship bias — PRESENT (real)

`config/universe.yaml` is a **static, current-membership** list: today's
Nifty-50 core plus names that are listed and actively traded *now*
(ETERNAL/Zomato, NYKAA, PAYTM, JIOFIN, LICI, ...). Backtesting ~8 years on this
fixed set:
- never includes companies that were dropped from the index, delisted, or went
  bankrupt over the window (only winners that survived to 2026);
- uses 2026 membership for all historical dates (no point-in-time index
  reconstruction).

This inflates apparent historical success and directly feeds the **long-tilt
"beta"** that Phase 28.11 found masquerading as alpha. Proper fix is a
point-in-time universe (historical index membership snapshots incl. delisted
tickers) — a data-engineering effort, deferred.

## Bottom line

The two hardest classic leaks (#2 look-ahead, #3 split overlap) are handled
rigorously; #1 is clean. The two genuine credibility caveats are **survivorship
bias (#4)** and the **purged-k-fold-vs-walk-forward optimism** in the trading
metrics. Both are *mitigated* by the Tier-1 move to a cross-sectional /
market-neutral target: a within-date relative label cancels the market beta that
survivorship inflates, and ranking-based long/short books are far less sensitive
to which survivors are in the pool.
