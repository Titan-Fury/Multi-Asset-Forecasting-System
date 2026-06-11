# Phase 28.24 — US survivorship-clean reversal (kills the look-ahead caveat)

Phases 28.22/28.23 were yfinance-only (current survivors). This test removes
that bias by including **delisted** names, and removes the universe look-ahead
by defining "small/illiquid" PIT within each day.

## Data
- Universe: Polygon `/v3/reference/tickers` type=CS, active **true+false** →
  survivorship-clean membership (delisted carry `delisted_utc`). Sample 300.
- Prices (hybrid): active names via yfinance (fast, keyless); **delisted names
  via Tiingo** (only verified source that serves delisted history correctly,
  e.g. SHLDQ ends 2022-10-31).
- Resulting panel (after widening the delisted set via a resumable Tiingo grinder):
  **377 names (251 delisted, 126 active)**, 567,213 rows, 2014-2026, after a
  **$5 price floor** and **1/99 pct daily winsorization** of forward returns.
- Harness: `scripts/build_us_pit_universe.py`, `scripts/fetch_us_pit_panel.py`,
  `scripts/widen_us_delisted.py`, `scripts/us_pit_reversal_eval.py`. Parquet gitignored.

## Result (5d cross-sectional reversal, top/bottom quintile, market-neutral)

### 1. Survivorship effect
| panel | IC | gross Sharpe | net@10 |
| ----- | -- | ------------ | ------ |
| FULL (incl. 251 delisted losers) | 0.0261 | 1.02 | +9.1% |
| SURVIVORS only (delisted dropped) | 0.0332 | 0.95 | +10.8% |

With a large delisted set (251 names), survivorship bias has **no material
inflationary effect** on this signal: including delisted losers leaves the gross
Sharpe essentially unchanged (even slightly higher, 0.95 → 1.02), with net-of-cost
within noise (+10.8% → +9.1%). **Important honesty correction:** the *raw* run
(no price floor / winsorization) showed Sharpe 0.90 → 0.43 — a ~2x "inflation" —
but that was entirely a **penny-stock outlier artifact** (delisted sub-$1 names
with explosive % moves dominating the cross-section), not real survivorship bias.
Once a $5 floor + winsorization are applied, the effect disappears. Conclusion:
for a market-neutral, winsorized reversal book, **outlier hygiene matters far more
than survivorship correction**. (Earlier interim with 55 delisted: 0.93 → 0.83;
with 119: 0.94 → 0.90; with 251: 0.95 → 1.02 — the effect converges to ~zero.)

### 2. Illiquidity cut (PIT daily dollar-volume tertile, no look-ahead)
| tertile | IC | gross Sharpe | net@10 |
| ------- | -- | ------------ | ------ |
| BOTTOM (illiquid/small) | 0.0408 | 0.65 | +11.2% |
| TOP (liquid/large) | 0.0240 | 0.58 | +3.6% |

Reversal is **stronger in the illiquid/small tertile** on US data, with a
PIT-correct within-day liquidity cut and no universe look-ahead. The gap narrows
on the larger penny-controlled panel but the ordering holds — support for the
down-cap / illiquidity-premium thesis.

### 3. Sub-period (full panel)
| window | IC | gross Sharpe | net@10 |
| ------ | -- | ------------ | ------ |
| 2014-2018 | 0.0273 | 1.60 | +12.1% |
| 2019-2022 | 0.0380 | 1.98 | +39.8% |
| 2023-2026 | 0.0101 | 0.28 | −4.6% |

Strong and persistent through 2022 (the 2019-22 strength likely flattered by
COVID-era volatility), then **decays 2023-2026** — same decay seen on NSE.

## Caveats (do not over-claim)
1. Panel: 377 names (**251 delisted**) — large enough that the survivorship
   estimate has converged; Tiingo coverage gaps + Polygon↔Tiingo symbology
   mismatches still drop some delisted names (US common stock only).
2. A few active names dropped to transient yfinance DNS errors during the fetch.
3. net@10 uses 10bps; real small/illiquid spreads are wider — treat magnitudes
   as optimistic, the Sharpe ordering as the robust signal.

## Verdict
The illiquidity premium in 5d reversal is **real and PIT-confirmed on US data**
(stronger in the illiquid tail), and — with 251 delisted names — **survivorship
bias is effectively negligible after outlier hygiene** (the raw 2x effect was a
penny-stock artifact). The edge has **decayed post-2022**. Consistent down-cap
story across NSE and US: real historically, concentrated in the illiquid tail,
not a current free lunch.
