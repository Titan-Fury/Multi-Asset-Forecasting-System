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
- Resulting panel: **181 names (55 delisted, 126 active)**, 320,710 rows,
  2014-2026, after a **$5 price floor** and **1/99 pct daily winsorization** of
  forward returns.
- Harness: `scripts/build_us_pit_universe.py`, `scripts/fetch_us_pit_panel.py`,
  `scripts/us_pit_reversal_eval.py`. Parquet gitignored.

## Result (5d cross-sectional reversal, top/bottom quintile, market-neutral)

### 1. Survivorship effect
| panel | IC | gross Sharpe | net@10 |
| ----- | -- | ------------ | ------ |
| FULL (incl. delisted losers) | 0.0292 | 0.83 | +7.2% |
| SURVIVORS only (delisted dropped) | 0.0332 | 0.93 | +10.2% |

After proper hygiene, survivorship bias inflates the edge only **modestly**
(Sharpe 0.93 → 0.83, ~10-30% of the signal). **Important honesty correction:**
the *raw* run (no price floor / winsorization) showed Sharpe 0.90 → 0.43 — a
~2x effect — but that was a **penny-stock outlier artifact** (delisted sub-$1
names with explosive % moves dominating the cross-section), not real
survivorship bias. The clean answer is the modest one. Outlier hygiene mattered
more than survivorship here.

### 2. Illiquidity cut (PIT daily dollar-volume tertile, no look-ahead)
| tertile | IC | gross Sharpe | net@10 |
| ------- | -- | ------------ | ------ |
| BOTTOM (illiquid/small) | 0.0369 | 0.76 | +19.4% |
| TOP (liquid/large) | 0.0257 | 0.56 | +4.7% |

Reversal is **stronger in the illiquid/small tertile** on US data, with a
PIT-correct within-day liquidity cut and no universe look-ahead. This is the
cleanest support yet for the down-cap / illiquidity-premium thesis.

### 3. Sub-period (full panel)
| window | IC | gross Sharpe | net@10 |
| ------ | -- | ------------ | ------ |
| 2014-2018 | 0.0348 | 1.33 | +12.0% |
| 2019-2022 | 0.0405 | 1.64 | +39.6% |
| 2023-2026 | 0.0076 | 0.42 | −1.7% |

Strong and persistent through 2022 (the 2019-22 strength likely flattered by
COVID-era volatility), then **decays 2023-2026** — same decay seen on NSE.

## Caveats (do not over-claim)
1. Small panel: 181 names, only **55 delisted** survived (Tiingo coverage gaps +
   Polygon↔Tiingo symbology mismatches; many delisted names 404'd). A larger
   delisted set would tighten the survivorship estimate.
2. A few active names dropped to transient yfinance DNS errors during the fetch.
3. net@10 uses 10bps; real small/illiquid spreads are wider — treat magnitudes
   as optimistic, the Sharpe ordering as the robust signal.

## Verdict
The illiquidity premium in 5d reversal is **real and PIT-confirmed on US data**
(stronger in the illiquid tail), survivorship bias is **modest after hygiene**,
but the edge has **decayed post-2022**. Consistent down-cap story across NSE and
US: real historically, concentrated in the illiquid tail, not a current free
lunch.
