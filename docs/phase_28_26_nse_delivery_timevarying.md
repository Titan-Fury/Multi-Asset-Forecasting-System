# Phase 28.26 — time-varying NSE delivery% (replaces the static snapshot)

Phase 28.25 used a single bhavcopy day as a static per-name delivery proxy.
This phase ingests the real multi-day delivery history and re-tests with
daily-changing membership — removing the static-snapshot look-ahead.

## Data ingestion (real, keyless)
- Source: NSE public archive `sec_bhavdata_full_<DDMMYYYY>.csv` (browser UA only;
  homepage 403 is irrelevant, the archive URL serves directly). Depth ~2020+.
- `scripts/fetch_nse_delivery_history.py` → `data/lake/nse_delivery_history.parquet`
  (gitignored): **1,660 trading days, 3.0M rows, 3,143 names, 2020-01-01…2026-05-22**
  of `[date, ticker, deliv_per, turnover_lacs]`. (1 day lost to a CSV encoding
  error; negligible.)

## Method
- Per name: trailing-20d mean delivery (uses only past+current data).
- Merge onto the 186-name price panel; classify each (name, date) low/high by the
  cross-sectional median that day → membership varies over time.
- `delivery_d` is known at end of day d, so conditioning the rev5→fwd(d,d+5)
  trade on it is information-clean. Harness: `scripts/nse_delivery_timevarying_eval.py`.

## Result (5d reversal, top/bottom 30%, market-neutral, 2020-2026 overlap)
| group | IC | gross Sharpe | net@10 |
| ----- | -- | ------------ | ------ |
| LOW delivery (speculative) | 0.0259 | 0.35 | −4.1% |
| HIGH delivery (investor-held) | 0.0377 | 0.70 | +1.4% |
| ALL (control) | 0.0330 | 0.74 | −0.2% |

## Finding
The Phase 28.25 result **holds under a proper time-varying test**: reversal is
stronger in HIGH-delivery (investor-held) names than LOW-delivery speculative
names (gross Sharpe 0.70 vs 0.35). The ordering is robust; the static snapshot
was directionally correct.

Magnitudes are lower than 28.25's full-sample numbers because delivery history
only exists from 2020, i.e. the **decayed era** — net-of-cost is roughly
breakeven. So: the *characterisation* (edge in investor-held names) survives, but
the recent-period *tradability* does not.

## Caveats
1. Delivery depth is 2020+ only (older NSE format lives at a different path).
2. Price panel is yfinance survivors-only (NSE survivorship still uncorrected).
3. 2020-2026 is post-decay; this is not a full-history retest.

## Next
Older NSE delivery (pre-2020) would extend the time-varying test into the strong
2014-2018 era; requires the legacy MTO/delivery archive format.
