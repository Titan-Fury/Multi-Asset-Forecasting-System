# Phase 28.23 — widened NSE mid/small-cap reversal (name-sample robustness)

Follow-up to Phase 28.22. That test used a hand-picked 56-name set. Here the
universe is **data-derived** from the NSE bhavcopy (`sec_bhavdata_full`,
2026-06-19), EQ series, turnover rank 55–305, excluding mega-cap F&O names and
ETFs → **186 names** with ≥300 trading days, 427,149 rows, 2014-01-01…2026-05-29.

Harness (committed, reproducible):
- `scripts/build_smallcap_universe.py` → `configs/universe_smallcap.yaml`
- `scripts/fetch_smallcap_panel.py` (yfinance) → `data/lake/smallcap_prices.parquet` (gitignored)
- `scripts/smallcap_reversal_eval.py`

## Result (5d cross-sectional reversal, top/bottom quintile, market-neutral)

Full sample: IC = 0.0357; gross Sharpe 0.51.

| cost | Sharpe | net ann |
| ---- | ------ | ------- |
| 0bps  | +0.51 | +8.4% |
| 5bps  | +0.20 | +3.3% |
| 10bps | −0.11 | −1.7% |
| 15bps | −0.41 | −6.8% |

Sub-period:

| window | IC | gross Sh | net@10 | net@15 |
| ------ | -- | -------- | ------ | ------ |
| 2014–2018 | 0.0455 | 1.48 | +14.1% | +9.0% |
| 2019–2022 | 0.0352 | 0.02 | −9.7% | −14.7% |
| 2023–2026 | 0.0219 | −0.36 | −15.4% | −20.4% |

## What changed vs the 56-name set (28.22)

| | 56 hand-picked (smaller) | 186 data-derived (more liquid) |
| --- | --- | --- |
| full gross Sharpe | 0.84 | 0.51 |
| 2019–2022 net@10 | +4.4% | −9.7% |
| breakeven cost | ~15bps | ~5–10bps |

The edge is **weaker and shorter-lived on the more-liquid band**, and the
2019–2022 persistence seen in 28.22 **does not survive** widening to liquid
mid-caps. Read together this is consistent with the illiquidity-premium thesis:
reversal concentrates in the **smaller / less-liquid tail** and decays as you
move up the liquidity ladder — and it has decayed over time regardless.

## Honest caveats (do not over-claim)

1. **Universe look-ahead.** The 186 names were selected on **2026** turnover —
   i.e. names that are liquid *today*. That is survivorship + look-ahead in
   universe membership; the 2014–2018 sub-sample is flattered by it.
2. **Survivorship in prices.** yfinance carries current survivors only; the
   "buy the losers" leg is still flattered. A point-in-time / delisted-inclusive
   universe is required before any tradable claim (US: Polygon/Tiingo keys now
   verified working; NSE: needs PIT constituents).
3. **Costs.** Breaks even at ~5–10bps — fragile for small/mid-cap spreads.

## Verdict

Down-cap reversal is **real historically and concentrated in the illiquid
tail**, but it is **decayed post-2018, cost-fragile, and not demonstrably
tradable** under honest universe construction. Not a deployable edge as-is.

Next: re-run on a **point-in-time** universe (US via Polygon delisted-inclusive)
to remove the look-ahead, and add the NSE `DELIV_PER` illiquidity feature
(Phase A2) to test whether the signal sharpens in low-delivery names.
