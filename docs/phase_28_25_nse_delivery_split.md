# Phase 28.25 — NSE delivery-% split (A2): reversal vs delivery character

Tests whether 5d reversal concentrates in low-delivery (speculative-churn) vs
high-delivery (investor-held) NSE mid/small-caps, using `DELIV_PER` from the
NSE bhavcopy (`sec_bhavdata_full`). Harness: `scripts/smallcap_delivery_split.py`.

## Data / method
- Panel: the 186-name NSE mid/small-cap set (Phase 28.23).
- Delivery%: matched for all 186 names from the bhavcopy. Median = 45.3%.
- Split low/high at the median; compare reversal (top/bottom 30%, market-neutral).

**Single-snapshot caveat:** only one bhavcopy day (2026-06-19) is available, so
delivery is used as a **static per-name character proxy** applied across
2014-2026. Delivery regime is fairly persistent for a name, but this is one
snapshot → **suggestive, not definitive**. A proper test needs delivery history
(multi-day bhavcopy ingestion — see `docs/data_ingestion_plan.md`).

## Result (full sample)
| group | names | IC | gross Sharpe | net@10 |
| ----- | ----- | -- | ------------ | ------ |
| LOW delivery (<45.3%, speculative) | 100 | 0.0345 | 0.44 | −2.0% |
| HIGH delivery (≥45.3%, investor) | 86 | 0.0392 | 1.20 | +7.2% |

## Finding (counter-intuitive)
Reversal is **stronger in HIGH-delivery (investor-held) names**, not the
low-delivery speculative-churn names — the opposite of the naive "reversal lives
in churny names" hypothesis. Interpretation: in high-delivery names, 5d moves
reflect real-money over/under-reaction that mean-reverts and survives cost; in
low-delivery churny names the 5d signal is noisier and net-negative after cost.

This refines the down-cap story: it is not *raw speculative churn* that carries
the reversal premium, but **genuine positioning reversals in investor-held
names**. Combined with Phase 28.24 (premium concentrates in the illiquid/small
tertile), the edge looks like: small + investor-held (not small + churny).

## Caveats
1. Single-day delivery snapshot used statically (above).
2. Full-sample only; not sub-period — and the panel decayed post-2018 (28.23).
3. Survivorship (yfinance current survivors only).

## Next
Ingest multi-day NSE bhavcopy `DELIV_PER` history to make delivery a true
time-varying feature, then re-test in a strict walk-forward.
