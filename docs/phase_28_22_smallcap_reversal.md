# Phase 28.22 — real small/mid-cap data (yfinance) + down-cap reversal test

The first structural test on **real, freshly-ingested data** (not the large-cap
training panel). yfinance works in this environment with no API key, so a real
small/mid-cap NSE universe was pulled and tested.

## Data ingested (real, via yfinance — no key, no fabrication)

56 currently-listed NSE mid/small-cap names (IT-midcaps, chemicals, NBFCs,
consumer, capital-goods, cement), 2014-01 → 2026-05, **154,780 daily bars**,
saved to `data/lake/smallcap_prices.parquet`. (1 of 57 failed: wrong symbol
`CANFIN` → should be `CANFINHOME`.)

## Result — 5d cross-sectional reversal (top/bottom 20%, 5d hold)

Cost sensitivity (full sample): gross Sharpe **0.84**; net +9.5% @5bps,
+4.5% @10bps, ~0 @15bps.

Sub-period:

| period | IC | gross Sharpe | net @10bps | net @15bps |
| ------ | --: | -----------: | ---------: | ---------: |
| 2014–2018 | 0.044 | 1.21 | +12.6% | +7.6% |
| 2019–2022 | 0.033 | 0.90 | **+4.4%** | -0.7% |
| 2023–2026 | 0.016 | 0.17 | -7.2% | -12.2% |

## Read (honest, vs the large-cap baseline of Phase 28.20)

- **The down-cap thesis is empirically supported.** Reversal is stronger
  (full gross 0.84 vs large-cap 0.71) and, crucially, **persisted longer**:
  still net-positive at 10bps through 2019–2022, whereas large-cap reversal was
  already ~0/negative by 2019. This is the illiquidity-reversal premium showing
  up exactly where theory says it should — in smaller, less-liquid names.
- **But it is not a current free lunch.** 2023–2026 has decayed even here
  (gross 0.17, net-negative), and it remains cost-fragile (dies ~15bps).
- **Survivorship caveat (material).** The yfinance universe is current survivors
  only; delisted small-caps (which kept falling) are absent, so the "long the
  losers" leg is flattered — the real, survivorship-clean edge is almost
  certainly weaker than shown, especially in the early years.

## Verdict

Genuinely the most positive structural result of the session, on real data: a
**longer-lived reversal/illiquidity edge down-cap**, net-positive as recently as
2019–2022. It justifies a *proper* small/mid-cap build — but only with a
**survivorship-clean universe (incl. delisted names)** and conservative cost
assumptions, both of which yfinance alone cannot provide. Nothing wired into
production; this is an evidence-gathering result, not a deployed signal.

See `docs/data_ingestion_plan.md` for the concrete feed/schema/integration plan.
