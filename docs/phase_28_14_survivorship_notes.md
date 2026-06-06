# Phase 28.14 — survivorship/selection bias: quantified, scoped, blocked on data

## Goal (c): "kill survivorship". Honest status: cannot fully fix without
external data that isn't in the lake — but quantified and scoped here.

## What's missing

`config/universe.yaml` is a static, current-membership list. The lake has
`fundamentals_pit` (point-in-time fundamentals) but **no index-membership /
constituent history** and **no delisted tickers**. Every name in the training
panel survives to 2026-05-22. You cannot reconstruct a point-in-time universe
from data that only contains survivors.

## Quantified bias (training panel: 59 names, 2011-05-09 → 2026-05-22)

- **Equal-weight survivor-basket CAGR ≈ 23.6%** vs NIFTY-50 ~12%/yr over the
  same window — a **~10 pp/yr survivorship + selection premium** baked into any
  long-only / absolute-direction backtest.
- **Selection smoking gun** — names listed >4y after panel start, with their
  cumulative return since listing:

  | ticker | listed | cum return since listing |
  | ------ | ------ | -----------------------: |
  | GRSE | 2018-10 | +3,851% |
  | MAZDOCK | 2020-10 | +3,070% |
  | HAL | 2018-04 | +828% |
  | BDL | 2018-03 | +673% |
  | COCHINSHIP | 2017-08 | +648% |
  | SBILIFE | 2017-10 | +169% |

  These defense-PSU / shipbuilding names were almost certainly added to the
  universe *because* they became popular winners — textbook look-ahead
  selection. A point-in-time universe would not have held them early, and would
  include the names that were dropped/delisted (the losers we never see).

## Why it's partly moot now (given 28.11-28.13)

Survivorship inflates **long-only / absolute-direction** returns — exactly the
beta the 28.11 directional book was riding. The Tier-1 **market-neutral
cross-sectional** book cancels basket-level drift, so it is structurally far
less exposed to this bias. And 28.13's strict walk-forward already showed no
robust edge regardless. So the bias mainly invalidates the *long-only/absolute*
numbers, which we have already stopped trusting.

## To actually fix (requires external data)

1. Source **NIFTY historical reconstitution** (semi-annual membership changes)
   and **delisted/suspended tickers** with their price history (NSE indices
   archive, or a paid survivorship-bias-free dataset).
2. Store membership as `[ticker, valid_from, valid_to]` intervals.
3. Filter the universe **as-of each date** at panel-build time, including names
   that were members then but are gone now.

Deliberately NOT scaffolded in code: an unused PIT-membership filter with no
data to feed it would be dead code (see the Phase 28.15 dead-code review). The
integration point is `scripts/build_training_set.py` universe resolution, to be
built when the membership feed exists.

## Bottom line

Survivorship/selection bias is **real and large (~10 pp/yr)** for long-only
metrics, **quantified here**, **structurally mitigated** by the market-neutral
framing, and **blocked on external membership data** for a full fix. Treat all
absolute/long-only backtest returns as upward-biased; rely on market-neutral +
`OMNI_CV_MODE=strict` for honest reads.
