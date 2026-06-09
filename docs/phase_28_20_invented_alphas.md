# Phase 28.20 — invented + tested alphas: reversal works historically, decayed now (no hallucinated win)

Brief: research current prediction practices, *invent* candidate algorithms,
test each locally under strict walk-forward, report honestly. No production
change — the honest result is a decayed anomaly, not a deployable edge.

## What was invented & tested (strict WF, cross-sectional long/short, vs baseline ML)

| signal | IC | gross Sharpe | net @5bps |
| ------ | --: | -----------: | --------: |
| baseline XGBoost (xs-median) | 0.017 | 0.11 | -3.5% |
| `rev5` — 5d cross-sectional reversal (−ret_5d) | 0.029 | 0.65 | +6.8% |
| `radjmom` — risk-adjusted 20d momentum | -0.018 | -0.53 | -13.7% |
| `RCMRS` — regime-conditional momentum/reversal switch (invented) | 0.008 | 0.17 | -1.9% |
| `model × reversal` rank-ensemble | 0.030 | 0.77 | +7.7% |

First-pass reads: short-horizon **cross-sectional reversal beats the ML model**
and is net-positive at 5bps; **momentum is negative** on this universe; my
fancy **regime-switch invention (RCMRS) did not work** (null); the model adds a
little on top of reversal.

## Stress test (the part that matters — don't trust a full-sample headline)

**Cost sensitivity** (rev5, full panel, top/bottom-20%, 5d hold):

| round-trip cost | Sharpe | net ann |
| --------------- | -----: | ------: |
| 0 bps  | 0.71 | +13.0% |
| 5 bps  | 0.44 | +7.9% |
| 10 bps | 0.16 | +2.9% |
| 15 bps | **-0.12** | **-2.2%** |
| 20 bps | -0.40 | -7.2% |

The edge dies at ~10–15 bps. Reversal concentrates trades in the most-moved
(higher-impact, wider-spread) names, so its *true* cost sits at the bad end of
that range — not the optimistic 5 bps.

**Sub-period stability** (rev5):

| period | IC | gross Sharpe | net @15bps |
| ------ | --: | -----------: | ---------: |
| 2011–2014 | 0.035 | 1.30 | +10.0% |
| 2015–2018 | 0.046 | 1.14 | +2.8% |
| 2019–2022 | 0.022 | 0.12 | **-12.9%** |
| 2023–2026 | 0.018 | 0.30 | **-9.1%** |

The reversal edge was strong in 2011–2018 and has been **arbitraged away** since
~2019 — net-negative in the recent, decision-relevant period. The +6.8%
full-sample headline was carried entirely by the old era.

## Honest verdict (no hallucination)

- I did NOT discover a deployable groundbreaking edge. I found *where the
  historical edge lived* (short-horizon reversal, 2011–2018) and confirmed it is
  **decayed and cost-fragile today** — the textbook life-cycle of a published
  anomaly on a liquid, now-efficient universe.
- The regime-switch invention (RCMRS) added nothing.
- This is fully consistent with the session's wall (28.8/28.13/28.18): liquid
  NIFTY large-cap daily/weekly direction has no robust, currently-exploitable
  cross-sectional alpha after costs.
- **Nothing implemented into production.** Deploying a decayed, cost-negative
  anomaly would be the opposite of an improvement.

## What this actually tells you (actionable, honest)

The model isn't "broken" — the *market* is efficient for this universe/horizon.
To find live edge you must change the problem, not the algorithm:
- **Less-efficient universe**: liquid mid/small-caps where reversal/illiquidity
  premia persist (the survivorship-clean version — Phase 28.14).
- **Different signal class**: event-driven (PEAD, earnings-surprise drift),
  intraday→multi-day, or alt-data with genuine information not yet in price.
- **Shorter decay-aware retraining** + live monitoring of IC decay (an anomaly
  that worked last year may already be gone, as 2019→ shows).
Methodology stays fixed: every candidate validated under `OMNI_CV_MODE=strict`
with cost + sub-period stress tests before it is ever called an edge.
