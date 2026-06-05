# Phase 28.10 — base-rate-aware blend neutral (hardening + honest validation)

## What

`BlendConfig.model_base_rate` (default `None` = no-op). When set to the
universal model's training positive base rate, `blend()` re-centers `proba_up`
from that base rate to a 0.5 prior via prior-correction
(`p*(1-b)/(p*(1-b)+(1-p)*b)`) before mixing with the rubric. Fixes the
base-rate/0.5-neutral mismatch from `phase_28_9`. Default-off path is
byte-identical to before, so the deployed (~balanced) model is unaffected.

## Unit verification

Transform maps any base rate -> 0.5 exactly, is monotone in p, stays in [0,1],
passes NaNs through. With a synthetic 0.33-centered model: off -> 6 short / 2
neutral (the bug); on -> 4 long / 3 short / 1 neutral (mean blended_p
0.418 -> 0.517).

## End-to-end validation (run_backtest, india tb_label_5d, 205,283 rows)

| | direction split | emitted | dir_hit_rate | mean_signed_ret |
| --- | --------------- | ------: | -----------: | --------------: |
| OFF (0.5 neutral) | -1: 75,883 / +1: 216 (99.7% short) | 76,099 | 0.469 | -0.0044 |
| ON (base_rate=0.33) | +1: 1,666 / -1: 604 (long-tilted) | 2,270 | **0.514** | -0.0064 |

## Honest read

The fix does exactly what it should and nothing more:
- **Mechanical bug fixed** — direction is no longer degenerate-short, the gate
  becomes appropriately selective (76k -> 2.3k emitted), and directional
  hit-rate crosses above 50% (0.469 -> 0.514).
- **It does NOT create alpha.** `mean_signed_fwd_ret` stays slightly negative
  (-0.0064): the emitted signals are directionally right >50% of the time but
  the losers are larger than the winners (magnitude asymmetry). This is fully
  consistent with the 28.8 finding that the raw signal's realized-return decile
  spread on this NIFTY-large-cap universe is ~0. Correcting a bias cannot
  manufacture edge that the base signal does not have.

## Status

Shipped as opt-in (default off); production behaviour unchanged. To actually
deploy it you would (a) store the universal model's training base rate in its
meta/contract and (b) pass it into `BlendConfig` at inference — deferred until a
universal model is actually trained on an imbalanced target. The real lever for
*positive expectancy* is improving the base signal, not the neutral point — see
`docs/prediction_efficiency_research.md`.
