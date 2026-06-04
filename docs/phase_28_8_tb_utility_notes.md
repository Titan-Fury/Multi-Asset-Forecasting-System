# Phase 28.8 — triple-barrier trading-utility A/B + duplicate-row fix

## Why

The 28.7 hyperparameter matrix proved the 28.4 model is optimal *on a fixed
label set*, and flagged that the apparent 0.5402→0.5156 OOF-AUC drop is a
label artifact (28.4 rebalanced TB labels from 25%→33% positive via
pt_mult 2.0→1.5, vol_window 20→40). OOF AUC is not comparable across label
definitions, so it can't say whether the rebalance helped or hurt the
*deployed* signal. This phase answers that on metrics judged against
**realized** `fwd_ret_5d` (label-independent), so the two TB configs are
compared on the same ground truth.

## Method

For each TB config, recompute `tb_label_5d` on the 205,283 unique-key india
rows, walk-forward fit (5 folds, min_train=2520), then score the pooled OOF
`proba_up` against realized 5d returns via `omni/backtest/metrics.py`:
top/bottom-decile mean realized return and the realized decile spread.

## Result (judged on realized fwd_ret_5d)

| metric | 28.4 current (33% pos) | 28.3 legacy (25% pos) |
| ------ | ---------------------: | --------------------: |
| pos_share | 0.328 | 0.250 |
| AUC *(label-confounded)* | 0.5295 | 0.5752 |
| top-decile realized ret | +0.00413 | +0.00484 |
| bottom-decile realized ret | +0.00491 | +0.00510 |
| **decile spread (top−bottom)** | **−0.0013** | **+0.0002** |

The higher 28.3 AUC is the label confound again (easier 25% labels), not a
real edge. On the label-independent metric the verdict is clear: **both
configs have a near-zero (even negative) realized-return decile spread.** The
raw XGBoost `proba_up` has negligible standalone ranking power on this
NIFTY-large-cap universe regardless of TB config — the most-confident-up
decile does not outperform the least-confident one in realized terms.

## Decision

**Keep 28.4 TB params.** The A/B is inconclusive on trading utility: there is
no evidence the 28.4 rebalance hurt deployed utility, and none that reverting
to 28.3 would help. The flat decile spread also confirms *why* the system
relies on the rubric + conviction gate rather than the raw model probability —
this raw-model analysis deliberately excludes that layer (the gate cherry-picks
high-conviction rows). A full `run_backtest` gated-metric evaluation is the
right tool for measuring the deployed signal and is left as future work.

## Side finding (fixed)

Building the A/B harness surfaced a real data bug: the training set carried
1,585 duplicate (date, ticker) rows (0.77%), 1,322 of them byte-identical.
Fixed in `build_training_set.py` (drop_duplicates keep=last before persist) —
see that commit. Unfixed, duplicates overweight those rows in the loss and
fan out (date, ticker) joins (this harness's first run exploded to 348M rows
from a dup-on-dup merge before the fix).
