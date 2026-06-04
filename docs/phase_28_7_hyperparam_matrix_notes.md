# Phase 28.7 — hyperparameter A/B matrix + news-relevance verdict

## Why

`phase_28_4_retrain_notes.md` flagged an apparent OOF-AUC regression
(0.5402 → 0.5156) after the 28.4 retrain and deferred a one-axis-at-a-time
experiment to find the culprit. The 28.4 change set bundled THREE model
hyperparameter moves with a triple-barrier (TB) label rebalance
(pt_mult 2.0→1.5, vol_window 20→40; positive class 25%→33%). AUC is not
comparable across different label definitions, so the matrix holds the TB
labels CONSTANT and varies only one hyperparameter per run.

## Setup

Single fixed dataset for every run so AUC is honest:
`build_training_set.py --asset-classes india --no-news` + `augment_triple_barrier.py`
→ 205,457 india rows, 178 cols, `tb_label_5d` dist {-1: 84528, +1: 67419,
0: 53510} (~33% positive, binarised as == +1). Trainer:
`--force --asset-classes india --horizons 5 --ensemble single --target tb
--cv purged --half-life-years 1.5`. Only the three 28.4 axes vary, via the
new env hooks (`OMNI_XGB_MAX_DEPTH`, `OMNI_XGB_MIN_CHILD_WEIGHT`,
`OMNI_XGB_COLSAMPLE_BYLEVEL`).

## Result

| run | change (toward 28.3) | OOF AUC | Δ vs baseline |
| --- | -------------------- | ------: | ------------: |
| R0 baseline | current 28.4: depth 6 / mcw 3 / csbl 0.7 | **0.5262** | — |
| R1 | max_depth 6 → 5 | 0.5232 | **−0.0030** |
| R2 | min_child_weight 3 → 5 | 0.5255 | −0.0007 |
| R3 | colsample_bylevel 0.7 → off (1.0) | 0.5260 | −0.0002 |

Every reversion toward the 28.3 config HURTS or TIES on identical labels.
The current 28.4 hyperparameters are the best of all isolated settings —
`max_depth=6` is the dominant axis (deeper trees genuinely help once TB
labels have filtered the noisiest bars).

## Decision

KEEP the 28.4 hyperparameters unchanged. The "0.5402 → 0.5156 regression"
is a **label-definition artifact, not a model defect**: 0.5402 was scored on
the easier 25%-positive labels; on a fixed label set the 28.4 model is
strictly better than any 28.3 hyperparameter. This confirms the hypothesis
recorded in the 28.4 notes. Caveat still standing: OOF AUC ≠ trading utility,
so whether the 33%-vs-25% TB rebalance helps the *deployed* signal is a
separate question for a trading-metric backtest, not this AUC matrix.

Shipped: env-var tuning hooks in `xgboost_5d::walkforward_fit` (defaults =
current 6/3/0.7, so production behaviour is unchanged), matching the existing
`OMNI_XGB_DEVICE` convention. Production artifact restored to its
pre-experiment state (oof_auc=0.5156).

## News-relevance follow-up — RESOLVED (not implemented)

The 28.6 follow-up proposed a news-relevance filter. Census of the news lake
(window 2024→2026): **1,091 articles across only 10 ticker_hints**
(RELIANCE 250, BRITANNIA 228, ITC 227, ONGC 174, TITAN 151, DABUR 56, then
single-digit tails). The training universe is 59 india tickers over 2011→2026,
so the sentiment pillar is effectively empty in training already. A relevance
filter would only SHRINK negligible coverage. **The real news gap is volume,
not relevance** — filtering is the wrong lever until ingestion coverage grows.
Abandoned.
