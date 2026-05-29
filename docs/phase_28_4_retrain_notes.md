# Phase 28.4 retrain notes (h=5 universal model)

## Outcome summary

Phase 28.4 ran an end-to-end retrain after the Track A + Track B
source-code improvements landed. Result: the new candidate scored
**OOF AUC 0.5156**, a regression from the **Phase 28.3 baseline of
0.5402**. Per the plan's revert criteria the model artifact was rolled
back to the Phase 28.3 baseline; all source-code improvements stay
live.

| Metric                | Phase 28.3 (kept) | Phase 28.4 attempt |
| --------------------- | ----------------: | -----------------: |
| OOF AUC (h=5)         | 0.5402            | 0.5156             |
| Training rows         | 205,536           | 206,868            |
| Features              | 128               | 128                |
| Folds                 | 5 (purged)        | 5 (purged kfold)   |
| Recency half-life     | 1.5y              | 1.5y               |
| `pt_mult` / `vol_window` | 2.0 / 20       | 1.5 / 40           |
| Positive class share  | ~25 %             | ~33 %              |

## What changed between the runs

Source code:
* Track A1 — feature contract emission alongside model joblib
* Track A2 — feature contract assertion at inference (soft-fail)
* Track A3 — diff_lagged ported to live inference
* Track A4 — circuit features ported to live inference
* Track A5 — quality features ported to live inference
* Track A6 — freshness gate on cross-sectional rank slice
* Track A7 — conformal calibrator passed through cache (de-dup)
* Track A8 — stack meta-LR mask NaN OOF rows
* Track B9 — `ret_60d` added (unblocks `mom_vol_adj_60d`)
* Track B10 — `_RANK_COLS` expanded by 9 columns
* Track B11 — `colsample_bylevel=0.7`
* Track B12 — `max_depth` 5→6, `min_child_weight` 5→3
* Track B13 — magnitude weight quantile binning
* Track B14 — TB `vol_window` 20→40
* Track B15 — TB `pt_mult` 2.0→1.5
* Track B16 — single-mode CV switched to `purged_kfold_indices`
* Track B17 — auto-switch isotonic→beta on small inner_val (< 2000)

Training data:
* `tb_label_*` columns regenerated with `pt_mult=1.5`, `vol_window=40`
* Class share moved from 25/75 to 33/67 — the binarised problem
  is genuinely harder (more rows on the boundary)

## Plausible cause of the regression

1. **Class rebalance**. AUC reflects both the model's discrimination
   AND the difficulty of the labelling. Going from a 25/75 problem
   (where ~50% of negatives are easy time-expired neutrals) to a
   33/67 problem (where the negative class includes more boundary-
   adjacent rows) makes the AUC ceiling lower even when the model
   is more useful for actual trading.

2. **Hyperparameter combination**. `max_depth=6` × `min_child_weight=3`
   × `colsample_bylevel=0.7` × the new quantile-binned magnitude
   weights × the auto-beta calibration is a four-axis change. A
   one-at-a-time A/B retrain matrix would isolate which axis hurt.

3. **Per-ticker purge** removes ~30 % more cross-ticker train rows
   from each fold — extra capacity that XGBoost gets to use, but
   also extra rows for it to overfit when the features it now sees
   include the cleaner (smaller) TB labels.

## Decision

Rolled back the model artifact to the Phase 28.3 baseline:
* `data/models/universal_5d.joblib` — Phase 28.3 (oof_auc=0.5402)
* `data/models/universal_5d.contract.json` — written by Track A1 hook
* `data/models/universal_5d.meta.json` — Phase 28.3 metadata

The source-code improvements (Tracks A, B, C, D) ship as-is. They are
correctness, efficiency, and hygiene fixes that are independent of
which model artifact is on disk.

## Follow-up experiments (deferred)

1. **One-at-a-time retrain matrix.** Re-run with each Track B change
   isolated:
   - Phase 28.3 baseline + B11 (colsample_bylevel) only
   - Phase 28.3 baseline + B12 (max_depth + min_child_weight) only
   - Phase 28.3 baseline + B13 (quantile mag weights) only
   - Phase 28.3 baseline + B14+B15 (TB rebalance) only
   - Phase 28.3 baseline + B16 (per-ticker purge) only
   - Phase 28.3 baseline + B17 (auto-beta) only
2. **Calibration curve diagnostic.** Class rebalance often hurts AUC
   while improving Brier score / log loss. Compare the two artifacts'
   reliability diagrams, not just AUC.
3. **Backtest comparison.** Both artifacts on the same recent
   walk-forward window. The 33/67 class balance produces probabilities
   with different operating characteristics — e.g. P>0.6 might be a
   high-conviction LONG under one model and a marginal LONG under
   the other. Trading utility, not OOF AUC, is the final word.

## Tooling
The retrain command used:

```bash
python scripts/augment_triple_barrier.py
python scripts/train_universal_model.py \
    --force --asset-classes india --horizons 5 \
    --ensemble single --target tb \
    --cv purged --half-life-years 1.5
```

The augment script needs to be run BEFORE the trainer because
B14 + B15 changed the TB-label hyperparameters and the existing
parquet labels were stale.

## Files affected

Source improvements (all committed in this phase):
* `omni/features/contract.py` (new)
* `omni/features/cross_sectional.py`
* `omni/features/diff_lagged.py` (no change; just ported to inference)
* `omni/features/quality.py` (no change; just ported to inference)
* `omni/features/circuit.py` (no change; just ported to inference)
* `omni/features/targets.py`
* `omni/features/technical.py`
* `omni/models/stack.py`
* `omni/models/xgboost_5d.py`
* `omni/ondemand/features_inline.py`
* `omni/ondemand/lookup.py`
* `omni/ondemand/model_registry.py`
* `omni/ondemand/predict.py`
* `omni/monitoring/predict_log.py`
* `omni/ingest/sources/currents.py`
* `scripts/augment_triple_barrier.py`
* `scripts/probe_predictions.py` (rename + commit)
* `scripts/train_universal_model.py`

Files removed (dead-code sweep):
* 12 probe / diag / smoke scripts under `scripts/`
* (`omni/storage/drive.py` was wrongly removed; restored in fix-up
  commit `e1d01fc`)
