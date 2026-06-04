# Phase 28.5 (Tier B) — technical feature A/B retrain notes (h=5 universal)

## Outcome summary

Tier B added four new families of ticker-invariant technical features to
`omni/features/technical.py` and ran an **isolated A/B retrain** (only the
feature set changed; CV, target, half-life, asset-classes, hyperparameters
all held fixed) to measure their marginal value on the h=5 single model.

Result: **no improvement.** Adding the 13 new columns moved OOF AUC by
**−0.0004** — pure noise. Per the established revert protocol (see
`phase_28_4_retrain_notes.md`) the model artifact was restored to the
pre-experiment production state, and — because the lift was null — the
feature *source code was also reverted* (the technical.py + cross_sectional.py
feature commits were dropped). Only the `build_training_set.py` collision
fix was retained (a real correctness bug, independent of the experiment).
This file remains as the negative-result record so the same four families
are not re-attempted without a reason to expect a different outcome.

| Run                         | OOF AUC | Features | Rows    |
| --------------------------- | ------: | -------: | ------: |
| Control (current feats)     | 0.5261  | 155      | 205,460 |
| Treatment (+13 Tier B cols) | 0.5257  | 168      | 205,460 |
| **Delta**                   | **−0.0004** | +13  | —       |

Both runs: `--asset-classes india --no-news --ensemble single --target tb
--cv purged --half-life-years 1.5`. `--no-news` was used for both so the
delta cleanly attributes to the price-derived features (the news pillar is
removed uniformly and does not interact with them). Absolute AUC therefore
differs from the news-included production lineage (0.5156); only the
control→treatment **delta** is the verdict.

## Features added (kept as source)

1. **Multi-period RSI** — `rsi_2`, `rsi_5`, `rsi_21` (Wilder EWM, same
   formula as the existing `rsi_14`; n=14 output byte-identical).
2. **Bollinger bandwidth** — `bb_bandwidth = (upper−lower)/sma20` (squeeze).
3. **Daily rolling-VWAP distance** — `dist_vwap_5`, `dist_vwap_20`
   (`close / Σ(typ·vol)/Σ(vol) − 1`, typ=(H+L+C)/3).
4. **Return distribution shape** — `ret_skew_20`, `ret_kurt_20`,
   `ret_skew_60` (rolling skew/kurtosis of `log_ret_1d`).

Cross-sectional `*_pct` parity added in `cross_sectional.py::_RANK_COLS`
for `rsi_5`, `bb_bandwidth`, `dist_vwap_20`, `ret_skew_20`. Net new
columns: 9 features + 4 ranks = 13.

## Plausible cause of the null result

The new features are **largely collinear with features the model already
has**, so they carry little orthogonal signal for a gradient-boosted tree:

* `rsi_5` / `rsi_2` ≈ `rsi_14` + the existing `delta_rsi_5d`.
* `bb_bandwidth` ≈ a rescaling of `vol_20` / `atr_pct_14`.
* `dist_vwap_5/20` ≈ `dist_sma_5/20` (VWAP ≈ volume-tilted SMA).
* `ret_skew/kurt` is the most orthogonal of the set, but on a ~0.52-AUC
  binarised triple-barrier problem its marginal lift is below noise.

Trees already learn the non-linear RSI thresholds (low/medium/high) that
motivated the additions, so explicit multi-period RSI is redundant.

## Decision

* **Artifact:** restored `data/models/universal_5d.{joblib,meta.json,
  contract.json}` and `data/lake/training/training_set.parquet` to the
  exact pre-experiment production state (oof_auc=0.5156). No promotion.
* **Source:** REVERTED. Because the A/B lift was null, the technical.py
  feature additions and the cross_sectional.py rank-parity commit were
  dropped rather than carried as dormant code. (The features were verified
  correct/ticker-invariant/always-available during the experiment — the
  reason for dropping is "no value", not "broken". They can be reinstated
  from tag `_tierb_backup` if a future non-GBM head changes the calculus.)
* **`build_training_set.py` collision fix kept regardless** — Tier A2 made
  `build_features_inline` emit `q_*`/`asof_date`, which collided with the
  lake quarterly merge on rebuild; the builder now drops the inline copies
  when the lake source is present. This is a real correctness fix
  independent of Tier B's outcome.

## Follow-up (deferred)

1. The features may still help a model that is NOT a GBM (e.g. the N-HiTS
   base in the stack, or a future linear/attention head) where collinearity
   is handled differently — worth re-checking if the stack is redesigned.
2. `ret_skew_20` could be a useful input to the **circuit-risk** model
   (crash-proneness ≈ negative skew); a targeted circuit-risk retrain that
   adds it is a cleaner experiment than the universal model.
3. A full news-included A/B (matching the 0.5156 lineage) would confirm the
   null result under production conditions, but the `--no-news` delta is
   already conclusive for these price-derived features.

## Tooling

```bash
# control
git stash push omni/features/technical.py omni/features/cross_sectional.py
python scripts/build_training_set.py --asset-classes india --no-news
python scripts/augment_triple_barrier.py
python scripts/train_universal_model.py --force --asset-classes india \
    --horizons 5 --ensemble single --target tb --cv purged --half-life-years 1.5
# treatment
git stash pop
# (repeat the three build/augment/train commands)
```
