# Phase 28.17 — modern cross-sectional ML practices (researched) + plan

Researched against current canonical repos/literature and mapped to concrete,
codebase-specific additions. Sources consulted:
- **Microsoft `qlib`** `qlib/data/dataset/processor.py` (fetched 2026-06): the
  reference cross-sectional pipeline — `RobustZScoreNorm` (median/MAD + outlier
  clip), `CSZScoreNorm` / `CSRankNorm` (per-`datetime` z-score / rank of
  features), `Fillna`. Explicit leakage guard: any *global* normalization's
  `fit_end_time` "must not include any information from the test data".
- **López de Prado, *Advances in Financial ML*** (mlfinlab): label
  **uniqueness / concurrency weighting**, purged CV (already in-repo), fractional
  differentiation.
- **Numerai** example pipeline: per-era (=per-date) feature neutralization +
  rank-gaussianize; optimize for cross-sectional rank correlation, not absolute.

## Where this codebase already aligns

- Purged k-fold + embargo + per-ticker purge (`cv/purged_kfold.py`) ✓
- Strict walk-forward option (`OMNI_CV_MODE=strict`, Phase 28.13) ✓
- Causal MAD-band OHLCV sanitizer (Phase 28.6) ✓ — same robust-stat idea as
  qlib RobustZScoreNorm
- Cross-sectional **z-score of fundamentals** within `(asset_class, run_date)`
  (`features/fundamental.py`) ✓
- Cross-sectional **rank** of a *subset* of features (`add_cross_sectional_ranks`
  → `*_pct`) ✓ — but only a hand-picked list, not systematic.

## Gaps vs. current best practice (prioritized)

1. **Systematic cross-sectional feature normalization (qlib CSRankNorm).**
   Today only a curated subset gets `*_pct`. The model would rank better if
   *every* unitless feature were expressed as its per-date cross-sectional
   percentile (leakage-free — uses only same-date peers). This is the single
   most-endorsed step for cross-sectional equity models (qlib default, Numerai).
   Lower risk than it sounds: additive `*_pct` columns, opt-in, A/B under strict
   WF on the cross-sectional L/S book.
2. **Label-uniqueness sample weighting (de Prado).** Overlapping h-day labels
   are co-dependent; weight each sample by average uniqueness so the loss isn't
   dominated by overlapping windows. The trainer has magnitude+recency weights
   but not uniqueness. Multiplies into the existing weight vector.
3. **Robust (median/MAD) global feature scaling fit on train-only.** For any
   non-cross-sectional features, qlib uses RobustZScoreNorm fit on the train
   window only. Must be wired through the walk-forward fold (fit per fold), not
   globally — otherwise it leaks (the exact trap the strict-WF work exposed).

## Decision / sequence

Implement #1 first (highest endorsement, leakage-free, cleanly A/B-able):
extend cross-sectional rank coverage to all eligible unitless features, gated so
production is unchanged until validated. Measure on the strict-WF cross-sectional
long/short book (the only honest metric, per Phase 28.13). Keep only if the IC /
net Sharpe improves; document either way. Then #2 (uniqueness weights), then #3.

Non-negotiable: every change validated under `OMNI_CV_MODE=strict` — purged-kfold
numbers are for AUC, not P&L (Phase 28.13).
