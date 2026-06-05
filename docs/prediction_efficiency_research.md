# Prediction-efficiency research — how to make the signal actually better

Companion to the Phase 28.7–28.10 notes. This is a prioritized, evidence-based
menu of ways to raise predictive edge (and, secondarily, compute efficiency),
specific to Omni_Stocks — not generic ML advice.

## The core diagnosis (what the data says)

From this session's measurements:
- Raw model OOF AUC ≈ 0.51–0.53; **realized-return decile spread ≈ 0** (28.8).
  The most-confident-up bars don't outperform the least-confident ones.
- After fixing the direction-bias bug (28.10), gated directional hit-rate is
  0.514 but mean signed return is still slightly negative.

Conclusion: the system is technically excellent (calibration, purged CV, regime
gating, conformal, contracts) but the **base signal carries almost no edge on a
liquid NIFTY-large-cap universe**, because **absolute daily/weekly direction on
liquid names is close to efficient**. The biggest gains will not come from more
layers on top — they come from changing *what we predict* and *how we measure
it*. Tiers below are ordered by expected leverage.

---

## Tier 0 — measurement first (do this before anything else)

You cannot improve what you cannot measure. The backtest currently reports
AUC/hit-rate but **no Sharpe, turnover, transaction costs, or max drawdown**
(confirmed: absent from `omni/backtest/metrics.py`). Every idea below is
unprovable without this.

- **Cost- and turnover-aware portfolio backtest.** Turn gated signals into a
  daily long/short book; subtract realistic costs (India ~10–20 bps round-trip
  incl. STT/impact); report **annualized return, Sharpe, Sortino, max drawdown,
  turnover, hit-rate, profit factor**, and an **edge-over-shuffled-baseline**.
- **Leave-one-layer-out ablation.** Strip each of {stack, regime, meta-label,
  conformal, rubric} and measure Sharpe delta. Almost certainly 1–2 layers
  carry the weight and 1–2 are inert — delete the inert ones. *Complexity you
  can't measure is complexity you can't justify.*
- Grow the **logged-prediction** store (only 480 rows/15 dates today) so live
  validation is statistically meaningful.

**Impact: foundational. Effort: medium. Risk: none (eval-only).**

---

## Tier 1 — reframe the target (highest signal leverage)

The model predicts **absolute** per-ticker direction. The features already
include cross-sectional percentile ranks, but the *label* is absolute. This is
the central missed opportunity.

1. **Cross-sectional ranking / learning-to-rank.** Predict *relative* ordering:
   "which names will outperform the cross-section over the next h days?" Trade
   top-decile vs bottom-decile. This **cancels the unpredictable market beta**
   and isolates the learnable idiosyncratic component — the standard quant-equity
   formulation. Implement either as a ranking objective (XGBoost
   `rank:pairwise`/`rank:ndcg`) or, simplest, keep XGBoost classification but set
   the **label to "beats the cross-sectional median return that day"** instead of
   "return > 0". Reuses the entire existing pipeline; only `targets.py` changes.
2. **Residualized (market/sector-neutral) returns.** Regress each name's return
   on the index (and sector) return; predict the **residual**. Removes the
   dominant common factor that is essentially unpredictable day-to-day. The
   `pair_signal.py` module already does residual-alpha thinking for pairs — lift
   that idea to the universal target.
3. **Event-conditioning.** Edge is concentrated around events. Post-earnings
   announcement drift (PEAD) and earnings-surprise momentum are among the most
   robust documented anomalies. A model that only fires in the N days after an
   earnings event (or weights those rows up) will likely beat an always-on model.

**Impact: high (this is where real alpha lives). Effort: medium. Risk: medium
(needs the Tier-0 harness to validate).**

---

## Tier 2 — richer signal content (features)

4. **Explicit factor exposures.** Add value / momentum / quality / low-vol /
   size factor loadings as features (quality.py already has Piotroski-style
   pieces). These are *documented* return drivers; the model can learn
   regime-dependent factor timing.
5. **Cross-sectional & industry momentum/reversal.** 1m–12m relative-strength,
   industry-relative momentum, short-horizon reversal — formalize beyond the
   single RS rubric pillar.
6. **Flow / microstructure depth.** You already ingest OI and FII/DII and NSE
   bhavcopy (delivery %). Expand: delivery-ratio trends, OI-build vs price,
   institutional-vs-retail delivery — this is where a non-HFT edge realistically
   exists on Indian equities.
7. **News *volume*, not relevance.** The 28.x research showed the news lake is
   ~1,091 articles / 10 tickers — far too sparse. Broaden ingestion (more
   sources, full-universe backfill, longer history) before any relevance
   filtering. Volume is the binding constraint on the sentiment pillar.

**Impact: medium–high. Effort: medium (mostly ingestion). Risk: low.**

---

## Tier 3 — training & validation rigor

8. **Label-uniqueness (concurrency) weighting.** Overlapping h-day labels mean
   adjacent rows share outcome information; de Prado's *average uniqueness*
   down-weights overlapping samples. Currently training uses magnitude + recency
   weights but **not** uniqueness — adding it reduces effective over-counting and
   usually improves OOF generalization.
9. **Combinatorial Purged CV (CPCV).** Stronger than single purged k-fold for
   estimating the *distribution* of backtest outcomes and detecting overfit;
   pairs naturally with the existing `purged_kfold` machinery.
10. **Model diversity in the stack.** Add LightGBM / CatBoost and seed-ensembling
    next to XGBoost; decorrelated learners in the existing stack usually lift the
    ensemble more than tuning a single learner.
11. **Triple-barrier tuning by *trading* metric, not AUC.** 28.7/28.8 showed AUC
    is confounded by label balance. Tune `pt_mult`/`sl_mult`/`vol_window` against
    the Tier-0 Sharpe, not OOF AUC.

**Impact: medium. Effort: low–medium. Risk: low.**

---

## Tier 4 — computational efficiency (if "efficiency" = speed/cost)

12. **GPU training** — `OMNI_XGB_DEVICE=cuda` hook already exists; use it for the
    5-fold walk-forward (10–50× on the FinBERT + XGB hot paths).
13. **Feature & layer pruning** — drop features XGBoost never splits on and the
    layers the Tier-0 ablation proves inert; less compute, same signal, simpler
    contract.
14. **Lazy/columnar IO** — `scan_lazy` over the parquet lake, only materialize
    needed columns; cache (already present) keyed by model-version tag.
15. **Batch inference** — FinBERT batch scoring already exists; extend batching to
    the per-ticker feature build for universe-wide nightly runs.

**Impact: cost/latency only (not edge). Effort: low. Risk: low.**

---

## Prioritized roadmap

| Rank | Change | Lever | Impact | Effort |
| ---- | ------ | ----- | ------ | ------ |
| 1 | Cost/Sharpe/turnover backtest + ablation | measure | foundational | M |
| 2 | Cross-sectional ranking target | edge | high | M |
| 3 | Residualized (market-neutral) returns | edge | high | M |
| 4 | News-volume ingestion | edge | med-high | M |
| 5 | Flow/microstructure & factor features | edge | medium | M |
| 6 | Uniqueness weighting + CPCV | rigor | medium | L |
| 7 | LightGBM/CatBoost stack diversity | edge | medium | L |
| 8 | GPU + feature/layer pruning | speed | n/a (cost) | L |

## The one-line takeaway

The stack is already sophisticated; the missing ingredient is **edge in the
base signal**, and the highest-leverage moves are (1) build a cost-aware
Sharpe backtest so improvements are provable, then (2) switch the prediction
target from *absolute direction* to *cross-sectional / market-neutral relative
return*, where learnable equity alpha actually lives. Everything else is
incremental until those two are done.
