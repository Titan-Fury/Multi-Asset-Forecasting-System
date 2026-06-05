# Omni_Stocks — System Architecture

A multi-asset (India equities, US equities, crypto, FX) daily-direction
prediction system. Given a ticker, it predicts the probability and direction
of the next *h*-day move (h ∈ {1, 5, 21}), attaches an uncertainty band and a
position size, and produces a human-readable explanation with citations.

This document is the end-to-end map: data → features → models → ensemble →
serving → evaluation, with a glossary of every term and a section of
system-design talking points.

---

## 1. Design goals & constraints

| Goal | How it shapes the design |
| ---- | ------------------------ |
| **One model, many assets** | A single "universal" model pools all tickers/asset-classes. Every feature is a *unitless ratio or percentile* so a ₹100 stock, a $40,000 BTC and a 1.08 EURUSD live in the same feature space. Raw price levels are never features. |
| **No look-ahead / no leakage** | Causal features only; purged k-fold CV with an embargo; calibration fit only on past data; labels masked where they can't be realized. |
| **Calibrated probabilities, honest uncertainty** | A raw classifier score isn't a probability — it's calibrated (isotonic/beta) and wrapped in a conformal interval before anyone trades on it. |
| **Explainable** | Every prediction carries a 9-pillar rubric breakdown, provenance, and an LLM "why" grounded in retrieved news. |
| **Degrade gracefully** | Missing news, missing fundamentals, no GPU, or a flaky vendor feed must downgrade the output, never crash it. |

---

## 2. High-level flow

```
                         ┌──────────────────────────────────────────┐
   vendors (yfinance,    │              INGEST  (omni/ingest)         │
   news APIs, fundamentals) ─────►  prices · news · quarterly · universe
                         └───────────────┬────────────────────────────┘
                                         ▼
                         ┌──────────────────────────────────────────┐
                         │      LAKE  (omni/lake) — parquet tiers     │
                         │      hot (recent) / cold (history)         │
                         └───────────────┬────────────────────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────────────────┐
        │   FEATURES  (omni/features)                                      │
        │   sanitize OHLCV → technical · fundamental · sentiment ·         │
        │   regime-interactions → cross-sectional ranks → targets (labels) │
        │   guarded by a FEATURE CONTRACT (train/infer parity)             │
        └───────────────┬───────────────────────────────────┬─────────────┘
                         ▼ (training)                         ▼ (inference)
   ┌──────────────────────────────────┐      ┌───────────────────────────────┐
   │  TRAIN  (scripts/ + omni/models)  │      │  SERVE (omni/ondemand/predict) │
   │  purged-kfold walk-forward XGB    │      │  build_features_inline →       │
   │  → calibrate → stack → regime     │      │  universal model →             │
   │  stacks → meta-label → conformal  │      │  rubric → blend → gate →       │
   │  → persist artifacts + contract   │      │  conformal → Kelly → card      │
   └──────────────────────────────────┘      └───────────────┬───────────────┘
                                                              ▼
                                          ┌────────────────────────────────┐
                                          │ MONITORING + BACKTEST            │
                                          │ drift gate · daily_backtest ·    │
                                          │ run_backtest · calibration diag  │
                                          └────────────────────────────────┘
```

---

## 3. Data layer

**`omni/ingest`** — pulls four feeds:
- **prices** (`prices.py`): daily OHLCV + adjusted close (`yfinance`). Only does
  `dropna(adj_close)` at ingest; bad-tick cleaning happens at the feature funnel.
- **news**: headlines/descriptions tagged with a `ticker_hint` (the query symbol).
- **quarterly**: fundamentals (balance-sheet / income items) keyed by `asof_date`.
- **universe**: the tradable ticker list per asset class.

**`omni/lake`** — a local **parquet "lake"** with **tiers**: a *hot* tier (recent
bars, fast path) and a *cold* tier (full history). `tiers.load_hot(...)` is the
read API. This is a poor-man's feature store: columnar, append-friendly, no DB
server. Snowflake/Cortex is only a historical fallback LLM backend, not a
runtime dependency.

---

## 4. Feature engineering layer (`omni/features`)

Order matters and is identical for training and inference (the single funnel is
`build_features_inline` / `build.py`):

1. **Sanitize OHLCV** (`ingest/sanitize.py`) — *causal* bad-tick remover. Drops
   duplicate dates, impossible bars (`high<low`, non-positive prices), and
   extreme bars where `|log-return|` breaks **both** a trailing robust band
   (median ± 8·MAD over the prior 60 bars, all `shift(1)` so no future leak) and
   an absolute per-asset cap (india 0.35, us 0.40, crypto 0.70, fx 0.15). Runs
   before features/labels so one corrupt bar can't poison ~60 bars of rolling
   windows and the labels.
2. **Technical** (`technical.py`) — returns, volatility (vol_20, Parkinson,
   Garman-Klass), RSI, ATR%, distance-to-SMA, MACD, Bollinger, etc. **All
   unitless** (ratios / z-scores / percent), never raw price.
3. **Fundamental** (`fundamental.py`) — quarterly ratios, *winsorized* to clip
   outliers; point-in-time joined via `asof_date` (never uses a quarter before
   it was reported).
4. **Sentiment** (`sentiment.py`) — news scored per-article by **FinBERT**, then
   aggregated per (ticker, date): mean signed score, count, std, momentum, plus
   keyword flags (panic / promoter / SEBI). Lagged 1 trading day to avoid
   after-close leakage.
5. **Regime interactions** (`regime_interactions.py`) — features × regime state.
6. **Cross-sectional ranks** (`cross_sectional.py`) — for each (date,
   asset_class), convert chosen features to **percentile ranks** (`*_pct`).
   "RSI of 70" means little; "RSI in the 95th percentile of all stocks today"
   is a real cross-sectional signal. Computed *after pooling*, so a single-row
   inference call must reuse the trained distribution.
7. **Targets / labels** (`targets.py`) — see §5.1.

**Feature contract** (`contract.py`) — when a model is trained, the exact
`feature_cols` are written to a JSON sidecar next to the artifact. At inference
`assert_contract` checks the live feature frame has every column the model was
trained on (in order). This is the guardrail against silent train/serve skew —
the single most common production-ML bug.

---

## 5. Modeling layer — every term explained

### 5.1 Labels: Triple-Barrier (TB)
Instead of the naive label "did price go up after 5 days?", TB (de Prado) sets
three barriers from each bar: an upper (profit) barrier at `pt_mult × volatility`,
a lower (stop) barrier at `sl_mult × volatility`, and a vertical (time) barrier
at the horizon. The label is which barrier is hit first: `+1` (up), `-1` (down),
`0` (timed out). Why: it ignores tiny noise wiggles and only labels *moves big
enough to trade*, and the barriers scale with each name's own volatility. We
binarize as `(== +1)` for the up-probability model. Current params
(`pt_mult=1.5, sl_mult=1.0, vol_window=40`) yield ~33% positive.

### 5.2 Base learner: XGBoost
Gradient-boosted decision trees (`models/xgboost_5d.py`). Chosen over deep nets
because the data is tabular, medium-sized, and heterogeneous — GBMs dominate
there, train in seconds, and give feature importances. Key regularization:
shallow trees (`max_depth=6`), `min_child_weight=3`, L1/L2 (`reg_alpha=0.5`,
`reg_lambda=1.0`), column subsampling per-tree and **per-level**
(`colsample_bylevel=0.7`, to stop trees splitting repeatedly on the same
collinear volatility family), and **monotone constraints** (architectural
insurance that, e.g., a distress feature can only push probability one way).

### 5.3 Cross-validation: Purged K-Fold + Embargo
(`cv/purged_kfold.py`) Standard k-fold leaks in time series: a train row whose
5-day label overlaps a test row's window lets the model "see the answer".
**Purging** removes train rows whose label window overlaps the test fold;
**embargo** additionally drops the `horizon+1` days right after each test fold.
Purge is done *per-ticker* (a row for TCS at t-3 doesn't leak into a Reliance
row at t), which keeps ~30% more legitimate cross-ticker training rows.

### 5.4 Walk-forward fitting + sample weighting
(`walkforward_fit`) Expanding-window folds in time order. Two sample weights are
multiplied: **magnitude weights** (quantile-binned by `|forward return|` so the
model focuses on bars that actually moved, not noise) and **recency weights**
(exponential half-life, default 1.5y, so recent regimes matter more).

### 5.5 Calibration: isotonic / beta
(`calibration/factory.py`) A raw XGBoost score is not a true probability. Inside
*each fold*, the last 20% of the train window (the "inner-val") is held out and
a calibrator is fit on it, then applied to the test fold (leak-safe). **Isotonic**
= non-parametric monotone fit (good at large N); **beta** = 3-parameter
parametric fit (robust at small N). We auto-switch to beta when inner-val < 2000
rows because isotonic overfits there. Calibration *preserves the base rate* — a
33%-positive label correctly yields probabilities centered near 0.33.

### 5.6 Conformal intervals
(`models/conformal.py`) Split-conformal prediction turns a point estimate into a
*statistically valid* band: "the 5-day move is in [-2%, +4%] with 80% coverage."
It uses the empirical distribution of past residuals — distribution-free, with a
coverage guarantee. Kelly sizing reads its upside/downside off this band; a tight
band → more conviction → larger size.

### 5.7 HMM regime gating
(`regime/hmm.py`) A Hidden Markov Model labels each day's market *regime*
(e.g. trend / chop / high-vol) from soft state probabilities. Signal quality and
the model/rubric blend weights are conditioned on regime — the same setup is
weighted differently in a calm trend vs a volatile chop. `regime_stacks_h5.joblib`
holds regime-conditioned stack models.

### 5.8 Stacking
(`stack_h{1,5,21}.joblib`) A meta-model that combines several base signals
(per-horizon models, specialists) into one. Classic stacked generalization: base
models' out-of-fold predictions become features for the stacker, which learns how
to weight them.

### 5.9 Meta-labeling
(`modeling/meta_label.py`, `meta_h{1,5,21}.joblib`) de Prado's two-stage idea:
the *primary* model decides **direction**; a *secondary* (meta) model decides
**whether to act** on that call (a bet-sizing / precision filter). It learns "when
is the primary model trustworthy?" and gates low-confidence calls. (Note: its
value depends on the primary having real precision — see the 28.x notes.)

### 5.10 The Rubric — 9 pillars
(`rubric/scorer.py`, `rubric/learned.py`) A transparent, scored checklist run
alongside the ML model, each pillar 0–100 (50 = neutral): **technical,
fundamental, sentiment, macro, microstructure, open-interest (F&O),
relative-strength, insider, earnings-momentum**, aggregated to `rubric_total`. It encodes domain priors,
stays interpretable, and provides a fallback signal when the ML model is
uncertain or data is thin. `learned.py` can fit pillar weights from data.

### 5.11 Blend + Conviction Gate
(`ensemble/combine.py`, `ensemble/gate.py`)
- **Blend**: `blended_p = w_model·proba_up + w_rubric·(rubric_total/100)`, where
  the weights are **regime-conditioned** (soft HMM probabilities, no hard step at
  regime boundaries). **Direction** = sign of `blended_p` vs a neutral point;
  **conviction** = `|blended_p − 0.5|·200` (0–100).
  *Design note:* the neutral is currently 0.5, which assumes `proba_up` is
  ~0.5-centered — true for the deployed model but fragile if it were retrained on
  an imbalanced (e.g. 33%) target (documented in `phase_28_9`).
- **Gate**: emits a signal only if conviction ≥ threshold and the regime filter
  passes (`soft=True` for a smooth cutoff). Everything else is suppressed to
  "no-trade".

### 5.12 Position sizing: Kelly
(`ondemand/kelly.py`) The Kelly criterion sizes a bet from edge and odds. We use
**half-Kelly** (halve the fraction) — full Kelly is famously too aggressive and
sensitive to estimation error. Upside/downside come from the conformal band; the
fraction is capped. This is *risk management*, not signal generation.

---

## 6. Serving / inference (`omni/ondemand/predict.py`)

`predict_ticker(ticker, asset_class)` runs:
```
lookup_ticker            → resolve symbol, display name, asset class
build_features_inline    → same funnel as training (sanitize→…→ranks), contract-checked
universal model          → calibrated proba_up (per horizon)
rubric scorer            → 9 pillars → rubric_total
blend + gate             → blended_p, direction, conviction, passes_gate
conformal                → uncertainty band (+ optional direction override)
Kelly                    → position fraction
LLM "why"                → RAG-grounded explanation + citations
                         → PredictionCard (typed, with full provenance)
```
- **`deep_analysis`** is the heavy (~30s) variant: trains a per-ticker
  walk-forward XGBoost on that single name's history and uses the last fold's
  calibrator for the live row.
- **Caching** (`ondemand/cache.py`) with TTLs avoids recomputing models/features.
- **Model registry** (`model_registry.py`) loads the right artifact + contract
  per horizon, version-tagged so a redeploy busts the cache.
- The **PredictionCard** is the typed output contract: direction, conviction,
  blended_p, proba_up, pillar scores, regime, gate result, reasons, why-text +
  citations, provenance, circuit-risk, etc.

---

## 7. Evaluation, monitoring, robustness

- **`omni/backtest/runner.py`** — walk-forward backtest computing AUC, hit-rate,
  Brier, log-loss, top/bottom-decile precision & realized return, decile lift,
  gated metrics, and a shuffled-label baseline (edge over noise). Built-in
  leakage + chronology assertions.
- **`scripts/daily_backtest.py`** — scores *logged live predictions*
  (`data/lake/predictions/`) against realized returns: rolling AUC/hit-rate/Brier
  + **ADWIN drift detection** (`modeling/drift_gate.py`).
- **`scripts/diagnose_calibration.py` / `monitor_drift.py` /
  `check_data_quality.py` / `check_sentiment_health.py`** — operational health
  probes.
- **Coverage haircut** — predictions on tickers with thin feature coverage get a
  conviction penalty (don't over-trust a half-empty feature row).

---

## 8. Training pipeline (scripts)

`build_training_set.py` (pool features, dedup, cross-sectional ranks) →
`augment_triple_barrier.py` (TB labels) → `train_universal_model.py` (per-horizon
base + calibration + contract) → `train_regime_stacks.py` / `train_tb_and_meta.py`
/ `train_multi_horizon_meta.py` (stacks + meta) → `train_conformal.py` (intervals)
→ `train_circuit_risk.py` (India lower-circuit risk) → `train_learned_rubric.py`
(pillar weights). `retrain_all.py` / `run_daily.py` orchestrate.

---

## 9. Cross-cutting principles

- **Train/serve parity** — one feature funnel, enforced by the feature contract.
- **Leak discipline** — causal sanitize, purged+embargoed CV, point-in-time
  fundamentals, 1-day sentiment lag, in-fold calibration.
- **Ticker invariance** — unitless features + cross-sectional ranks so one model
  spans four asset classes and any price level.
- **Graceful degradation** — env gates (`OMNI_DISABLE_OHLCV_SANITIZE`,
  `OMNI_XGB_DEVICE=cuda`, calibration method, hyperparameter overrides); every
  external dependency (news, GPU, FinBERT, RAG) has a fallback.
- **Provenance + explainability** — rubric breakdown + RAG-cited why-text on
  every card.

---

## 10. System-design talking points (CTO round)

**Why this architecture?**
- *Single universal model vs per-ticker models*: pooling shares statistical
  strength across thousands of names and avoids 1000s of fragile tiny models;
  unitless features make it valid. Per-ticker `deep_analysis` is the opt-in
  escape hatch.
- *GBM vs deep learning*: tabular + medium data + need for fast retrains and
  feature importances → XGBoost is the right tool; DL would overfit and obscure.
- *Probabilities, not scores*: calibration + conformal turn a number into a
  decision-ready probability with a coverage guarantee — essential before sizing.
- *Transparent rubric beside the black box*: regulators/PMs need a "why";
  the rubric also stabilizes the signal when ML is uncertain.

**Hard problems solved**
- Time-series leakage (purge + embargo + in-fold calibration + PIT joins).
- Train/serve skew (feature contract sidecar).
- Multi-asset unit mismatch (ratios + cross-sectional ranks).
- Imbalanced labels (TB + magnitude/recency weighting); and the documented
  base-rate-vs-0.5-neutral fragility in the blend.

**Known limitations / honest gaps**
- Base signal edge is marginal on liquid large-caps (daily direction is near
  efficient); realized decile spread ≈ 0 — the value is in the conviction layer.
- News coverage is thin (volume, not relevance, is the gap).
- `run_backtest` can't yet measure the *deployed* gated edge without a base-rate
  fix or a larger logged-prediction sample.

**What I'd do next**
- Leave-one-layer-out ablation to prove which of {stack, regime, meta, conformal}
  actually add edge, and delete the inert ones (complexity you can't measure is
  complexity you can't justify).
- Make `blend()` neutral base-rate-aware (robustness).
- Grow news ingestion volume; add transaction-cost-aware backtest (Sharpe / max
  drawdown / turnover), which the current harness lacks.

---

## 11. Glossary (quick reference)

| Term | One-line meaning |
| ---- | ---------------- |
| **OHLCV** | Open/High/Low/Close/Volume daily bar. |
| **Adjusted close** | Close adjusted for splits/dividends. |
| **Triple-barrier (TB)** | Label = which of {profit, stop, time} barrier hits first; volatility-scaled. |
| **Purged k-fold** | CV that removes train rows whose label window overlaps the test fold. |
| **Embargo** | Extra gap (horizon+1 days) dropped after each test fold to kill boundary leakage. |
| **Walk-forward** | Expanding-window, time-ordered training/eval. |
| **Isotonic / beta calibration** | Map raw scores → true probabilities (non-parametric / 3-param). |
| **Conformal interval** | Distribution-free prediction band with a coverage guarantee. |
| **HMM regime** | Hidden-Markov latent market state (trend/chop/high-vol). |
| **Stacking** | Meta-model combining base models' out-of-fold predictions. |
| **Meta-labeling** | Secondary model deciding *whether to act* on the primary's direction call. |
| **Rubric** | 9-pillar interpretable 0–100 scorecard run beside the ML model. |
| **Blend** | Regime-weighted mix of model probability and rubric score. |
| **Conviction gate** | Emit a signal only above a conviction threshold + regime filter. |
| **Kelly (half)** | Bet-sizing from edge/odds; halved for safety. |
| **Feature contract** | JSON of trained feature columns enforced at inference. |
| **Cross-sectional rank** | A feature's percentile within (date, asset_class). |
| **Magnitude / recency weights** | Up-weight big-move bars / recent bars in training. |
| **ADWIN drift** | Online detector that fires when live accuracy degrades. |
| **Circuit risk** | India-specific lower-circuit (limit-down) probability model. |
| **PIT (point-in-time)** | Joining fundamentals only as of when they were actually reported. |
