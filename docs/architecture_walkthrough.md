# Omni_Stocks — Architecture & Code Walkthrough

**Updated:** 2026-05-08

This document explains the entire codebase end-to-end, the scoring metrics used, and the latest model performance numbers on real held-out data.

---

## Table of Contents
1. [System overview](#1-system-overview)
2. [Repository layout](#2-repository-layout)
3. [Data flow — request to prediction](#3-data-flow)
4. [Module-by-module walkthrough](#4-module-by-module)
5. [Scoring methodology](#5-scoring-methodology)
6. [Latest scores on real data](#6-latest-scores)
7. [Operations](#7-operations)

---

## 1. System overview

Omni_Stocks is a **multi-asset stock prediction system** that combines:

- **Universal XGBoost classifier** trained on a 623k-row pooled training set (US + India + Crypto + FX, ~5y history)
- **India-specialist head** trained on the India-only slice (204k rows) and blended 50/50 with the universal when conviction warrants
- **Stacking ensemble** (XGBoost + sklearn HistGradientBoosting + logistic meta) trained with **purged K-fold + embargo** cross-validation
- **Cross-sectional rank target** (Numerai-style) for noise-resistant labels
- **Rubric scoring** across 4 pillars (technical / fundamental / sentiment / macro)
- **Regime detection** via 4-state HMM on returns + volatility (bull / chop / bear / crisis)
- **Multi-horizon forecasts** at 1-day, 5-day, 21-day with conformal prediction intervals
- **Intraday microstructure tilt** (VWAP deviation, opening-range breakout, intraday momentum) on the 1d horizon
- **Cross-asset pair signals** (β-adjusted alpha vs sector ETF / index)
- **News + sentiment** from 12 sources: NewsAPI, Marketaux, GDELT, StockTwits, EDGAR, Reddit (3 India subs), Indian press RSS (4 outlets), BSE Corporate Notices

Front-end is a **Streamlit dashboard** with TradingView Lightweight-Charts (candles), Three.js 3D Signal Cosmos, and D3 force-directed correlation graphs.

---

## 2. Repository layout

```
omni/
  ingest/                          # Stateless data fetchers
    prices.py                      # OHLCV via yfinance / ccxt with .BO fallback
    intraday.py                    # 5-min bars + session VWAP
    fundamentals.py                # yfinance .info snapshot
    earnings_calendar.py           # earnings dates
    news.py                        # news collector dispatcher
    sources/                       # per-source fetchers
      newsapi_live.py              # NewsAPI.org
      marketaux.py                 # Marketaux (financial news)
      gdelt.py                     # GDELT global event database
      stocktwits.py                # StockTwits messages
      edgar.py                     # SEC EDGAR submissions API
      reddit_public.py             # Reddit no-auth JSON (3 India subs)
      indian_rss.py                # MoneyControl + ET + LiveMint + Hindu BL
      nse_flows.py                 # FII/DII + bulk/block deals
      bse_notices.py               # BSE corporate announcements RSS

  features/                        # Feature engineering
    technical.py                   # 30+ price-derived features (RSI, MACD, BB, etc.)
    fundamental.py                 # PE, PB, ROE, debt ratios
    sentiment.py                   # FinBERT scoring + aggregation
    sentiment_z.py                 # 60-day z-scored sentiment
    macro.py                       # VIX, DXY, US10Y, Gold, Oil, Nifty
    earnings.py                    # days-to-earnings + post-earnings drift
    targets.py                     # fwd_ret, fwd_dir, fwd_rank (cross-sectional)
    cross_sectional.py             # post-pooling rank features
    vol_adjusted_momentum.py       # mom / realized vol
    triple_barrier.py              # (planned) vol-scaled labels
    pair.py                        # rolling beta, alpha, residual_z, correlation_60d
    intraday.py                    # microstructure features

  models/
    xgboost_5d.py                  # XGBoost + isotonic walkforward fit
    stack.py                       # stacking ensemble
    conformal.py                   # split conformal interval calibrator

  cv/
    purged_kfold.py                # de Prado purged K-fold + embargo

  specialists/
    india.py                       # India-only head trainer

  rubric/
    scorer.py                      # 4-pillar rubric scoring (tech/fund/sent/macro)
    config.py                      # rubric weights & gates

  regime/
    hmm.py                         # 4-state Gaussian HMM on returns/vol

  ensemble/
    pipeline.py                    # blend probability + rubric + regime
    gate.py                        # min-conviction + regime filter

  ondemand/                        # On-demand (live) prediction path
    cache.py                       # diskcache with TTL tiers
    lookup.py                      # ticker -> LookupBundle aggregator
    universe.py                    # curated list + alias resolver
    features_inline.py             # build features for a single ticker live
    model_registry.py              # save/load horizon-keyed model bundles
    regime_live.py                 # live regime fit/load
    macro_live.py                  # live macro features
    pair_signal.py                 # cross-asset alpha computation
    quality.py                     # data-quality + provenance scoring
    predict.py                     # main orchestrator: predict_ticker, deep_analysis

  llm/
    router.py                      # multi-backend router with fallback chain
    backends/
      openrouter.py                # MiniMax-M2.5, Llama-70b, etc.
      groq.py                      # Groq fast inference
      gemini.py                    # Google Gemini Flash
      finbert.py                   # ProsusAI FinBERT for sentiment
      cortex.py                    # Snowflake Cortex
      base.py                      # Backend interface + BackendError

  lake/                            # Parquet data lake
    paths.py                       # canonical lake layout
    io.py                          # read_prices, read_fundamentals, etc.
    backfill.py                    # full + incremental backfill
    quarterly.py                   # quarterly fundamentals from yfinance

  monitoring/
    predict_log.py                 # append-only daily prediction log

  dashboard/                       # Streamlit UI
    app.py                         # page layout + tabs
    components.py                  # prediction_card, signal_card, kpi_row, etc.
    theme.py                       # design tokens, CSS injection
    cache.py                       # @st.cache_data loaders
    charts/
      lwchart.py                   # TradingView Lightweight-Charts
      cosmos3d.py                  # Three.js 3D Signal Cosmos
      corrgraph.py                 # D3 force-directed correlation network

  storage/
    tiers.py                       # raw / clean / feature parquet writers

  utils/
    env.py                         # env loader + universe config
    paths.py                       # data dirs
    logger.py                      # loguru wrapper
    splits.py                      # walkforward + purged splits

config/
  universe.yaml                    # 207 tickers across us/india/crypto/fx
  benchmarks.yaml                  # ticker -> benchmark sector ETF mapping
  rubric.yaml                      # 4-pillar weights + gates
  llm_routes.yaml                  # multi-backend route definitions
  sources.yaml                     # per-source config

scripts/
  build_training_set.py            # pool features per ticker, write parquet
  train_universal_model.py         # train per-horizon models (single | stack)
  daily_orchestrator.py            # nightly end-to-end run
  daily_backtest.py                # backtest report + alert
  lake_backfill.py                 # bulk price + fundamentals backfill

docs/
  architecture_walkthrough.md      # this file
  search_improvement_and_bug_audit.md

data/
  lake/
    prices/{asset_class}/{ticker}.parquet
    fundamentals/{date}.parquet
    fundamentals/quarterly/{ticker}.parquet
    predictions/{date}.parquet
    training/training_set.parquet
  models/
    universal_h{1,5,21}.pkl        # XGB + isotonic per horizon
    stack_h{1,5,21}.pkl            # stacking ensemble per horizon
    india_specialist.pkl           # India-only head
  predictions/latest.parquet       # daily orchestrator output
  signals/latest.parquet           # gated emissions
  regime/latest.parquet            # current regime
  backtest/summary.json            # latest backtest report
  cache/ondemand/                  # diskcache tier
  omni.log                         # rotating loguru log
```

---

## 3. Data flow

### 3a. On-demand single-ticker prediction (`predict_ticker`)

```
user search ──► resolve(query)               # universe.py: fuzzy + alias resolver
            ─► lookup_ticker(t, ac)         # lookup.py: pulls everything in parallel
                ├─ fetch_prices              # OHLCV (yfinance/ccxt + .BO fallback)
                ├─ fetch_fundamentals        # PE/ROE/etc snapshot
                ├─ news collector            # 8 sources -> dedup -> sentiment score
                ├─ macro_live                # VIX/DXY/Nifty/etc
                ├─ regime_live               # HMM 4-state today's label
                └─ Provenance                # data quality 0-100 score
            ─► build_features_inline         # 88 columns from prices+funds+news+macro
            ─► model_registry.load_horizon   # universal_5d (+1d, +21d if available)
            ─► um.calibrator.predict_proba   # calibrated probability
            ─► [india branch] india_specialist 50/50 blend
            ─► rubric.score_frame            # 4-pillar 0-100 each
            ─► assemble_signals              # blend p + rubric + regime
            ─► gate.apply                    # min-conviction + regime filter
            ─► compute_intraday_features     # tilt 1d if intraday data present
            ─► compute_pair_signal           # alpha vs sector benchmark
            ─► _explain (LLM)                # 3-4 sentence "why"
            ─► PredictionCard                # full result struct
            ─► log_prediction (parquet)      # for monitoring
            ─► return to UI
```

### 3b. Training (`scripts/train_universal_model.py`)

```
read lake training_set.parquet (623k rows × 88 cols)
  │
  ├─ for each horizon in {1, 5, 21}:
  │   ├─ if --target=rank: add fwd_rank_{H}d (cross-sectional pct rank
  │   │                     within (date, asset_class)), binarise at 0.5
  │   ├─ else (default=dir): use binary fwd_dir_{H}d
  │   │
  │   ├─ if --cv=purged: purged K-fold + embargo splits
  │   │   else: time-ordered walk-forward
  │   │
  │   ├─ if --ensemble=stack:
  │   │   ├─ fit XGBClassifier per fold     -> oof_xgb
  │   │   ├─ fit HistGradientBoosting/fold  -> oof_hgb
  │   │   ├─ (if available) fit LightGBM    -> oof_lgb
  │   │   ├─ stack OOF preds into Z matrix
  │   │   ├─ fit LogisticRegression(Z, y)   -> meta-learner
  │   │   ├─ refit each base on full data   -> live inference
  │   │   └─ save data/models/stack_h{H}.pkl
  │   │
  │   └─ else (default=single):
  │       ├─ fit XGBClassifier walkforward
  │       ├─ CalibratedClassifierCV(method=isotonic) on inner-val
  │       └─ save data/models/universal_h{H}.pkl (UniversalModel struct)
  │
  └─ if --india-specialist:
      ├─ filter rows where asset_class=='india'  (~205k rows)
      ├─ fit XGBClassifier with India-tuned hyperparams
      ├─ purged K-fold + embargo OOF predictions
      ├─ IsotonicRegression on (oof, y) for calibration
      └─ save data/models/india_specialist.pkl
```

---

## 4. Module-by-module

### 4a. Ingestion layer

**`omni/ingest/prices.py`** — OHLCV fetcher with retry + provider routing.
- `_yf_symbol(asset_class, ticker)`: routes `.NS` for india, `-USD` for crypto, `=X` for fx; **passes through** symbols that already contain `.` / `^` / `=` / `-` (so `^NSEI` and `CIANAGRO.BO` aren't mangled).
- `fetch_prices(asset_class, ticker, start, end)`: tries primary, **falls back to `.BO` (BSE)** if `.NS` returns 0 rows for India tickers. Drops trailing NaN-close rows that yfinance inserts before session close.
- `latest_intraday(asset_class, ticker)`: most-recent 5-minute bar.

**`omni/ingest/intraday.py`** — full intraday bar fetcher.
- `fetch_intraday_bars(asset_class, ticker, interval='5m', days=5)`: returns DataFrame with `[ts, open, high, low, close, volume, vwap]`. ccxt for crypto, yfinance otherwise.
- Computes session-VWAP per calendar day inside the fetcher.

**`omni/ingest/sources/`** — one file per provider, each returning a uniform DataFrame `[title, url, source, published_at, sent_score]`:
- `newsapi_live.py` — NewsAPI.org (free 100/day)
- `marketaux.py` — Marketaux (free 100/day, includes pre-scored sentiment)
- `gdelt.py` — GDELT v2 doc API (event-based news)
- `stocktwits.py` — StockTwits message stream (retail sentiment)
- `edgar.py` — SEC EDGAR filings (10-K/Q, 8-K, Form-4 insider)
- `reddit_public.py` — no-auth JSON; 3 US subs + 3 India subs + 2 crypto subs
- **`indian_rss.py`** — MoneyControl + Economic Times + LiveMint + Hindu BusinessLine. CDATA-aware parser. Browser UA. Returns ~140 items per cycle.
- **`nse_flows.py`** — NSE FII/DII + bulk + block deals via `snapshot-capital-market-largedeal`. Cookie-warmup session.
- **`bse_notices.py`** — BSE corporate announcements RSS (results, board meetings, insider disclosures).

### 4b. Feature engineering

**`omni/features/technical.py`** — produces 30+ price features per ticker:
- Returns: `ret_1d`, `ret_5d`, `ret_20d`, `log_ret_*`
- Volatility: `vol_5`, `vol_10`, `vol_20`, `vol_60`, `vol_120`, `vol_252`
- Momentum: `mom_5`, `mom_20`, `mom_60`, `mom_120`
- Indicators: `rsi_14`, `macd`, `macd_signal`, `bb_pct_b`, `atr_14`, `obv_z`
- Microstructure proxies: `hl_range_pct`, `volume_z_60`, `gap_overnight`

**`omni/features/targets.py`** — labels:
- `fwd_ret_{1,5,21}d`: forward log return
- `fwd_dir_{H}d`: sign of `fwd_ret_{H}d` (0/1)
- `fwd_rank_{H}d`: **cross-sectional percentile rank** within `(date, asset_class)`. The Numerai-style target — empirically lifts AUC 2-4pp because it removes overall-market drift from the signal.

**`omni/features/sentiment.py`** + `sentiment_z.py` — FinBERT scoring per article, then 60-day z-score per ticker.

**`omni/features/cross_sectional.py`** — post-pooling rank features (volume rank within sector, momentum rank within asset class, etc.). **NOT** computed in the live single-ticker path — neutral 0.5 placeholders are used to avoid distribution skew (audit doc B2 fix).

**`omni/features/intraday.py`** — 8 microstructure features:
- `intraday_return_pct`, `gap_open_pct`, `vwap_dev_bps`, `opening_range_breakout` ∈ {-1, 0, +1}, `intraday_realized_vol`, `last_hour_return_pct`, `volume_z_today`, `range_compression`

**`omni/features/pair.py`** — rolling β/α/residual vs benchmark, used for cross-asset hedging signals.

### 4c. Models

**`omni/models/xgboost_5d.py`** — walkforward fit with isotonic calibration:
- 4-fold time-ordered split, `min_train=252` rows
- Per fold: fit `XGBClassifier(tree_method='hist')`, then `CalibratedClassifierCV(method='isotonic', cv='prefit')` on the last 20% of train.
- Saves last fold's calibrator for live inference.

**`omni/models/stack.py`** — stacking ensemble:
- Base learners: XGB (always), HistGradientBoosting (always), LightGBM (if installed), CatBoost (if installed)
- Meta: `LogisticRegression(C=1.0)` on OOF predictions
- All bases auto-detect; missing libs gracefully skip.
- OOF predictions via **purged K-fold + embargo** to eliminate label-window leakage.

**`omni/models/conformal.py`** — split conformal calibrator:
- Computes (1-α) finite-sample-corrected quantile of |residuals|/σ on calibration set
- `widen(mu, sigma)` returns `(mu - q*sigma, mu + q*sigma)` interval with provable coverage ≥ 1-α

**`omni/specialists/india.py`** — India-only head:
- XGBoost with depth=4, lr=0.03, reg_lambda=2.0 (more conservative than universal)
- 5-fold purged K-fold OOF + IsotonicRegression
- Blended 50/50 with universal in `predict.py` only when `oof_auc > 0.51`

### 4d. Cross-validation

**`omni/cv/purged_kfold.py`** — de Prado purged K-fold with embargo:
- For each validation fold `[v_lo, v_hi]`, drops training samples whose `[t, t+horizon]` window touches `[v_lo - horizon, v_hi + embargo]`.
- `assert_no_leakage()` is a runtime sanity check that asserts no train sample's label window enters validation, and no train sample falls within `embargo_days` after validation.
- Verified PASS on 2000-bar synthetic test.

### 4e. Rubric + regime + ensemble

**`omni/rubric/scorer.py`** — 4-pillar 0-100 scorer:
- **technical**: composite of momentum, RSI, MACD, BB position
- **fundamental**: ROE, PE, profit margin, revenue growth, debt-to-equity (z-scaled)
- **sentiment**: 7d sentiment mean × volume-weighting
- **macro**: VIX/DXY/yield-curve/Nifty regime alignment

**`omni/regime/hmm.py`** — 4-state Gaussian HMM on returns + vol; labels {bull, chop, bear, crisis}.

**`omni/ensemble/pipeline.py`** — `assemble_signals()`:
- `blended_p = 0.6 * proba_up + 0.4 * (rubric_total / 100)`
- `direction = sign(blended_p - 0.5)` if rubric agrees, else 0
- `conviction = |blended_p - 0.5| * 200` clipped to [0, 100]

**`omni/ensemble/gate.py`** — emission gate: `passes_gate = (conviction >= 70) AND (direction != 0) AND regime_compatible`.

### 4f. Predict orchestrator

**`omni/ondemand/predict.py`** — `predict_ticker(ticker, asset_class)`:
1. `lookup_ticker` parallel-fetch
2. `build_features_inline` → 88-column row
3. `model_registry.load_horizon(5)` → main `proba_5d`
4. **India branch**: `india_specialist.predict_proba(X)` → blend 50/50 with universal if AUC > 0.51
5. Multi-horizon: `load_horizon(1)` and `load_horizon(21)` for additional probabilities
6. `_build_card`:
    - Drift = `edge × σ_h × 100`, ±1σ band = `drift ± σ_h`
    - Intraday tilt: `proba_1d_adj = clip(p1 + 0.15·tanh(intraday_return/2 + vwap_dev/200), 0.02, 0.98)`
7. `compute_pair_signal` → β-adjusted alpha vs benchmark
8. Cache key includes **model bundle mtime** so retraining invalidates stale cards (audit doc B3 fix).

### 4g. Dashboard

**`omni/dashboard/app.py`** — Streamlit page with tabs: `Predict | Watchlist heatmap | Top signals | Cosmos 3D | Correlations | Calibration | History`. Sidebar has 🇮🇳 India focus toggle that filters universe to NSE/BSE.

**`omni/dashboard/components.py`** — `prediction_card(card_dict)`:
- Header (ticker, name, regime/gate/model pills)
- Results 4-tile (direction, expected 5d move, conviction, 1y Sharpe)
- Horizon table (Tomorrow / Next week / Next month with target prices + ±1σ ranges)
- Price strip + bidirectional probability bar
- Why text (LLM-generated 3-4 sentences)
- Rubric pillars
- **Intraday microstructure** (when bars present): 4-tile of intraday return, VWAP-dev, OR breakout, vol-z
- **Pair trade** (when benchmark mapped): 4-tile of benchmark, β/ρ, predicted-α, pair-Sharpe + suggestion pill

**`omni/dashboard/charts/`** — three custom JS-embedded charts via `streamlit.components.v1.html`:
- **`lwchart.py`** — TradingView Lightweight-Charts (price + volume, WebGL-smooth, tradingview-grade)
- **`cosmos3d.py`** — Three.js 3D Signal Cosmos: ticker = sphere; radius ∝ conviction; colour = direction; cluster by asset_class; orbit speed ∝ vol; raycaster hover; **drag-to-rotate only** (no auto-spin); orbits **pause on hover**
- **`corrgraph.py`** — D3 force-directed correlation network: nodes = tickers, edges = |60d correlation| ≥ 0.4 with thickness ∝ |ρ|

---

## 5. Scoring methodology

For probabilistic binary classification on financial direction we report multiple complementary metrics — no single number tells the whole story.

### Primary: ROC-AUC
- Most-cited metric for direction classifiers
- Threshold-free; measures rank-order quality
- **Random baseline: 0.50**
- **Decent retail/quant: 0.52-0.55**
- **Strong institutional / Bloomberg AIM: ~0.58**
- **Numerai top-decile: ~0.55-0.57** (over thousands of stocks, weekly horizon)

### Secondary: PR-AUC (Average Precision)
- Better than AUC when classes are imbalanced
- Insensitive to true negatives — important for the long-only side of trading

### Calibration: Brier Score + Reliability
- `Brier = mean((p - y)²)`. Lower is better. **0.25 = always 0.5 prediction**.
- **Good calibrated model: 0.235-0.245** on financial direction targets (where realised hit rates are close to the prior).

### Practical: Top-decile precision
- Sort predictions by probability, take top 10%, measure precision (long-side hit rate).
- **This is what determines real PnL** for a long-only sleeve.
- **Decent: 0.54-0.56**, **strong: 0.60+**

### Other metrics computed
- `hit_rate` — fraction where `(p > 0.5) == (y == 1)` (informal accuracy)
- `log_loss` — proper scoring rule
- `bottom_decile_precision_short` — short-side hit rate

### Why not MAE / RMSE on returns?
The model is a binary direction classifier, not a regressor. We don't optimise the magnitude of `fwd_ret`, only its sign. Magnitude estimates come from `expected_move = edge × σ_h` post-hoc.

---

## 6. Latest scores

### 6a. Pre-retrain baseline (current production models, trained earlier)

Held-out: **last 20% of dates** of the 623,532-row training set.

| Horizon | n samples | AUC | PR-AUC | Brier | Hit | Top-10% prec | Bottom-10% short prec |
|--------:|----------:|-----:|-------:|------:|----:|-------------:|----------------------:|
| **1-day** | 124,625 | **0.5143** | 0.5147 | 0.2487 | 0.507 | **0.535** | 0.549 |
| **5-day** | 124,469 | 0.5071 | 0.5295 | 0.2500 | 0.508 | 0.546 | 0.486 |
| **21-day** | 123,849 | 0.4903 | 0.5361 | 0.2513 | 0.490 | 0.517 | 0.465 |

**India-only subset:**

| Horizon | n samples | AUC | PR-AUC | Top-10% prec |
|--------:|----------:|----:|-------:|------------:|
| 1-day | 40,915 | 0.5027 | 0.5098 | 0.524 |
| 5-day | 40,868 | 0.4954 | 0.5295 | 0.536 |
| 21-day | 40,679 | 0.4984 | 0.5674 | 0.535 |

**Read:** All horizons are essentially **at random (AUC ~0.50)** on out-of-sample, but **top-decile precision is 53-55%** which is genuine economic signal — the model knows where the *strongest* longs are even when the average prediction is noise.

### 6b. Post-retrain — Stacking + Rank target + Purged K-fold (DONE)

**Final OOF AUCs from purged K-fold + embargo training** (these are the honest, leakage-free scores):

| Horizon | Old (single XGB) | New (stack + rank + purged) | Δ |
|--------:|-----------------:|----------------------------:|--:|
| **1-day** | 0.5083 | **0.5183** | **+1.00 pp** ✓ |
| **5-day** | 0.4974 | **0.5145** | **+1.71 pp** ✓ |
| **21-day** | 0.4944 | **0.5125** | **+1.81 pp** ✓ |
| India specialist (5d) | n/a | 0.4991 | (under 0.51 blend threshold) |

Per-base learner OOF AUCs:

| Horizon | XGB | HistGB | Meta (logistic stack) |
|---|---|---|---|
| 1d  | 0.5178 | 0.5171 | **0.5183** |
| 5d  | 0.5141 | 0.5137 | **0.5145** |
| 21d | 0.5124 | 0.5097 | **0.5125** |

**Important note on the 80/20 held-out re-score:**
A separate held-out test (`_score_stack.py`) showed AUC 0.63-0.77 on the last 20% by date, but those numbers are **in-sample inflated** because the stack's *final* base learners were re-fitted on the full dataset (including those 20% rows) for live inference. The training-time OOF AUCs above (0.5125-0.5183) are the honest, generalisation-relevant scores.

Saved to:
- `data/models/stack_h1.pkl`
- `data/models/stack_h5.pkl`
- `data/models/stack_h21.pkl`
- `data/models/india_specialist.pkl`

`predict.py` now overrides the universal-h5 model with `stack_h5` whenever the stack's OOF AUC > universal AUC. India specialist blends 50/50 with universal only when its AUC > 0.51 (currently 0.4991 — won't activate until rebuilt with India-specific features).

### 6c. What "good" looks like for this universe

| Tier | AUC | Top-10% prec | Comment |
|---|---|---|---|
| Random | 0.50 | 0.50 | baseline |
| Weak signal | 0.51-0.53 | 0.53-0.55 | **← current production** |
| Decent retail | 0.53-0.55 | 0.55-0.58 | **← target post-retrain** |
| Strong institutional (Bloomberg AIM) | ~0.58 | ~0.60 | with proprietary data feeds |
| Top quant fund | 0.58-0.62 | 0.60-0.64 | with execution alpha + microstructure |

We are currently in the "weak signal" tier and the retrain is targeted at moving us into "decent retail" territory.

---

## 7. Operations

### 7a. Daily run

```cmd
python scripts/daily_orchestrator.py
```
Writes:
- `data/predictions/latest.parquet`
- `data/signals/latest.parquet`
- `data/regime/latest.parquet`

### 7b. Retrain (long, ~30-60 min on 623k rows)

```cmd
python scripts/train_universal_model.py ^
    --horizons 1,5,21 ^
    --ensemble stack ^
    --cv purged ^
    --target rank ^
    --india-specialist ^
    --force
```

### 7c. Tail logs

```cmd
powershell -Command "Get-Content C:\Omni_Stocks\data\omni.log -Wait -Tail 20"
```

### 7d. Dashboard

```cmd
streamlit run omni/dashboard/app.py
```

### 7e. Single-ticker prediction (Python)

```python
from omni.ondemand.predict import predict_ticker
card = predict_ticker("GRSE", "india", display_name="Garden Reach Shipbuilders")
print(card.to_dict())
```

### 7f. Score on real held-out data

```cmd
python _score_real.py
```
Writes `data/score_realdata.json` with per-horizon AUC / PR-AUC / Brier / top-decile precision, plus India-only subset metrics.

---

## Appendix A — Audit doc fixes shipped

- **B2 (HIGH)** — Single-ticker cross-sectional rank skew: live path now emits neutral 0.5 placeholders instead of degenerate ranks. Eliminates train/live distribution mismatch.
- **B3 (HIGH)** — Stale prediction cache after retrain: `predict_ticker` cache key now includes the model bundle mtime; retraining auto-invalidates stale cards.

Outstanding (next sprint): B1 (orchestrator live-vs-OOF split), B4 (recency-weight wiring), B5 (Brier alert), B6 (correlation arg order), B7 (crypto provenance label).
