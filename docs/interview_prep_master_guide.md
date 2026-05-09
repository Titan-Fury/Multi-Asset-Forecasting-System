# Omni-Finance — The Complete Beginner-to-Interview Guide

> A from-scratch explanation of every concept, every module, and every
> design decision in this project. Read this end-to-end and you should
> be able to defend every line of the system in a technical interview.

---

## Table of contents

1. [What this project is, in plain English](#1-what-this-project-is-in-plain-english)
2. [The core problem we are solving](#2-the-core-problem-we-are-solving)
3. [Glossary for absolute beginners](#3-glossary-for-absolute-beginners)
4. [System architecture — the big picture](#4-system-architecture--the-big-picture)
5. [Data flow — from raw API to UI prediction](#5-data-flow--from-raw-api-to-ui-prediction)
6. [Module-by-module walkthrough](#6-module-by-module-walkthrough)
7. [The model stack explained](#7-the-model-stack-explained)
8. [The 4-pillar rubric explained](#8-the-4-pillar-rubric-explained)
9. [LLMs being used and why](#9-llms-being-used-and-why)
10. [Caching strategy](#10-caching-strategy)
11. [Common interview questions and answers](#11-common-interview-questions-and-answers)
12. [Glossary of metrics](#12-glossary-of-metrics)
13. [How to demo the project in an interview](#13-how-to-demo-the-project-in-an-interview)

---

## 1. What this project is, in plain English

**Omni-Finance** is a system that takes a stock ticker (e.g. `RELIANCE`)
and produces a structured prediction:

> "Over the next 5 trading days, RELIANCE has a **54.5 % chance of
> going up**, with a **conviction of 1.6 / 100**, in a **bull regime**.
> Expected move: +0.18 % (band: −10.85 % to +11.20 %)."

It does this by combining four kinds of evidence:

| Pillar | Examples |
|---|---|
| **Technical** | RSI, MACD, Bollinger bands, momentum, volatility |
| **Fundamental** | P/E, ROE, debt-to-equity, free-cash-flow yield |
| **Sentiment** | News headlines from ~16 sources, scored via FinBERT + LLMs |
| **Macro** | VIX, USD/INR, US 10-year yield, FII/DII flows, market regime |

These four streams feed **three machine-learning models** (universal
XGBoost, an India-specialist sub-model, and a stacking ensemble), which
each emit a probability of "up". Those probabilities get blended with a
rule-based 4-pillar rubric, then passed through a **conviction gate**
(only signals with conviction ≥ 70 are emitted as actionable).

Everything is wrapped in a Streamlit dashboard with 9 tabs (Predict,
Watchlist, Top Movers, Heatmap, Top Signals, Cosmos 3D, Correlations,
Calibration, History).

---

## 2. The core problem we are solving

Stock-price prediction is **hard** because:

1. **Markets are nearly efficient.** The signal-to-noise ratio is so
   low that achieving even 53 % directional accuracy is a real edge.
2. **The training distribution shifts.** A model trained on the 2020
   COVID crash behaves completely differently in a 2024 bull regime.
3. **News is unstructured text** at a much higher frequency than the
   structured price data — you need NLP just to use it.
4. **Indian small-caps have thin coverage** — major news outlets
   ignore them, so we need niche sources (Telegram, Reddit, BSE
   notices, SEBI filings).

This project addresses each of those four problems explicitly:

| Problem | Our answer |
|---|---|
| Low signal-to-noise | Stack ensemble + purged K-fold CV + conviction gating |
| Distribution shift | Regime-conditional rubric (bull / bear / chop / crisis) |
| Unstructured news | 6-backend LLM router with task-specific routing |
| Small-cap coverage | 16 ingestion sources including SEBI, Telegram, BSE |

---

## 3. Glossary for absolute beginners

### Market / data terms

- **OHLCV** — Open, High, Low, Close, Volume. The five numbers that
  describe one trading day for one ticker.
- **Ticker** — The symbol an exchange uses for a stock.
  Examples: `RELIANCE` (NSE), `AAPL` (NASDAQ), `BTC-USD` (crypto).
- **NSE / BSE** — National Stock Exchange / Bombay Stock Exchange.
  The two main Indian exchanges. Yahoo Finance suffixes them as
  `.NS` and `.BO` respectively.
- **F&O** — Futures and Options. Derivatives whose value depends on
  the underlying stock.
- **PCR** — Put–Call Ratio. Total open interest of put options divided
  by call options. > 1 = bearish positioning, < 1 = bullish.
- **Max-pain strike** — The option strike price at which the largest
  number of options expire worthless. Often acts as a price magnet
  near expiry.
- **FII / DII** — Foreign Institutional Investors / Domestic
  Institutional Investors. Their daily net buy/sell flows are a
  classic India-market regime signal.
- **PIT-2015** — SEBI's Prohibition of Insider Trading Regulations
  2015. Promoters / KMPs must publicly disclose every trade.
- **ADR** — American Depositary Receipt. A US-listed proxy for a
  foreign stock (e.g. INFY on NYSE represents Infosys NSE).

### ML / statistics terms

- **Feature** — One numerical input column to the model. `rsi_14`,
  `macd`, `pe_ratio`, `news_sent_3d` are all features.
- **Target** — What we're predicting. Here it's `up_5d` — a 0/1
  label: 1 if the close 5 trading days from now is higher.
- **Training vs inference** — Training fits model parameters from
  labelled history. Inference uses the fitted model to predict on
  new data. We train weekly; we infer on every dashboard click.
- **AUC (ROC)** — Area Under the Receiver Operating Characteristic
  curve. 0.5 = random guessing. 1.0 = perfect. We see ~0.51–0.52,
  which is a small but **real and statistically meaningful** edge
  given our 124k training samples (SE ≈ 0.003).
- **Brier score** — Mean squared error of probability forecasts.
  Lower is better. < 0.22 is the project target.
- **Sharpe ratio** — Mean excess return divided by its standard
  deviation. > 1 is good for a strategy.
- **Hit rate** — Fraction of directional predictions that were correct.
- **Calibration** — How well a model's probabilities match reality.
  If the model says 70 % up, 70 % of those days should actually go up.
  We use **isotonic regression** (a monotonic non-parametric fit) to
  calibrate raw XGBoost outputs.
- **Cross-validation (CV)** — Train on part of the data, test on the
  held-out part, rotate. Naïve K-fold leaks information in time
  series; we use **purged K-fold** instead.
- **Purged K-fold + embargo** — From López de Prado's *Advances in
  Financial Machine Learning* §7. Drop training rows whose target
  window overlaps the test fold (purge), and drop a buffer of
  rows immediately after the test fold (embargo). Prevents look-ahead
  leakage.
- **Walk-forward** — Train on `[t₀, t₁]`, predict on `[t₁, t₂]`,
  retrain including new data, repeat. The way you actually deploy.
- **Stacking / stack ensemble** — Train multiple base learners
  (XGBoost + sklearn HistGradientBoosting), then train a second-stage
  meta-learner (logistic regression) on their **out-of-fold** (OOF)
  predictions. The stack often outperforms any single base learner.
- **Conformal prediction** — A way to wrap any model so its outputs
  carry a guaranteed coverage interval. We use **split-conformal**
  (Lei 2018) for our prediction bands.
- **Regime** — A persistent market state (bull / chop / bear / crisis).
  We classify the regime with a 4-state Gaussian **Hidden Markov
  Model** (HMM) on rolling-volatility / momentum features.
- **Cross-sectional rank target** — Instead of predicting "will RELIANCE
  go up?", predict "will RELIANCE rank in the top half of NSE stocks
  this week?". A Numerai-style trick that is more robust to absolute
  return scale.

### Software / infra terms

- **TTL cache** — Time-to-live cache. Stored values expire after a
  configurable interval. Lets us hit Yahoo / news APIs sparingly.
- **Diskcache** — A Python library for an SQLite-backed local cache
  that survives restarts.
- **Parquet** — Columnar binary file format. Fast and small for
  numeric data; what our `data/lake/` is made of.
- **Streamlit** — Python framework that turns scripts into web apps.
  Our dashboard is one Streamlit app.
- **Multi-source resolution** — Searching multiple APIs (Yahoo + NSE
  + CoinGecko) at once and merging by score.
- **Provenance** — Metadata recording *where* each data point came
  from (e.g. `prices_sources: ['yfinance']`). Critical for
  reproducibility and debugging.

---

## 4. System architecture — the big picture

```
┌────────────────────────────────────────────────────────────────────┐
│                       USER (browser)                               │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│       STREAMLIT DASHBOARD  (omni/dashboard/app.py)                 │
│  Tabs: Predict | ★ Watchlist | Top Movers | Heatmap |              │
│         Top Signals | Cosmos 3D | Correlations | Calibration | … │
└──────┬───────────────────────────────────────┬─────────────────────┘
       │ on-demand single-ticker path          │ batch path (cached)
       ▼                                       ▼
┌──────────────────────────┐           ┌─────────────────────────────┐
│  ON-DEMAND LAYER          │           │  DATA LAKE                  │
│  omni/ondemand/           │           │  data/lake/*.parquet         │
│   - search.py (NEW)       │           │  - prices                    │
│   - universe.py           │           │  - fundamentals              │
│   - lookup.py             │           │  - news_raw                  │
│   - features_inline.py    │           │  - features                  │
│   - predict.py            │           │  - models  (.pkl bundles)    │
└──────────┬────────────────┘           └─────────────────────────────┘
           │
           ▼
┌──────────────────────────┐    ┌─────────────────────────────┐
│  INGESTION (16 sources)   │    │  MODEL ENSEMBLE             │
│  omni/ingest/sources/     │    │  - Universal XGBoost (5d)   │
│   - prices (yfinance)     │    │  - India specialist         │
│   - fundamentals          │    │  - Stack: XGB + HGB + meta  │
│   - news (16 sources)     │    │  - Conformal intervals      │
│   - macro (FRED, NSE OI…)│    └──────────┬──────────────────┘
└────────────┬──────────────┘               │
             │                              ▼
             ▼                    ┌──────────────────────────────┐
┌─────────────────────────────┐   │  ENSEMBLE PIPELINE            │
│  FEATURE ENGINEERING        │   │  - rubric (4 pillars)         │
│  omni/features/ (13 modules)│──▶│  - regime gate                │
│  technical / fundamental /  │   │  - conviction gate ≥ 70       │
│  sentiment / macro /        │   │  - PredictionCard out         │
│  cross-sectional / pair /   │   └──────────────────────────────┘
│  intraday / earnings        │
└─────────────────────────────┘
```

The arrows go top-down for an on-demand request and bottom-up for
the nightly orchestrator.

---

## 5. Data flow — from raw API to UI prediction

A walkthrough of what happens when you type **`RELIANCE`** and click
**Run prediction**.

### Step 1: Resolve the ticker
`omni/ondemand/universe.py::resolve("reliance")`

```
"reliance"
   │
   ├── strip whitespace, normalise case
   ├── try explicit suffix (.NS / -USD / =X) → no
   ├── try alias map (curated 150 tickers)   → match: RELIANCE / india
   └── return Resolution(ticker='RELIANCE', asset_class='india',
                          display_name='Reliance Industries', confident=True)
```

### Step 2: Fetch the LookupBundle
`omni/ondemand/lookup.py::lookup_ticker(...)`

This pulls **everything** for one ticker, with TTL caching:

| Sub-fetch | Source | TTL |
|---|---|---|
| `prices` | yfinance `RELIANCE.NS` (with `.BO` fallback) | 4 h |
| `fundamentals` | yfinance `Ticker(...).info / balance_sheet` | 24 h |
| `news_raw` | Indian RSS + BSE notices + Telegram + SEBI PIT + Reddit + GDELT + MarketAux + NewsAPI | 30 min |
| `macro` | FRED (VIX, DGS10) + NSE OI features | 24 h |
| `regime` | HMM on rolling vol + momentum | 24 h |

Returns a `LookupBundle` dataclass with `.prices`, `.fundamentals`,
`.news_raw`, `.news_sentiment`, `.macro`, `.regime`, `.provenance`.

### Step 3: Build features
`omni/ondemand/features_inline.py::build_inline_features(bundle)`

Calls each builder in `omni/features/`:
- `technical.py` → 20+ TA indicators (RSI, MACD, ATR, Bollinger, OBV, …)
- `fundamental.py` → P/E, ROE, debt/equity, FCF yield, growth
- `sentiment.py` → 7-day rolling mean of news_sent + n_articles
- `sentiment_z.py` → z-score versus the ticker's own baseline
- `macro.py` → VIX, USD/INR, yield-curve slope as columns
- `pair.py` → spread vs sector benchmark (NIFTY for India, SPY for US)
- `regime_interactions.py` → feature × regime crosses
- `cross_sectional.py` → rank within (date, asset_class)
- `vol_adjusted_momentum.py` → Sharpe-style risk-adjusted return
- `earnings.py` → days-to-next-earnings, surprise history

Result: a single-row DataFrame with ~59 features.

### Step 4: Run the models
`omni/ondemand/predict.py::predict_ticker(...)`

```
                    ┌─ universal XGBoost  →  proba_up = 0.541
features (59 cols)──┼─ stack ensemble     →  p_stack = 0.547   (used: OOF AUC > universal)
                    └─ india specialist   →  proba blocked (OOF AUC 0.499 < 0.51 threshold)

final proba_up = p_stack = 0.547
```

### Step 5: Score the rubric
`omni/rubric/scorer.py::score_frame(...)`

Computes 4 pillar scores (0–100 each) from feature buckets:
- **Technical**: weighted RSI / MACD / momentum / vol regime
- **Fundamental**: weighted P/E / ROE / debt / growth
- **Sentiment**: weighted sent_score / sent_momentum / coverage
- **Macro**: weighted VIX / yields / FX / FII flows

`rubric_total = mean(pillars)`. For RELIANCE: 45.2 / 100.

### Step 6: Blend, gate, and emit
`omni/ensemble/pipeline.py::assemble_signals(...)`

```
blended_p   = 0.5 * proba_up + 0.5 * (rubric_total / 100)
            = 0.5 * 0.547 + 0.5 * 0.452
            = 0.500

direction   = +1 if blended_p > 0.52 else
              -1 if blended_p < 0.48 else 0
            = 0  (FLAT)

conviction  = |blended_p - 0.5| * 200
            = 0.06 / 100
            = 1.6  (very low)

passes_gate = (conviction >= 70) AND (direction != 0) AND regime_supportive
            = False  (conviction < 70, direction = 0)
```

### Step 7: Wrap in a PredictionCard and return
The `PredictionCard` dataclass carries everything the UI needs:
direction, conviction, blended_p, proba_up, pillars, regime,
passes_gate, reasons, why_text, news_top5, provenance, model_kind,
expected moves at 1d / 5d / 21d.

The UI reads `card.to_dict()` from `st.session_state` and renders it
in the Predict tab with badges, expected-move bands, and the
TradingView Lightweight chart.

---

## 6. Module-by-module walkthrough

### `omni/utils/`
- **`logger.py`** — `loguru`-based structured logger; ANSI colours
  for terminal, JSON-line format for file. Used everywhere.
- **`env.py`** — `.env` loader using `python-dotenv`.
- **`paths.py`** — One source of truth for `data_dir() / hot_dir() /
  lake_root() / models_dir()` etc.

### `omni/lake/`
- **`io.py`** — `read_prices() / write_features()` parquet helpers.
- **`paths.py`** — Partition path conventions (`prices/{asset_class}/{ticker}.parquet`).
- **`backfill.py`** — Multi-ticker bulk-fetch orchestration with
  concurrent threads + retry.
- **`quarterly.py`** — Quarterly-statement backfill with the
  idempotent-suffix guard (so `RELIANCE.BO` doesn't become
  `RELIANCE.BO.NS`).

### `omni/storage/`
- **`local.py`** / **`drive.py`** / **`mount.py`** — three storage
  tiers: hot SSD → warm HDD → cold Google Drive (via rclone).
- **`tiers.py`** — Promotion / demotion policy by file age.

### `omni/ingest/`
- **`prices.py`** — yfinance + ccxt with the **idempotent-suffix
  guard** + the **`.BO` fallback when `.NS` is empty or returns
  fewer than 30 trading days** (catches delisted-NSE ghosts like
  HINDMOTORS).
- **`fundamentals.py`** — yfinance `Ticker(...).income_stmt`, etc.
- **`news.py`** — Top-level aggregator that fans out to all 16
  sources in parallel.
- **16 source modules** under `sources/`. See [section 9](#9-llms-being-used-and-why)
  for which ones produce raw text and which produce structured features.

### `omni/features/`
13 modules; see [step 3](#step-3-build-features) above for what each
builds. The orchestrator is `build.py`.

### `omni/cv/`
- **`purged_kfold.py`** — `PurgedKFold(n_splits=5, embargo_days=10)`.
  Drops training rows whose target window overlaps the test fold,
  plus a 10-day buffer after the test fold.

### `omni/models/`
- **`xgboost_5d.py`** — Universal classifier. ~59 features, isotonic
  calibration. OOF AUC: ~0.5083.
- **`specialists.py`** + **`omni/specialists/india.py`** — India-only
  sub-model. OOF AUC: 0.499 (below the 0.51 blend threshold, so it
  is currently **not** activated in production).
- **`stack.py`** — Stacking ensemble. Base learners: XGBoost +
  `HistGradientBoosting`. Meta-learner: logistic regression.
  OOF meta AUC: ~0.5183 (h=1d), 0.5145 (h=5d), 0.5125 (h=21d).
- **`conformal.py`** — Split-conformal calibration → prediction
  bands.
- **`training_utils.py`** — Class-weight balancing, early stopping.

### `omni/regime/`
- **`hmm.py`** — 4-state Gaussian HMM. States: bull / chop / bear /
  crisis. Trained on rolling vol + momentum. Persisted to
  `data/regime/hmm.pkl`.

### `omni/rubric/`
- **`scorer.py`** — The hand-crafted 4-pillar scoring engine.
  See [section 8](#8-the-4-pillar-rubric-explained) for details.

### `omni/ensemble/`
- **`combine.py`** — `blended_p = 0.5 * proba_up + 0.5 * rubric/100`.
- **`gate.py`** — Conviction floor + regime filter.
- **`pipeline.py`** — Top-level signal-assembly entry point.

### `omni/llm/`
- **`router.py`** — Reads `config/llm_routes.yaml`, routes a task
  through its `primary → fallback` chain on rate-limit / error.
- **`cache.py`** — `diskcache`-backed response cache, keyed on
  `(backend, model, prompt_hash)`. Default TTL 24 h.
- **`ratelimit.py`** — Token-bucket per backend with TPM + RPM
  pre-flight check.
- **`backends/`** — One subclass of `Backend` per provider. See
  [section 9](#9-llms-being-used-and-why).

### `omni/ondemand/`
- **`universe.py`** — Resolver + suggestions + Sørensen-Dice
  fuzzy match for typo tolerance.
- **`search.py`** ⭐ — Multi-source live search (Yahoo Finance Search
  API + NSE autocomplete + CoinGecko). 30s in-process TTL cache.
- **`lookup.py`** — `lookup_ticker()` → `LookupBundle`.
- **`features_inline.py`** — Inline feature builder for one ticker
  bundle.
- **`macro_live.py`**, **`regime_live.py`**, **`pair_signal.py`** —
  Live-only equivalents that don't require a backfilled lake.
- **`predict.py`** — `predict_ticker()` and `deep_analysis()`.
- **`cache.py`** — `TTL` constants + `purge_ticker(ticker)` helper.
- **`model_registry.py`** — Lazy-loads model bundles with mtime
  invalidation so retraining auto-refreshes predictions.

### `omni/dashboard/`
- **`app.py`** — Streamlit entry. 9 tabs.
- **`theme.py`** — Glassmorphism CSS, hero, pill helpers, **sticky
  tabs with smooth fade-in transitions**.
- **`components.py`** — `kpi_row`, `regime_banner`, `prediction_card`,
  `provenance_strip`, `signal_card`, `universe_heatmap`, etc.
- **`cache.py`** — `@st.cache_data` wrappers for parquet loaders +
  realistic synthetic demo data.
- **`watchlist.py`** — Persistent JSON favorites at
  `data/watchlist/favorites.json`.
- **`movers.py`** — Top-N gainers/losers across the curated universe.
- **`charts/cosmos3d.py`** — Three.js 3D Signal Cosmos.
- **`charts/corrgraph.py`** — D3 force-directed correlation graph.
- **`charts/lwchart.py`** — TradingView Lightweight-Charts ESM.

### `omni/orchestrator/`
- **`daily.py`** — End-to-end nightly pipeline: ingest → build features
  → score → write `data/predictions/latest.parquet`.
- **`persist.py`** — Writes the parquets the dashboard reads.
- **`cli.py`** — `python -m omni.orchestrator` entrypoint.

---

## 7. The model stack explained

Three layers from bottom to top:

```
                  ┌─────────────────────────┐
   meta-learner   │  Logistic Regression    │  ← trained on OOF preds
                  └──────────┬──────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
  base learner: XGBoost            base learner: HistGradientBoosting
  (gradient boosted trees)         (sklearn's fast histogram impl)
                ▲                         ▲
                └──────────┬──────────────┘
                           │
                           │  inputs: 59 features
                           │
                  ┌────────┴────────┐
                  │   feature panel  │  (purged-K-fold-CV split)
                  └─────────────────┘
```

### Why a stack?
Each base learner has different inductive biases. XGBoost handles
non-linearity well. HistGradientBoosting is faster on the high-end and
sometimes captures different interactions. The meta-learner figures
out **how to combine them** purely from their out-of-fold predictions,
avoiding the most common stacking mistake (training the meta on
leaky in-fold predictions).

### Why purged K-fold?
Vanilla K-fold leaks information when targets span multiple days. If
your target is "5-day forward return" and you split on day, the
training fold (day 100) and test fold (day 102) **see overlapping
forward returns**. The fold purge drops those training rows, and the
embargo adds a 10-day gap after the test fold so post-test prices
can't influence training either.

### Why the rank target instead of raw "up/down"?
Cross-sectional rank (`will RELIANCE be top half of NSE this week?`)
is **more learnable** than raw direction because:

1. It cancels out market-wide moves (no need to also predict NIFTY).
2. It is more uniform across regimes (always 50 % positive class).
3. It is what most institutional alpha signals actually predict
   (see Numerai, WorldQuant, etc.).

We still output a "direction" label by post-hoc thresholding the
rank prediction — but the model's loss is on rank.

---

## 8. The 4-pillar rubric explained

The rubric is a **hand-crafted scoring engine** that runs alongside
the model. Each pillar produces a 0–100 score:

| Pillar | What it scores | Sub-features |
|---|---|---|
| **Technical** | Price-action quality | RSI, MACD divergence, ATR-normalised range, momentum z-score, volume confirmation |
| **Fundamental** | Business quality | P/E vs sector, ROE, debt-to-equity, FCF yield, revenue growth |
| **Sentiment** | News mood | 7-day rolling sent mean, sent momentum, n_articles, source diversity |
| **Macro** | External tailwinds | VIX level, USD/INR z-score, US 10y yield, FII/DII net flow |

Each sub-feature is converted to a 0–100 score via piecewise-linear
mapping defined in `config/rubric.yaml`, then weighted into the pillar
total. The **rubric total** = mean of pillars.

The rubric exists for three reasons:

1. **Explainability.** A trader can read "fundamental: 1.8 / 100"
   and immediately understand why a signal didn't fire.
2. **Robustness.** When the model is wrong (markets do change), the
   rubric is a sanity check that prevents wild blended_p excursions.
3. **Diversification.** The rubric is fundamentally a different
   estimator than gradient-boosted trees, so blending them lowers
   variance.

---

## 9. LLMs being used and why

The system uses **8 distinct LLM backends** orchestrated by a router.

### The full inventory

| Backend | Model | Cost | What it's used for |
|---|---|---|---|
| **FinBERT** | `ProsusAI/finbert` | local, free | Per-article 3-class sentiment |
| **Ollama** | `qwen2.5:7b-instruct-q5_K_M` | local, free | Theme extraction fallback |
| **Ollama** | `phi3.5:mini` | local, free | Cheap classification fallback |
| **Groq** | `llama-3.3-70b-versatile` | free tier | Theme extraction (primary) |
| **Groq** | `llama-3.1-8b-instant` | free tier | **Bulk classification (primary)** |
| **Gemini** | `gemini-2.0-flash` | free tier | Cross-validation (primary) |
| **Gemini** | `gemini-2.0-flash-lite` | free tier | Bulk classification fallback |
| **OpenRouter** | `meta-llama/llama-3.1-70b-instruct:free` | free | Generic 70B fallback |
| **OpenRouter** | `google/gemma-2-9b-it:free` | free | Lightweight fallback |
| **OpenRouter** | `minimax/minimax-m2.5:free` ⭐ | free | **Signal explanation (primary)** |
| **OpenRouter** | `minimax/minimax-m2.5` ⭐ | paid | Signal explanation (paid fallback) |
| **Snowflake Cortex** | `claude-3-5-sonnet` | paid | **Daily market synthesis (primary)** |
| **Snowflake Cortex** | `llama3.1-405b` | paid | Heavy-duty fallback |
| **Snowflake Cortex** | `SENTIMENT` (built-in) | free in trial | Quick sentiment check |

### The task → backend chain (`config/llm_routes.yaml`)

```
sentiment_per_article    →  finbert
theme_extraction         →  groq_70b → gemini_flash → ollama_qwen → openrouter_70b → ollama_phi
bulk_classification      →  groq_8b  → gemini_lite  → ollama_phi  → ollama_qwen
daily_market_synthesis   →  cortex_claude → groq_70b → gemini_flash → cortex_llama405 → openrouter_70b
signal_explanation       →  minimax_free ⭐ → groq_70b → minimax_paid → gemini_flash → ollama_qwen → gemma → cortex_claude
cross_validation         →  gemini_flash → groq_70b → openrouter_70b → gemma → ollama_qwen
```

### Yes — MiniMax M2.5 is being used

It is the **primary model for the `signal_explanation` task** — i.e.
the natural-language "Why this signal" text you see beside every
prediction card on the dashboard. The free-tier (`minimax-m2.5:free`,
12 RPM / 200 RPD) is tried first; if that hits a quota, the paid
tier (`minimax-m2.5`, 60 RPM / 5000 RPD) is the next fallback.

### Why this many backends?

1. **Free-tier rate limits are tiny.** No single free backend can
   handle even a dozen news classifications per minute. The router
   sums them.
2. **Latency-sensitive vs throughput-sensitive tasks.** Groq's 8B is
   ~80 ms/req — perfect for bulk classifying 500 headlines. Claude
   3.5 Sonnet via Cortex is ~2 s/req but writes much better daily
   synthesis prose.
3. **Privacy / offline.** Local Ollama lets the system keep working
   when the laptop is offline.
4. **Cost ceiling.** Free tiers are tried before paid; paid is used
   only for `daily_market_synthesis` (1 call/day) and the rare
   MiniMax-paid fallback.

### Caching is critical

The LLM cache (`omni/llm/cache.py`) is `diskcache`-backed and keyed
on `(backend, model, prompt_hash)`. Default TTL: 24 h, up to 7 days
for slow-changing tasks like theme extraction. **Without this
cache, a single dashboard refresh would consume the entire daily
free-tier quota of every provider.**

---

## 10. Caching strategy

| Layer | TTL | Storage | Purpose |
|---|---|---|---|
| News headlines | 30 min | `data/cache/ondemand` | Fresh sentiment without spamming feeds |
| Prices | 4 h | `data/cache/ondemand` | yfinance has rate limits |
| Fundamentals | 24 h | `data/cache/ondemand` | Quarterlies don't change daily |
| Macro | 24 h | `data/cache/ondemand` | FRED is daily-frequency anyway |
| Regime label | 24 h | `data/cache/ondemand` | HMM state changes slowly |
| **Prediction card** | **30 min** | `data/cache/ondemand` | Tracks the news TTL |
| LLM responses | 24 h–7 d | `data/cache/llm` | Free-tier quota protection |
| Streamlit data loaders | mtime-keyed | in-process | Auto-invalidates on lake update |

The **Force refresh** button on the Predict tab calls
`purge_ticker(ticker)` which drops every cache entry referencing the
ticker, so the next call rebuilds everything from live APIs.

**Important: caching is NOT model retraining.** The trained model
captures the statistical edge from years of history (124k samples).
Retraining on every dashboard click would overfit to the last 60 bars
and destroy that edge. Retraining is the orchestrator's job (run
`python scripts/train_universal_model.py` weekly).

---

## 11. Common interview questions and answers

**Q: How do you avoid look-ahead leakage in cross-validation?**
A: Three layers. (1) Targets are constructed only from future prices,
never current. (2) Cross-validation uses purged K-fold + 10-day
embargo (de Prado AFML §7) to drop overlapping target windows.
(3) Walk-forward retraining at deployment time means the production
model is always trained on a closing window that does not see the
rows it will be scored on.

**Q: Why XGBoost specifically?**
A: Three reasons. (1) Gradient-boosted trees handle missing values
natively, which matters when fundamentals are quarterly and prices
are daily. (2) They're translation-invariant — we don't need to
normalise features as long as the splits are based on relative order.
(3) They train fast enough that we can do nested CV (purged K-fold ×
walk-forward) within a daily-orchestrator window. We did experiment
with LightGBM and CatBoost; XGBoost was within 0.5pp AUC and we
already had the operational know-how.

**Q: Why ensembling? Aren't you overfitting?**
A: We OOF-stack, not in-fold-stack — the meta-learner sees only the
out-of-fold predictions of the base learners, so it can't see any
data the base learners saw. That bounds the additional capacity to
"how to mix two complementary biases" rather than "memorise the
training set". OOF AUC of the stack (0.5183) beats the universal
alone (0.5083) by ~3σ given our 124k samples (SE ≈ 0.003).

**Q: Why is your AUC only 0.51 — isn't that almost random?**
A: For a daily directional signal across 64 instruments, 0.51 is in
the upper end of what published equity-prediction literature reports.
Markets are nearly efficient; sustained AUC > 0.55 typically requires
either (a) higher-frequency data (intraday), (b) micro-structure
features (order flow), or (c) alternative data we don't have. The
edge becomes economic via *position sizing* (Kelly), *gating* (only
trade conviction ≥ 70), and *cross-sectional* application (long top
decile / short bottom decile rather than bet on every name).

**Q: How do you detect regime change in real time?**
A: A 4-state Gaussian HMM trained on rolling 21-day vol + 63-day
momentum. The Viterbi-decoded state at each timestep is one of bull /
chop / bear / crisis. The regime is then **a feature** to all models
(via `regime_interactions.py`) AND a **gate** in the ensemble pipeline
(some signals only fire in supportive regimes). HMM was chosen over
a more elaborate model (e.g. RegSwitchVAR) because it's
small-data-friendly and produces interpretable transition probabilities.

**Q: What are conformal prediction intervals and why do you use them?**
A: Conformal prediction (Vovk, Lei) wraps any black-box predictor and
produces calibrated intervals — `[lo, hi]` such that the true future
return lies inside `[lo, hi]` at least `1 - α` fraction of the time
(under exchangeability). We use **split-conformal**: hold out a
calibration fold, compute non-conformity scores, take the
`(1−α)` quantile to set the band width. The expected-move band you
see on the dashboard is the conformal-adjusted ±1σ interval.

**Q: Why so many news sources?**
A: Two reasons. (1) Coverage — Indian small-caps like OMAXAUTO have
zero hits in international sources but appear in Telegram channels and
BSE notices. (2) Robustness — single-source sentiment is biased by
that source's editorial slant; aggregating ~16 sources averages the
bias out.

**Q: How does the LLM router survive rate-limit errors?**
A: Each task has a primary→fallback chain in `llm_routes.yaml`. The
router catches `RateLimitError` / `QuotaError` / `TimeoutError` from
the current backend and tries the next one. A token-bucket per
backend lets the router skip a backend that is *already saturated*,
without wasting an HTTP round-trip on a guaranteed 429.

**Q: How do you prevent data leakage in your sentiment features?**
A: Two safeguards. (1) `published_at` timestamps are timezone-aware
and we drop any article whose `published_at >= prediction_date`.
(2) The 7-day rolling sentiment is computed up to but not including
the prediction date, then merge-joined back-as-of by date.

**Q: Why Streamlit instead of a "real" web framework?**
A: This is a research / single-user analytics tool, not a multi-user
SaaS. Streamlit gives us reactive UI in pure Python, which means the
same code that builds features for the model also drives the
dashboard. For a multi-tenant production product I'd reach for
FastAPI + React, but the cost-benefit at this scale tilts heavily
to Streamlit.

**Q: What's the most surprising thing you learned building this?**
A: How much of the work is data-quality plumbing rather than
modelling. The XGBoost call is ~50 lines. The 16 ingestion modules,
the cookie-warmed NSE session, the SEC EDGAR XBRL parser, and the
sys.path bootstrap fix in `app.py` — together they're ~3,000 lines.
For nearly any ML project the modelling is a thin layer over a thick
data foundation.

---

## 12. Glossary of metrics

| Metric | Definition | Good value | Where it appears |
|---|---|---|---|
| **OOF AUC** | Out-of-fold ROC AUC | > 0.51 (markets are hard) | model cards, dashboard banner |
| **Brier score** | Mean (proba − outcome)² | < 0.22 | calibration plot |
| **Sharpe ratio** | μ / σ of strategy returns | > 1 | backtest summary |
| **Hit rate** | TP+TN / total | > 0.54 | backtest summary |
| **IR** | Info Ratio: alpha / tracking error | > 0.5 | backtest summary |
| **Max DD** | Max peak-to-trough drawdown | > −20 % | backtest summary |
| **Conviction** | \|blended_p − 0.5\| × 200 | 0–100 | every PredictionCard |
| **Rubric total** | Mean of 4 pillar scores | 0–100 | every PredictionCard |
| **Coverage** | Fraction of days the conformal band contains the realised return | ~ 1−α (e.g. 80 %) | conformal report |

---

## 13. How to demo the project in an interview

A 5-minute demo that hits everything:

1. **Open the dashboard** → point at the regime banner and KPI row.
   "The HMM-decoded state is bull right now; here are 4 KPIs from the
   latest backtest fold."
2. **Type `RELIANCE`** → click Run.
   "Resolved via curated alias to RELIANCE NSE. The lookup bundle
   pulled prices from yfinance with `.NS / .BO` fallback, and news
   from 16 sources cached at 30 minutes."
3. **Walk through the prediction card.**
   "Direction is FLAT because conviction is 1.6 — well below our 70-gate.
   The pillars tell the story: technicals 60, fundamentals 1.8.
   The why-text was written by MiniMax M2.5 via OpenRouter."
4. **Switch to ★ Watchlist** → "Persisted to JSON, survives restarts."
5. **Switch to Top Movers** → "Top 10 gainers / losers in the last
   1d, 5d, 21d, computed from live OHLCV with a 4-hour cache."
6. **Switch to Cosmos 3D** → "Three.js raycaster picking. Each
   sphere is a ticker; radius is conviction, colour is direction,
   cluster is asset class."
7. **Type a typo: `oman autos`** → resolves to OMAXAUTO via
   Sørensen-Dice bigram fuzzy match.
8. **Type `palantir`** → resolves to PLTR US via the multi-source
   live search (Yahoo Finance Search API).
9. **Click Force refresh** → "Drops every cache entry for this
   ticker — prices, news, fundamentals, the prediction card — and
   rebuilds in ~10 s."

Talk track for any pause: "*the model itself is a stack ensemble of
XGBoost + HistGradientBoosting with a logistic meta-learner, trained
with purged 5-fold CV and a 10-day embargo to prevent target leakage.
Calibration is isotonic. Conformal split bands give us guaranteed
coverage on the expected-move intervals.*"

---

**Author's note.** Read this once for understanding, twice for
recall, three times for fluency. Every paragraph here is something
you should be able to say without notes if asked.
