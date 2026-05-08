# Architecture

This document describes the static design of Omni_Stocks. For setup steps,
see [`setup.md`](setup.md). For the project overview, see the top-level
[`README.md`](../README.md).

## High-level data flow

```
+--------+   +--------+   +--------+   +--------+   +--------+   +-----------+
|  Raw   |-->|  Hot   |-->|Features|-->| Model  |-->|Ensemble|-->|Predictions|
| (Drive)|   |(Local) |   |(Local) |   |(Local) |   | + Gate |   |  (Local)  |
+--------+   +--------+   +--------+   +--------+   +--------+   +-----------+
                  ^                          ^                         |
                  |                          |                         v
              +--------+                +---------+              +-----------+
              | Ingest |                | Regime  |              | Dashboard |
              | yfin / |                |  HMM    |              | Streamlit |
              |  ccxt  |                +---------+              +-----------+
              | newsapi|                       ^
              | fred   |                       |
              +--------+                  +----------+
                                          |  Rubric  |
                                          | (4 pillar|
                                          |  x 4 reg)|
                                          +----------+
                                                ^
                                                |
                                         +-------------+
                                         | LLM Router  |
                                         | finbert /   |
                                         | ollama /    |
                                         | openrouter /|
                                         |  cortex     |
                                         +-------------+
```

## Storage tiers

| Tier         | Location                  | Contents                          | Lifecycle                  |
|--------------|---------------------------|-----------------------------------|----------------------------|
| Raw          | `Z:\Omni_Stocks\raw\...`  | Untransformed source data         | Append-only, dedup'd       |
| Hot          | `data\hot\`               | Cleaned OHLCV (parquet)           | Refreshed daily            |
| Features     | `data\features\`          | Engineered ML features            | Rebuilt nightly            |
| Models       | `data\models\`            | `.joblib` calibrated XGBoost      | Walk-forward retrains      |
| Predictions  | `data\predictions\<date>` | Per-day signal output             | Append-only                |
| LLM cache    | `data\cache\llm\`         | diskcache of LLM responses        | TTL per task, 5GB cap, LRU |

All I/O routes through `omni.storage.tiers`. **Other modules never call
`omni.storage.local` or `omni.storage.drive` directly.** The `tiers` module
adds a Drive-down fallback to `data\raw_fallback\` for resilience.

## Module responsibilities

### `omni.utils`
- `paths.py` -- single source of truth for all paths. Loads `.env` once at
  import; everything else reads through `paths.*` functions.
- `logger.py` -- loguru logger with rotating file sink at `data/omni.log`.
- `env.py` -- YAML loader with `${VAR}` expansion; convenience accessors
  `universe()`, `sources()`, `rubric()`, `llm_routes()`.

### `omni.storage`
- `local.py` -- low-level parquet helpers (write, read, append, scan_lazy).
- `drive.py` -- same surface area but enforces `is_mounted()` first.
- `tiers.py` -- the only public I/O surface. Encodes the path convention
  `<tier>/<source>/<asset_class>/<ticker>.parquet`.

### `omni.ingest`
- `prices.py` -- OHLCV via yfinance (india/us/fx) and ccxt (crypto). Saves to
  raw + hot tiers in one pass.
- `fundamentals.py` -- yfinance `Ticker.info` snapshot, 15 fields, keyed by
  `(ticker, run_date)`.
- `news.py` -- NewsAPI per-ticker queries + RSS feeds. Dedup by URL. **Per-
  article sentiment is NOT scored here** -- that's the LLM router's job in the
  feature stage.
- `macro.py` -- FRED daily series via `fredapi`.

### `omni.llm`
- `router.py` -- `Router.route(task, prompt)` looks up `llm_routes.yaml`,
  walks the primary -> fallback chain, applies cache + rate limit per backend,
  returns an `LLMResult`.
- `cache.py` -- diskcache, sha256 keyed by `(backend, task, prompt, params)`,
  TTL per task, 5GB cap with LRU eviction.
- `ratelimit.py` -- token-bucket per backend (rpm + rpd composite). Thread-safe,
  shared across routes via a module-level registry.
- `backends/`
  - `base.py` -- `Backend` ABC + `LLMResult` dataclass + `BackendError`.
  - `finbert.py` -- HuggingFace pipeline; lazy-loaded (avoids torch import
    cost unless FinBERT is actually used). Returns signed score in [-1, 1].
  - `ollama.py` -- local HTTP `/api/generate`. Pairs with qwen2.5:7b and phi3.5.
  - `openrouter.py` -- OpenAI-compatible `/chat/completions`; rate-limited
    aggressively to stay within the free tier (20rpm/200rpd).
  - `cortex.py` -- Snowflake `CORTEX.SENTIMENT` and `CORTEX.COMPLETE` via
    `snowflake.connector` using a named connection from
    `~/.snowflake/connections.toml`.

### `omni.regime` (Phase 5)
4-state Gaussian HMM on (returns, realized vol, breadth, VIX) -> regime label
fed into the rubric.

### `omni.models` (Phase 5)
Walk-forward XGBoost classifier for the 5-day horizon. **Never random-split
time series.** Calibrated probabilities via Platt or isotonic.

### `omni.rubric` (Phase 5)
Computes a 0-100 score per pillar (technical / fundamental / sentiment /
macro), applies regime-conditional weights from `rubric.yaml`, returns a
weighted total.

### `omni.ensemble` (Phase 6)
Combines model probability with rubric score; applies the conviction gate
(`min_conviction: 70`) and regime-supportive direction filter.

### `omni.backtest` (Phase 7)
Walk-forward harness with leakage assertions, calibration plots, and a
shuffled-baseline comparison.

### `omni.dashboard` (Phase 8)
Streamlit app surfacing today's gated signals with rubric breakdown and
LLM-generated explanation.

### Daily orchestrator (Phase 9)
Single entry point: mount Drive -> ingest -> features -> regime -> predict ->
rubric -> gate -> persist -> unmount.

## LLM routing

`config/llm_routes.yaml` defines two top-level sections:

1. **`backends`** -- physical endpoints with type, model, rate limits, enabled flag.
2. **`tasks`** -- logical jobs (sentiment_per_article, theme_extraction,
   bulk_classification, daily_market_synthesis, signal_explanation,
   cross_validation) with primary + fallback backend chains and cache TTL.

The router:
1. Looks up the chain for the requested task.
2. Checks the cache for each backend in order (cache key includes backend
   name, so different backends cache independently).
3. On cache miss, instantiates the backend (lazily), waits on the rate
   limiter, calls it.
4. On `BackendError`, moves to the next backend in the chain.
5. Stores the success in cache with the task's TTL.

This means **adding/removing/reordering backends is a YAML edit**, no code change.

## Why a rubric on top of a model

Pure ML on noisy financial features regularly hits a ceiling around 54-58%
directional accuracy. A rubric provides:

1. **Interpretability** -- the dashboard can show "why" a signal fired, not
   just a probability.
2. **Regime adaptation** -- weights shift across bull/bear/chop/crisis without
   needing a separate model per regime.
3. **Robustness to model staleness** -- if a feature distribution shifts
   between retrains, the rubric absorbs some of the slack.
4. **A clean place to inject LLM-derived signals** -- sentiment pillar.

The model and rubric are **combined** (not chosen between) at the ensemble
stage; the gate then enforces selectivity.

## Anti-leakage discipline

1. **Walk-forward only** -- no `train_test_split`, no `shuffle=True` anywhere
   on time-indexed data.
2. **Feature timestamps must precede target timestamps** -- enforced by a
   `tests/test_no_leakage.py` assertion.
3. **News features lag by 1 trading day** -- avoids look-ahead from same-day
   articles published after close.
4. **Calibration on a held-out walk-forward fold** -- not on the training fold.
5. **Shuffled baseline** -- a deliberately leakage-free permutation test that
   should produce ~50% accuracy. If the real model fails to beat it
   meaningfully, something is wrong.

## Risk and disclaimers

- This is research code. Free APIs are rate-limited and intermittently flaky;
  the system is built to fail soft (skip a ticker rather than crash a run).
- Sentiment pillars derived from news are noisier than the headline accuracy
  numbers suggest.
- Crypto on Binance vs yfinance has small price differences (~0.1%); the
  raw tier records which source produced which row.
- Indian fundamentals from yfinance are sparser than US -- expect more NaNs
  in the fundamental pillar for NSE tickers.
