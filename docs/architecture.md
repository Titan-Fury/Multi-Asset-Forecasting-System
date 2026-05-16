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
                                          | (9 pillar|
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
| Lake         | `data\lake\`              | OI snapshots, NSE bhavcopy, FII/DII flows, RAG vector store, monitoring | Append-only / overwrite by date |
| Features     | `data\features\`          | Engineered ML features incl. microstructure | Rebuilt nightly  |
| Models       | `data\models\`            | `.joblib` calibrated stack ensembles | Walk-forward retrains   |
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
  `(ticker, run_date)`. Falls back to **Screener.in** scraper
  (`sources/screener.py`) for IN equities when yfinance returns sparse data.
- `news.py` -- NewsAPI per-ticker queries + RSS feeds. Dedup by URL. **Per-
  article sentiment is NOT scored here** -- that's the LLM router's job in the
  feature stage.
- `macro.py` -- FRED daily series via `fredapi`.
- `sources/nse_bhavcopy.py` -- NSE bhavcopy (delivery%, traded value).
- `sources/nse_oi.py` -- F&O option chain with cookie warmup + retry against
  401 / Cloudflare interstitials. Persists per-ticker snapshots to the lake.
- `sources/nse_flows.py` -- daily FII/DII net cash flow.
- `sources/screener.py` -- Screener.in fallback parser for IN fundamentals.

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

### `omni.features`
9 modules feeding the universal feature schema. Per ticker the daily build
(`omni.features.build.build_for_ticker`) runs technical → fundamental →
sentiment → macro → **microstructure** → targets. Microstructure features
(delivery%, OI change, FII/DII flow z, Amihud illiquidity, signed-volume
proxy) are now persisted to the lake by the daily walk-forward feature
build (Phase 24.16); the on-demand path mirrors the same compute via
`omni.ondemand.features_inline`. Bhavcopy + FII/DII flows are preloaded
once per universe build and broadcast across all India tickers, avoiding
N× redundant fetches.

### `omni.regime`
4-state Gaussian HMM on (returns, realized vol, breadth, VIX) → regime
label fed into the rubric. Hardened against degenerate inputs:
- `_label_states` sorts states by mean and applies stable index-based
  tiebreak (no more unlabeled states on tied means).
- A `predict()` probe inside `fit()` catches non-PSD covariance and routes
  to a fallback model instead of raising at inference time.
- `_run_with_timeout` only uses SIGALRM on the main thread; daemon-thread
  fallback used on Windows + worker threads.

### `omni.models`
**Stack ensemble** (not a single XGBoost): XGBoost + HistGradientBoost +
LightGBM + N-HiTS base learners with a logistic-regression meta-learner.
**Walk-forward only** with **purged K-fold** CV (no leakage between train
and validation across the embargo window). Calibrated probabilities via
isotonic regression. **Never random-split** time series.

### `omni.rubric`
**9-pillar scorer**, each pillar producing a 0-100 score:

| # | Pillar | Inputs |
|---|--------|--------|
| 1 | technical | momentum, trend, RSI, vol, volume, breakout |
| 2 | fundamental | valuation, leverage, ROE, growth (yfinance + Screener) |
| 3 | sentiment | FinBERT-aggregated news + Reddit + GDelt |
| 4 | macro | VIX, yield curve, DXY, country regime |
| 5 | microstructure | delivery%, OI change, FII/DII z, Amihud illiq, signed-vol |
| 6 | oi | F&O PCR, max-pain dist, OI buildup |
| 7 | relative_strength | RS vs sector + market; pair-signal forward alpha |
| 8 | insider | bulk/block deals, promoter holding delta, FII/DII broadcast |
| 9 | earnings_momentum | revenue + earnings growth, EPS revisions, surprises |

Per-regime weights from `rubric.yaml` (bull/bear/chop/crisis), all summing
to 1.0. **Every pillar degrades to neutral 50** when its input is missing,
so the system stays robust across asset classes — crypto skips
fundamental/insider/earnings entirely; non-NSE-F&O names skip OI.
Backwards-compat: `score_row` uses `weights.get(name, 0.0)` so older
configs (4/5/6-pillar) still load unchanged. Phase 25 adds
`omni.ondemand.altdata_patch` so the live prediction row gets
`insider_net_z_20d`, `fii_dii_net_z_20d`, and `block_deal_net_value_cr`
populated from live SEBI PIT-2015 / NSE FII-DII / NSE block-deal pulls
instead of falling through to neutral 50; missing live data still
degrades gracefully.

### `omni.ensemble` (Phase 6, soft-gate Phase 25)
Combines model probability with rubric score; applies the conviction gate
and the regime-direction filter. Phase 25 made two changes:

- The regime filter is now a *soft multiplier* (`gating.soft_regime: true`
  in `config/rubric.yaml`) instead of a binary block — counter-trend
  signals can still emerge if conviction is high enough. Set
  `soft_regime: false` to restore the legacy binary block for backtest
  reproducibility.
- `min_conviction` can be overridden per `(asset_class, regime)` via
  `gating.min_conviction_overrides` so crypto/fx and crisis regimes get
  their own floors. The output frame carries an `effective_conviction`
  column reflecting the post-multiplier value used for the threshold
  check.

### `omni.backtest` (Phase 7)
Walk-forward harness with leakage assertions, calibration plots, and a
shuffled-baseline comparison.

### `omni.dashboard` (Phase 8, evolved through Phase 24)

Streamlit app at `omni/dashboard/app.py`. Surfaces today's gated signals,
on-demand prediction lookups, watchlist, portfolio sizing, calibration,
earnings, history, and a developer panel for retraining + cache control.

**Module layout** (LOC as of Phase 24.13):

| File | LOC | Purpose |
|------|-----|---------|
| `app.py` | ~1600 | Entry point; section dispatch; sidebar config; per-section render functions (`_render_predict_tab`, `_render_portfolio_tab`, `_render_dev_tab`, etc.) |
| `theme.py` | ~2460 | All CSS + JS injection. `inject_theme()` writes `<style>` and `<script>` blocks via `st.markdown(unsafe_allow_html=True)`. Aurora mesh, glassmorphism, Cmd+K palette live here |
| `components.py` | ~1176 | Reusable widgets — KPI row, regime banner, signal card, prediction card, calibration panel, signal history table, news block |
| `cache.py` | ~188 | `@st.cache_data` wrappers around the four daily-orchestrator artifacts (predictions, signals, regime, backtest). 1-hour TTL safety floor + mtime-keyed invalidation |
| `marquee.py` | ~174 | Live ticker tape (NIFTY 50, SENSEX, S&P 500, BTC, GOLD, USDINR, etc.) via yfinance `fast_info`. 5-min cache. Last-known-good fallback in `st.session_state` |
| `searchhistory.py` | ~155 | Persistent Google-style search history at `data/watchlist/search_history.json`. Atomic write via `tmp.replace()` |
| `watchlist.py` | ~81 | Favorite tickers at `data/watchlist/favorites.json`. Same atomic pattern |
| `movers.py` | ~87 | Top movers within filtered universe |
| `charts/lwchart.py` | ~106 | Lightweight-charts price chart (TradingView library, CDN-loaded) |
| `charts/corrgraph.py` | ~102 | D3 force-directed correlation graph |
| `charts/cosmos3d.py` | ~192 | Three.js sphere cosmos — radius ∝ conviction, cluster ∝ asset class |

**Navigation model (Phase 24.10):** the original 12-tab horizontal
`st.tabs(...)` strip was replaced with a sidebar `st.radio` + section
dispatch. Twelve sections — Predict, Watchlist, Portfolio, Top movers,
Top signals, Watchlist heatmap, Cosmos 3D, Correlations, Calibration,
Earnings, History, Dev. Section state persists in
`st.session_state["__section"]`.

**Permanent sidebar (Phase 24.11):** the native Streamlit sidebar
collapse control is hidden via CSS (`display: none !important` on
`stSidebarCollapseButton` and adjacent testids). The sidebar owns
navigation + filters + theme picker; closing it would strand the user.
Three earlier attempts at a custom toggle button (Phase 24.5/24.9/24.10)
were abandoned because Streamlit's testids change between versions and
the JS-injected fallback added complexity without solving the underlying
problem.

**Theme system (Phase 24.7):** seven themes — Auto, Dark, Light,
Cyberpunk, Synthwave, Matrix, Nord. Each defines a CSS-variable contract:

```
--theme-c1..c4         (aurora gradient stops, oklch with alpha)
--theme-bg-stop1..3    (background base colors)
--theme-accent         (primary)
--theme-accent-2,-3    (secondary, tertiary)
color-scheme           (dark / light, drives form-control rendering)
```

`_theme_override(choice)` in `app.py` injects the chosen palette as inline
`style[data-theme]` on `:root`. The aurora mesh, scrollbar, hero gradient
and search-input border consume these via `var(--theme-c*)`. Special
themes also inject extras (Cyberpunk → glow text-shadows on KPI values,
Matrix → JetBrains Mono everywhere, Light → opaque metric backgrounds).

**Caching tiers:**

| Source | TTL | Where | Rationale |
|--------|-----|-------|-----------|
| Predictions / signals / regime / backtest artifacts | 1h + mtime | `cache.py` | Daily orchestrator overwrites; mtime is primary, TTL is safety floor |
| Live yfinance prices (per-ticker) | 4h | `omni/ondemand/cache.py` | Intraday updates not needed for prediction cards |
| News + sentiment (per-ticker) | 30 min | `omni/ondemand/cache.py` | Material headlines surface fast without spamming API |
| Predictions (per-ticker) | 30 min | `omni/ondemand/cache.py` | Balance freshness vs. inference cost |
| Fundamentals (per-ticker) | 24h | `omni/ondemand/cache.py` | Quarterly cadence in reality |
| Marquee live quotes | 5 min | `marquee.py` | Tape feels live without yfinance hammering |

`purge_ticker(ticker)` in `omni/ondemand/cache.py` drops all per-ticker
keys atomically; wired to the "Force refresh" button on the predict tab.

**Performance patterns:**
- Portfolio + Earnings + Correlations sections fan out per-ticker work
  via `ThreadPoolExecutor(max_workers=4-8)`. Per-future `timeout=30..60s`
  bound prevents a single hung yfinance call from freezing the whole
  section (Phase 24.13). Serial fallback if the executor itself raises
  (logged with `_log.warning`).
- All HTML interpolated via `st.markdown(unsafe_allow_html=True)` passes
  through `html.escape()` for user-derived strings (ticker, display
  name, news titles, why-text). News URLs are also defanged —
  `javascript:` / `data:` schemes get rewritten to `#` (Phase 24.13).
- `prefers-reduced-motion: reduce` triggers a universal
  `*, *::before, *::after { animation-duration: 0.01ms; transition-duration: 0.01ms; }`
  rule (Phase 24.12) so users with motion sensitivity get an instant
  static UI without breaking layout.

**Cross-platform dev tab (Phase 24.13):** the developer panel wraps
`subprocess.run([sys.executable, _script("..."), ...])` with bounded
timeouts behind buttons for: refit regime (HMM), retrain universal
model, retrain stack ensemble, refresh marquee, run SBI benchmark,
rebuild training set. Output streamed back via
`st.code(rc.stdout[-1000:])`. Earlier hardcoded `.venv\Scripts\python.exe`
+ `scripts\foo.py` paths were Windows-only — replaced with
`sys.executable` and `Path("scripts") / ...` for macOS/Linux support.

**Dead-code purges (Phase 24.13):** ~280 lines of unused CSS/JS were
deleted in a single pass after a mechanical audit:
- Dead utility classes: `.fx-tilt`, `.fx-pulse`, `.fx-skeleton`,
  `.fx-counter`, `.bento-grid`, `.accent-tint-*`, `.violet-tint-soft`,
  `.ultra-digit*`, `.ultra-num`
- Dead JS hooks: `tickFollower()` (infinite no-op rAF loop on null
  cursor elements), `bindHot()` / `rebindHot()` (cursor-hot class never
  consumed), `buildSlotRoll()` / `rollAll()` (animated slot-machine
  digits never emitted), `bindCounters()` (animated `--counter-num`
  never used)
- Dead opt-ins: `body.fx-goo` filter hook (class never set)
- Dead per-input chrome: `⌘K` chip on every text input label (Cmd+K
  overlay still works; just no per-input advertisement)
- Dead cache surface: `FEATURE_PATH` + `reload_keys()["features"]` (no
  consumer)
- All sidebar-toggle remnants from Phases 24.5/24.9/24.10 (CSS rules,
  JS-injected button, MutationObservers, idempotent guards)

The remaining `<style>` block is roughly **half presentation + half
defensive overrides** of Streamlit internal classes. The 310 `!important`
declarations are a maintenance smell but reflect Streamlit's own CSS-in-JS
architecture — without them, the framework stylesheets win the cascade.

### RAG layer (Phase 24.15) — `omni.llm.rag`
Retrieval-Augmented Generation for the why-text in prediction cards.
Replaces the previous "ungrounded" `signal_explanation` task with a
grounded pipeline that cites concrete news articles.

| Module | Lines | Role |
| --- | ---: | --- |
| `chunker.py` | 274 | Section-aware chunking (news / 10-K / transcript) with stable SHA1-based ids for idempotent upserts |
| `store.py` | 284 | Qdrant local-mode persistent store at `data/rag/qdrant/`. One collection per `(asset_class, doc_type)`. 2048-d cosine. Uses `query_points` (1.10+) + `DatetimeRange` for date filters |
| `embedder.py` | 155 | Two-tier cache wrapper around Nemotron 1B v2 (`OpenRouter free tier, 2048-d`). LRU(4096) + diskcache TTL 90 days. Auto-batches to 96/req. Bumps cache key when model dim changes |
| `retriever.py` | 241 | Hybrid retriever: dense (Qdrant) + BM25 + Reciprocal Rank Fusion + bge-reranker-base on CUDA (RTX 4070 Super). Falls back to CPU then to RRF order on failure |
| `prompts.py` | 115 | Citation-enforcing prompts. Strict rules: cite every claim `[N]`, never invent numbers, refuse on insufficient context |
| `validators.py` | 246 | Anti-hallucination layer: numeric grounding (regex + substring/normalised match), citation check (1-based, must be in retrieved set), sentence-level cosine grounding score (default threshold 0.55). Sanitises unsupported numbers to `[unverified]` |
| `pipeline.py` | 291 | End-to-end orchestrator. `RAGPipeline.explain_signal()` retrieves → prompts → routes via `signal_explanation_rag` → validates → retries once with temp=0.1 on unsafe → returns deterministic refusal on second failure |

**Wiring into prediction flow** (`omni/ondemand/predict.py:_explain`):

1. If `bundle.news_raw` is empty → deterministic fallback immediately
2. Otherwise: idempotent ingest into Qdrant, retrieve top-5, generate, validate
3. Three result paths, surfaced to the UI via `card.why_path`:
   - `"rag"` — green "grounded · 0.7X" badge
   - `"rag_low_grounding"` — amber "low confidence" badge
   - `"deterministic"` — no badge, shows pillar-based fallback text
4. `card.why_citations` carries `[{n, source, date, url, title, score}]`
   for the chip row below the why-text

**LLM routes (`config/llm_routes.yaml`):**
```yaml
signal_explanation_rag:
  primary: [cortex_claude]
  fallback: [groq_llama70b, gemini_flash, openrouter_minimax_free, ollama_qwen]
  cache_ttl_days: 1
  rag:
    embedder: nemotron
    reranker: bge-reranker-base
    retrieval: { doc_types: [news], date_window_days: 14, top_k: 5 }
    validators: { grounding_min: 0.55, require_citations: true,
                  strip_unsupported_numbers: true }
```

**Anti-hallucination contract.** Three independent checks:
- **Citation requirement** — at least one `[N]` marker; all referenced N must be ≤ len(retrieved). Hallucinated indices flagged in `invalid_citations`.
- **Numeric guardrail** — every number extracted from the answer (`$35.1B`, `94%`, `2024`, etc.) must appear in at least one retrieved chunk text (raw + normalised). Unsupported tokens are replaced with `[unverified]` so the LLM can't smuggle invented figures past the user.
- **Grounding score** — embed each answer sentence + each retrieved chunk; per sentence take max cosine; mean across sentences. Below threshold ⇒ retry with stricter temperature; second failure ⇒ deterministic fallback.

**Performance.** Cold start: bge-reranker-base ~270 MB downloaded once on first call (huggingface_hub cache), then loaded on CUDA in ~2-3s. Per-request: dense search ~30 ms, BM25 build+score over 50-doc pool ~5 ms, reranker top-30 pairs on RTX 4070 Super ~80 ms, Nemotron embed (cached after first hit) <1 ms. End-to-end RAG explain typically 3-6 s, dominated by the LLM call.

**Failure-soft design.**
- No `OPENROUTER_API_KEY` ⇒ embedder returns zeros ⇒ store skips upserts ⇒ deterministic fallback
- Qdrant lock conflict / corrupted store ⇒ caught in `_ensure_collection`, returns deterministic
- Reranker CUDA OOM ⇒ caught in `_get_reranker`, falls back to CPU then to plain RRF
- Every LLM backend in the chain failing ⇒ caught at pipeline `_generate`, returns deterministic
- Any exception in `_explain()` ⇒ logged, deterministic returned

**Dev tab affordances.** Three new buttons under "RAG store":
- Live stats (`total_points`, per-collection counts, storage path)
- "Re-index news → RAG" (walks `data/raw_fallback/news_*` parquets, ingests by ticker with asset-class heuristic)
- "Purge old chunks" (configurable N-day cutoff via DatetimeRange filter)

### Deferred items (post-Phase 24.15)
Items captured during planning but not implemented in this phase. Each
has a clear acceptance criterion so future agents can resume cleanly:

| Phase | Item | Notes |
| --- | --- | --- |
| C | Earnings call summarizer | Needs transcript ingest pipeline + new task `earnings_summary` in `llm_routes.yaml`. Reuse `chunk_transcript` + `RAGPipeline.answer_question` |
| D | "Ask Omni" Q&A tab | Streamlit chat UI bound to `RAGPipeline.answer_question`. Multi-turn history, ticker scope picker |
| E | 10-K risk-factor extractor | Needs SEC EDGAR ingest. Reuse `chunk_filing` (already supports section-aware split) |
| — | Self-consistency 3× sampling | Toggle in `signal_explanation_rag.rag.validators`. Triples LLM cost; only worth it for paid Cortex Claude path |
| — | Daily cron for news → RAG refresh | Today the auto-ingest happens lazily inside `_explain()`. A scheduled bulk re-index keeps cold-load latency low for tickers users haven't visited recently |
| — | Voyage Finance embeddings upgrade | Paid (~$0.10/1M tokens); finance-tuned. Drop-in replacement for `omni.llm.embed_nemotron`. Bump `_CACHE_TASK` and `_QDRANT_DIM` (Voyage Finance 2 is 1024-d) |
| — | qdrant in-server mode | If chunk count grows beyond ~5M, swap local mode for a docker-compose qdrant service. Existing `data/rag/qdrant` snapshot can be loaded directly |
| — | PII / source whitelist on retrieved chunks | Currently we trust whatever was ingested. A whitelist of approved domains would harden the citation surface against low-quality blogs |

### Daily orchestrator (Phase 9)
Single entry point: mount Drive -> ingest -> features -> regime -> predict ->
rubric -> gate -> persist -> unmount.

`omni/orchestrator/production_runner.py` wraps the daily run for headless
deployment (cron, systemd, container). Atomic parquet writes with
read-back validation in `omni/orchestrator/persist.py`.

### `omni.api` (Phase 24.16)
Production FastAPI service layer, separate from the Streamlit UI. Lets the
same prediction code run as a stateless HTTP service for headless callers.

- `app.py` -- FastAPI app with Pydantic schemas for `/predict`, `/health`,
  `/regime`, `/explain`. Lifespan warmup loads the universal stack ensemble
  + RAG store before the first request lands.
- `serve.py` -- uvicorn entry point (`uvicorn omni.api.serve:app`).
- `client.py` -- typed Python client for downstream callers; uses the same
  HTTP connection pool pattern (`Session` + `HTTPAdapter` + `Retry`).

### `omni.observability` (Phase 24.16)
Production telemetry. Graceful fallback when `prometheus_client` isn't
installed (counters/histograms become no-ops).

- `metrics.py` -- Prometheus counters, histograms, and gauges for
  prediction latency, cache hit rate, RAG grounding score, regime
  distribution, and per-backend LLM route success rates.
- RAG degradation monitor (`omni.llm.rag.degradation`) with rate-limited
  webhook alerts when grounding falls below threshold across consecutive
  windows.

### `omni.ondemand` (live prediction path)
- `predict.py` -- the on-demand prediction entry point. HTTP connection
  pooling: `_get_session()` (with retries) for predict POSTs vs
  `_get_probe_session()` (no retries) for `/health` probes — keeps probe
  latency low when one backend is degraded.
- `cache.py` -- diskcache with **in-flight singleflight** so concurrent
  requests for the same `(ticker, asof)` collapse into one upstream call.
  Cache timestamp is recorded post-completion (not pre-flight) so a hung
  request can't poison the cache.
- `pair_signal.py` -- pair-trading forward alpha estimator feeding the
  relative_strength pillar.
- `features_inline.py` -- DataFrame-input mirror of the daily feature
  builder. Identical schema, never persists.

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
2. **Purged K-fold CV** with embargo window in stack-ensemble training so
   adjacent train/validation rows can't share information across the fold
   boundary.
3. **Feature timestamps must precede target timestamps** -- enforced by a
   `tests/test_no_leakage.py` assertion.
4. **News features lag by 1 trading day** -- avoids look-ahead from same-day
   articles published after close.
5. **Calibration on a held-out walk-forward fold** -- not on the training fold.
6. **Shuffled baseline** -- a deliberately leakage-free permutation test that
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

## Phase 25 — empirically-driven model + UX improvements

This phase landed nine commits motivated by the May 8 2026 CIANAGRO crash
backtest, in which the quant model correctly turned bearish on every name
in the test basket but every signal was structurally blocked by either
the regime gate or the conviction threshold. Each commit ships behind an
opt-in env flag where applicable so production behaviour stays unchanged
unless explicitly enabled.

| Layer | Change | Module / file |
|-------|--------|---------------|
| ondemand | Live insider z, FII/DII z, block-deal patch on the prediction row | `omni/ondemand/altdata_patch.py` |
| training | Per-horizon recency half-life (`{1: 1.5y, 5: 3.0y, 21: 5.0y}`) | `scripts/train_universal_model.py` |
| calibration | Beta calibration as opt-in (`OMNI_CAL_METHOD=beta`) | `omni/calibration/{beta_cal,factory}.py` |
| features | Difference-encoded technicals + lagged realised labels | `omni/features/diff_lagged.py` |
| models | Regime-conditional stack bundle (`OMNI_REGIME_STACKS=1`) | `omni/models/regime_stack.py` |
| ui | Plain-English verdict banner on prediction cards | `omni/dashboard/verdict.py` |
| gate | Soft regime multiplier + per-(asset_class, regime) min_conviction | `omni/ensemble/gate.py` |
| storage | Point-in-time fundamentals snapshots for replay reproducibility | `omni/storage/fundamentals_snapshots.py` |
| features | Triple-barrier as default `--target` + India circuit-bar label masking | `omni/features/circuit.py`, `scripts/train_universal_model.py` |
| rubric | Learned per-regime weights via opt-in `OMNI_LEARNED_RUBRIC=1` | `omni/rubric/learned.py` |
| monitoring | PSI drift monitor + daily script | `omni/monitoring/drift.py`, `scripts/monitor_drift.py` |
| dev-tab | Repaired Refit-regime + Run-daily-orchestrator buttons | `scripts/refit_regime.py`, `scripts/run_daily.py` |
| perf | Vectorized `score_frame` and `gate.apply`; dropped `df.copy()` in scorer | `omni/rubric/scorer.py`, `omni/ensemble/gate.py` |

### Soft regime gate (`omni.ensemble.gate`)

The legacy hard regime block (longs only in bull/chop, shorts only in
bear/crisis) was replaced with a multiplicative penalty applied to
conviction *before* the threshold check:

```
direction \\ regime  bull  chop  bear  crisis
long  (+1)           1.00  0.90  0.50  0.30
short (-1)           0.50  0.90  1.00  0.80
```

Set `gating.soft_regime: false` in `config/rubric.yaml` to restore the
legacy binary block for backtest reproducibility. Per-row threshold is
also resolved from `gating.min_conviction_overrides` keyed by
`(asset_class, regime)` so crypto floors are higher than equity floors,
crisis tightens everything, etc. The output frame now carries an
`effective_conviction` column for auditability.

### Triple-barrier + India circuit-bar masking

`scripts/train_universal_model.py` now defaults to `--target tb`; when
`tb_label_<H>d` columns are missing the trainer logs a warning and
falls back to `fwd_dir` (no hard failure). `omni.features.circuit`
detects bars where `open == high == low == close` AND `|daily_ret|`
matches a +/-5/10/20% band, and `mask_circuit_labels` NaNs out
`fwd_dir_<H>d` / `tb_label_<H>d` on those source rows so they're
dropped from training via `dropna(subset=[target])`. Verified on real
CIANAGRO data: 16/59 bars correctly flagged across the parabolic +
crash sequence.

### Plain-English verdict banner

`omni.dashboard.verdict.verdict(card_dict)` returns a 4-field
`Verdict(headline, summary, disagreement, tone)` consumed by both the
`signal_card` (Top-signals grid) and `detail_card`. Tone drives the
banner colour:

| Headline | Tone | When |
|----------|------|------|
| Strongly Bullish | long  | proba >= 0.65 AND conv >= 30 |
| Strongly Bearish | short | proba <= 0.35 AND conv >= 30 |
| Cautious Bullish/Bearish | long/short | softer proba or low conv |
| Sit Out | flat | conv < 8 |
| (any) | warn | rubric and quant disagree on direction AND gate blocked |

The disagreement string is the trader-facing version of "model is
bearish but the rubric pillars look constructive — late-cycle warning",
generated from the strongest constructive pillars when the quant
disagrees with the rubric lean.

### PSI drift monitor

`omni/monitoring/drift.py` computes Population Stability Index per
feature with the conventional severity buckets (PSI < 0.10 ok, < 0.25
watch, >= 0.25 alarm). `scripts/monitor_drift.py` is the daily driver:

  - Baseline = older slice of training set, current = latest N business
    days (prediction snapshots if their schema overlaps, otherwise tail
    slice of training).
  - `--save` writes `data/monitoring/drift_<date>.parquet`.
  - First run on the current training set surfaced 22/110 features in
    alarm — top drifters are quarterly fundamentals coverage gaps and
    `amihud_illiq_20d` (real liquidity shift).

### Performance

Two hot-path optimizations validated by benchmarks:

  - `omni.rubric.scorer.score_frame` — replaced `df.iterrows()` +
    full-frame `df.copy()` with `df.to_dict("records")` + pre-allocated
    np arrays + a single `df.assign(...)` at the end. Bench:
    **17.9k rows/s** on 5000 random rows. Saves ~200 MB on the 700k-row
    pooled training frame.
  - `omni.ensemble.gate.apply` — vectorized the (regime, direction)
    multiplier and the per-row threshold lookup via flat tuple-keyed
    dicts; replaced O(n) Python method calls with O(n) C dict.get.
  - `omni.rubric.scorer` module init — hoisted the
    `omni.rubric.learned` import out of `score_row` so the per-call
    sys.modules dict lookup goes away on training-set sweeps.

### Anti-leakage refresh

The lagged-realised-label features are deliberately constructed with a
shift of `(horizon + 1)` trading days so the label's realisation
window has closed strictly before today's row date. The diagnostic
script (`scripts/diagnose_diff_features.py`) confirmed uniformly
positive AUC lift across h ∈ {1, 5, 21}.

### Env flags introduced this phase

| Variable | Default | Effect |
|----------|---------|--------|
| `OMNI_CAL_METHOD` | `isotonic` | `beta` opts into Kull-Filho-Flach beta calibration in walk-forward training |
| `OMNI_REGIME_STACKS` | unset | `1` loads `data/models/regime_stacks_h5.joblib` and routes inference by HMM proba_today |
| `OMNI_LEARNED_RUBRIC` | unset | `1` blends YAML rubric weights with the learned posterior at score time |
| `OMNI_LEARNED_RUBRIC_ALPHA` | `0.5` | Learned-prior blend weight (0 = pure YAML, 1 = pure learned) |

