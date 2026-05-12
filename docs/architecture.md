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
