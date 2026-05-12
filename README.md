# Omni_Stocks

A multi-asset, multi-LLM market prediction system for Indian (NSE), US,
crypto, and FX markets. Combines technical, fundamental, sentiment, macro,
microstructure, F&O open-interest, relative-strength, insider, and earnings-
momentum signals through a regime-aware **9-pillar rubric** layered on top of
a calibrated XGBoost stacking ensemble.

Designed to run locally on Windows + an RTX 4070 Super with Google Drive cold
storage and Snowflake Cortex for occasional heavy LLM synthesis. A FastAPI
service layer + Prometheus observability layer make the same codebase
deployable to a small VPS or containerized environment.

> **Status: production-grade.** Streamlit dashboard renders real predictions
> across 9 tabs, multi-source ticker search is live, model stack is trained
> and calibrated, and a RAG-backed LLM router produces evidence-cited
> explanations.

> **For a complete from-scratch explanation of every concept, module, and
> design decision in this project — read the master guide:**
> **[`docs/interview_prep_master_guide.md`](docs/interview_prep_master_guide.md).**

Other docs:

- [`docs/architecture.md`](docs/architecture.md) — subsystem design, data flow, phase plan
- [`docs/architecture_walkthrough.md`](docs/architecture_walkthrough.md) — module-by-module deep dive
- [`docs/project_tree.md`](docs/project_tree.md) — annotated directory layout
- [`docs/setup.md`](docs/setup.md) — manual machine setup (Python, Ollama, rclone, API keys)
- [`docs/startup_guide.md`](docs/startup_guide.md) — first-run quickstart

## What it does

1. Ingests OHLCV, fundamentals, news, macro, NSE bhavcopy, FII/DII flows,
   F&O option chains, and bulk/block deal data for **64 instruments**
   (20 NSE + 30 US + 10 crypto + 4 FX).
2. Engineers technical, fundamental, sentiment, macro, microstructure, and
   earnings features into a universal feature schema.
3. Detects market regime (bull / bear / chop / crisis) via a hardened
   4-state HMM with degenerate-data fallback.
4. Trains a stacked **XGBoost + HistGradientBoost + LightGBM + N-HiTS**
   ensemble with isotonic calibration and purged K-fold CV.
5. Combines model probability with a regime-conditional **9-pillar rubric**
   (technical, fundamental, sentiment, macro, microstructure, OI,
   relative-strength, insider, earnings-momentum). Every pillar degrades
   to neutral 50 when its input is missing, so the system stays robust
   across asset classes.
6. Emits **confidence-gated** signals only when conviction >= configured
   threshold AND the regime supports the direction.
7. Renders a Streamlit dashboard with explanations from a multi-LLM router
   (FinBERT for sentiment, OpenRouter / Ollama / Cortex for natural-language
   reasoning) backed by a Qdrant-powered RAG layer.

## 9-pillar rubric

| Pillar              | Inputs                                                                |
|---------------------|-----------------------------------------------------------------------|
| technical           | momentum, trend, RSI, vol, volume, breakout                           |
| fundamental         | valuation, leverage, ROE, growth (yfinance + Screener.in fallback)    |
| sentiment           | FinBERT-aggregated news + Reddit + GDelt                              |
| macro               | VIX, yield curve, DXY, country regime                                 |
| microstructure      | delivery%, OI change, FII/DII flow z, Amihud illiq, signed-vol proxy  |
| oi                  | F&O PCR, max-pain dist, OI buildup (NSE F&O names)                    |
| relative_strength   | RS vs sector + market benchmarks; pair-signal forward alpha           |
| insider             | bulk/block deals, promoter holding delta, FII/DII broadcast           |
| earnings_momentum   | revenue + earnings growth, EPS revisions, surprises                   |

Weights per pillar vary by regime (`config/rubric.yaml`). All four regime
profiles sum to 1.0000 and reweight the pillars to suit the market context
(e.g. `crisis` puts 30% on macro; `bull` puts 20% on technical).

## Production hardening

The codebase is more than a research notebook — it ships with:

- **FastAPI service layer** (`omni/api/`) with Pydantic schemas, lifespan
  warmup, and a typed `client.py` for downstream callers.
- **Prometheus metrics** (`omni/observability/metrics.py`) — counters,
  histograms, and gauges with graceful fallback when the client lib is absent.
- **RAG layer** (`omni/llm/rag/`) — Qdrant vector store, chunker, embedder,
  retriever, validators, and degradation monitor with rate-limited webhooks.
- **HTTP connection pooling** — `requests.Session` + `HTTPAdapter` +
  `urllib3.Retry` for every outbound API.
- **In-flight singleflight** in the on-demand cache to dedupe concurrent
  prediction requests for the same ticker.
- **Hardened HMM** — handles degenerate / non-PSD covariance, tied means,
  and Windows worker-thread timeouts (SIGALRM is main-thread only).
- **NSE OI scraper** with cookie warmup + retry against 401/Cloudflare.
- **Atomic parquet writes** with read-back validation (`omni/orchestrator/persist.py`).
- **Diagnostic scripts** (`scripts/check_data_quality.py`, `scripts/scan_bugs.py`)
  for per-ticker DQS + AST-based static bug detection.

## Realistic accuracy targets

| Slice                        | Target          |
|------------------------------|-----------------|
| Overall directional accuracy | 54-60%          |
| Top-decile gated signals     | 65-72%          |
| Calibration (Brier score)    | <= 0.22         |

The headline differentiators:

- Regime-conditional weighting across 9 pillars (signals weighted differently
  per regime).
- Explicit conviction gating (no signal is always better than a weak signal).
- Multi-LLM router with RAG (free-tier first, paid Snowflake Cortex on demand,
  evidence-cited explanations).
- Universal feature schema — the same model works across India / US / crypto
  / FX because pillars degrade gracefully on missing inputs.

## Repository layout

```
C:\Omni_Stocks\
  config\               YAML configs (universe, sources, rubric, llm_routes)
  data\
    hot\                Cleaned OHLCV (local SSD)
    features\           Engineered feature parquets
    lake\               OI snapshots, bhavcopy, FII/DII flows (parquet)
    models\             Trained model artifacts (.joblib)
    predictions\        Daily prediction outputs
    cache\llm\          diskcache for LLM responses
  deploy\               docker-compose for Qdrant + service deployment
  docs\                 architecture, setup, walkthrough, master guide
  omni\
    api\                FastAPI service + typed client
    dashboard\          Streamlit UI (9 tabs, 7 themes)
    features\           technical / fundamental / sentiment / macro /
                        microstructure / earnings / cross-sectional /
                        news_embeddings / regime_interactions / pair / targets
    ingest\             prices, fundamentals, news, macro, NSE bhavcopy,
                        OI, FII/DII, bulk-deals, Screener.in fallback
    llm\                multi-backend router with cache + rate limit
      backends\         finbert, ollama, openrouter, cortex
      rag\              Qdrant store, chunker, embedder, retriever,
                        validators, degradation monitor
    models\             XGB + HGB + LGBM + N-HiTS stacking ensemble
    monitoring\         predict_log.py, drift detection
    observability\      Prometheus metrics
    ondemand\           live-prediction path (cache, singleflight, OI augment)
    orchestrator\       daily walk-forward + production_runner
    regime\             4-state HMM with hardened fallback
    rubric\             9-pillar scorer
    storage\            tier-aware parquet I/O (local + Drive)
    utils\              paths, logger, env/yaml loader
  scripts\              training, refresh, diagnostics, smoke tests
  tests\                pytest suite
  pyproject.toml
  .env / .env.example   Secrets (.env is gitignored)
  README.md
```

Cold/raw data lives on **Google Drive** mounted via rclone at
`Z:\Omni_Stocks\`. Hot/processed data lives on **local SSD**. The split is
invisible to callers — all I/O routes through `omni.storage.tiers`, which
falls back to a local `raw_fallback/` directory if the Drive mount is
unavailable.

## Quickstart (after manual setup)

```powershell
cd C:\Omni_Stocks
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .

# Smoke test config loading
python -c "from omni.utils.env import universe; print(len(universe()['asset_classes']['us']['tickers']))"

# Mount Drive (one-shot)
& "$env:USERPROFILE\OneDrive\Desktop\Mount-OmniDrive.bat"

# Run the dashboard
streamlit run omni/dashboard/app.py

# Or run the FastAPI service
uvicorn omni.api.serve:app --port 8080

# End-to-end smoke test of the prediction path
python scripts/predict_cianagro.py
```

See [`docs/setup.md`](docs/setup.md) for the full setup including Ollama,
OpenRouter, NewsAPI, FRED, Qdrant, and Snowflake Cortex.

## Configuration

All behavioral knobs live in `config/*.yaml`:

| File              | Controls                                                    |
|-------------------|-------------------------------------------------------------|
| `universe.yaml`   | Tickers, suffixes, lookback years, macro reference series   |
| `sources.yaml`    | API keys (by env name), RSS feeds, FRED series list         |
| `rubric.yaml`     | 9-pillar weights per regime + conviction gate threshold     |
| `llm_routes.yaml` | Backends, rate limits, primary -> fallback chains, RAG opts |

Edit YAML, restart — no code changes required for routing/weighting tweaks.

## License & scope

Personal research project. Not investment advice. Free-tier APIs only;
paid services (Cortex) used selectively.
