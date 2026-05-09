# Project tree

```
Omni-Finance/
├── .env.example                 # Template for API keys (real .env is gitignored)
├── .gitignore
├── .streamlit/                  # Streamlit theme + config
├── README.md
├── pyproject.toml               # Dependency manifest
│
├── config/                      # Project-level YAML configuration
│   ├── benchmarks.yaml
│   ├── llm_routes.yaml          # LLM router fallback chain
│   ├── rubric.yaml              # 4-pillar scoring weights
│   ├── sources.yaml             # Data-source manifest
│   └── universe.yaml            # Curated ticker list
│
├── docs/                        # Architecture + walkthrough docs
│   ├── architecture.md
│   ├── architecture_walkthrough.md
│   ├── search_improvement_and_bug_audit.md
│   ├── setup.md
│   └── startup_guide.md
│
├── plans/                       # Phase-by-phase implementation plans
│
├── omni/                        # Main package
│   ├── backtest/                # Walk-forward backtest runner + metrics + checks
│   ├── cv/                      # Purged K-fold + embargo (de Prado AFML)
│   ├── dashboard/               # Streamlit UI
│   │   ├── app.py               # Main entry point (9 tabs)
│   │   ├── theme.py             # Glassmorphism CSS
│   │   ├── components.py        # Reusable widgets
│   │   ├── cache.py             # st.cache_data wrappers + demo fallback
│   │   ├── watchlist.py         # JSON-persisted favorites
│   │   ├── movers.py            # Top gainers/losers
│   │   └── charts/
│   │       ├── cosmos3d.py      # Three.js 3D Signal Cosmos
│   │       ├── corrgraph.py     # D3 force-directed correlation
│   │       └── lwchart.py       # TradingView Lightweight-Charts
│   ├── ensemble/                # Blend (XGB + rubric + specialist) + gate + pipeline
│   ├── features/                # 13 feature builders (technical, fundamental, etc.)
│   ├── ingest/
│   │   ├── prices.py            # yfinance + ccxt
│   │   ├── fundamentals.py
│   │   ├── news.py
│   │   ├── intraday.py
│   │   ├── macro.py
│   │   ├── runner.py            # Daily ingest orchestrator
│   │   └── sources/             # 16 per-source modules
│   │       ├── bse_notices.py
│   │       ├── edgar.py
│   │       ├── fmp.py
│   │       ├── fred_live.py
│   │       ├── gdelt.py
│   │       ├── indian_blogs.py  # Finshots, Trade Brains, etc.
│   │       ├── indian_rss.py    # ET, Mint, BS, Moneycontrol
│   │       ├── marketaux.py
│   │       ├── newsapi_live.py
│   │       ├── nse_flows.py     # FII/DII, bulk + block deals
│   │       ├── nse_oi.py        # F&O option chain (PCR, max-pain)
│   │       ├── polygon.py
│   │       ├── reddit_public.py
│   │       ├── sebi_insider.py  # SEBI PIT-2015 disclosures
│   │       ├── stocktwits.py
│   │       └── telegram_channels.py
│   ├── lake/                    # Parquet IO + partition paths + backfill
│   ├── llm/                     # 6-backend router (cloud + local + FinBERT)
│   │   ├── router.py
│   │   ├── cache.py
│   │   ├── ratelimit.py
│   │   └── backends/
│   │       ├── base.py
│   │       ├── groq.py          # Llama-3.1-8b-instant
│   │       ├── gemini.py        # Flash Lite
│   │       ├── openrouter.py
│   │       ├── cortex.py        # Snowflake Cortex
│   │       ├── ollama.py        # Local phi3.5 / qwen2.5
│   │       └── finbert.py       # ProsusAI/finbert sentiment
│   ├── models/
│   │   ├── xgboost_5d.py        # Universal 5-day classifier
│   │   ├── specialists.py
│   │   ├── stack.py             # XGB + HGB + logistic meta
│   │   ├── conformal.py         # Split-conformal intervals
│   │   └── training_utils.py
│   ├── monitoring/              # Append-only prediction log
│   ├── ondemand/                # Single-ticker stateless prediction
│   │   ├── universe.py          # Resolver (curated + alias + fuzzy)
│   │   ├── lookup.py            # LookupBundle (everything for 1 ticker)
│   │   ├── predict.py           # predict_ticker() + deep_analysis()
│   │   ├── features_inline.py
│   │   ├── macro_live.py
│   │   ├── regime_live.py
│   │   ├── pair_signal.py
│   │   ├── quality.py
│   │   ├── cache.py             # diskcache TTL tiers
│   │   └── model_registry.py
│   ├── orchestrator/            # Nightly daily pipeline
│   ├── regime/                  # Gaussian HMM regime classifier
│   ├── rubric/                  # 4-pillar heuristic scorer
│   ├── specialists/
│   │   └── india.py             # India-specific sub-model
│   ├── storage/                 # Tiered local + Drive storage
│   └── utils/                   # logger, env, paths
│
├── scripts/                     # Operational entry points
│   ├── backfill_universe.py
│   ├── backfill_quarterly.py
│   ├── build_training_set.py
│   ├── train_universal_model.py
│   ├── tune_universal_model.py
│   └── daily_backtest.py
│
├── tests/                       # End-to-end tests
│   ├── e2e_synthetic.py
│   └── e2e_real.py
│
├── data/                        # GITIGNORED — local data lake + caches + models
│   ├── lake/                    # Long-term parquet store
│   ├── hot/                     # Recent / on-demand cache
│   ├── features/                # Built feature parquets
│   ├── models/                  # Trained model bundles (.pkl)
│   ├── predictions/             # latest.parquet (orchestrator output)
│   ├── signals/                 # latest.parquet (gated signals)
│   ├── regime/                  # Regime label parquets
│   ├── backtest/                # Backtest reports + plots
│   ├── cache/                   # diskcache for ondemand layer
│   └── watchlist/               # User-favorited tickers (JSON)
│
├── scratch/                     # GITIGNORED — local debug scripts (_*.py)
├── logs/                        # GITIGNORED — runtime logs + stderr/stdout
└── secrets/                     # GITIGNORED — OAuth JSONs, service keys
```
