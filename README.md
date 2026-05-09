# Omni_Stocks

A multi-asset, multi-LLM market prediction system for Indian (NSE), US, crypto,
and FX markets. Combines technical, fundamental, sentiment, and macro signals
through a regime-aware rubric layered on top of calibrated XGBoost classifiers.
Designed to run locally on Windows + an RTX 4070 Super with Google Drive cold
storage and Snowflake Cortex for occasional heavy LLM synthesis.

> **Status: production-ready.** All 9 dashboard tabs render real
> predictions, multi-source ticker search is live, model stack is
> trained.

> 📚 **For a complete from-scratch explanation of every concept,
> module, and design decision in this project — read the master
> guide:**
> **[`docs/interview_prep_master_guide.md`](docs/interview_prep_master_guide.md).**
> Everything you need to defend the project end-to-end in a technical
> interview is in there.

Other docs:

- [`docs/architecture.md`](docs/architecture.md) — original phase plan
- [`docs/architecture_walkthrough.md`](docs/architecture_walkthrough.md) — module-by-module deep dive
- [`docs/project_tree.md`](docs/project_tree.md) — annotated directory layout
- [`docs/setup.md`](docs/setup.md) — manual machine setup (Python, Ollama, rclone, API keys)
- [`docs/startup_guide.md`](docs/startup_guide.md) — first-run quickstart

## What it does

1. Ingests OHLCV, fundamentals, news, and macro series for **64 instruments**
   (20 NSE + 30 US + 10 crypto + 4 FX).
2. Engineers technical, fundamental, sentiment, and macro features.
3. Detects market regime (bull / bear / chop / crisis) via a 4-state HMM.
4. Trains a 5-day-horizon XGBoost classifier with isotonic calibration.
5. Combines model probability with a regime-conditional 4-pillar rubric.
6. Emits **confidence-gated** signals only when conviction >= configured threshold.
7. Renders a Streamlit dashboard with explanations from a multi-LLM router.

## Realistic accuracy targets

| Slice                        | Target          |
|------------------------------|-----------------|
| Overall directional accuracy | 54-60%          |
| Top-decile gated signals     | 65-72%          |
| Calibration (Brier score)    | <= 0.22         |

The headline differentiators are:
- Regime-conditional weighting (signals weighted differently per regime)
- Explicit conviction gating (no signal is always better than a weak signal)
- Multi-LLM router (free-tier first, paid Snowflake Cortex on demand)

## Repository layout

```
C:\Omni_Stocks\
  config\               YAML configs (universe, sources, rubric, llm_routes)
  data\
    hot\                Cleaned OHLCV (local SSD)
    features\           Engineered feature parquets
    models\             Trained model artifacts (.joblib)
    predictions\        Daily prediction outputs
    cache\llm\          diskcache for LLM responses
  docs\                 Documentation (architecture, setup)
  omni\
    utils\              paths, logger, env/yaml loader
    storage\            Tier-aware parquet I/O (local + Drive)
    ingest\             prices, fundamentals, news, macro
    llm\                Multi-backend router with cache + rate limit
      backends\         finbert, ollama, openrouter, cortex
    features\           (Phase 4)
    regime\             (Phase 5)
    models\             (Phase 5)
    rubric\             (Phase 5)
    ensemble\           (Phase 6)
    backtest\           (Phase 7)
    dashboard\          (Phase 8)
  tests\                pytest suite (TBD)
  pyproject.toml
  .env / .env.example   Secrets (.env is gitignored)
  README.md
```

Cold/raw data lives on **Google Drive** mounted via rclone at `Z:\Omni_Stocks\`.
Hot/processed data lives on **local SSD**. The split is invisible to callers --
all I/O routes through `omni.storage.tiers`, which falls back to a local
`raw_fallback/` directory if the Drive mount is unavailable.

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

# Ingest a small slice
python -c "from omni.ingest.prices import fetch_prices; print(fetch_prices('us','AAPL',start='2024-01-01',end='2024-02-01').shape)"
```

See [`docs/setup.md`](docs/setup.md) for the full setup including Ollama,
OpenRouter, NewsAPI, FRED, and Snowflake Cortex.

## Configuration

All behavioral knobs live in `config/*.yaml`:

| File              | Controls                                                  |
|-------------------|-----------------------------------------------------------|
| `universe.yaml`   | Tickers, suffixes, lookback years, macro reference series |
| `sources.yaml`    | API keys (by env name), RSS feeds, FRED series list       |
| `rubric.yaml`     | Pillar weights per regime + conviction gate threshold     |
| `llm_routes.yaml` | Backends, rate limits, primary -> fallback chains, cache  |

Edit YAML, restart -- no code changes required for routing/weighting tweaks.

## Documentation index

- [`docs/architecture.md`](docs/architecture.md) -- Subsystem design, data flow,
  storage layout, LLM routing, and phase roadmap.
- [`docs/setup.md`](docs/setup.md) -- Manual machine setup (Python, Ollama,
  rclone, API keys).

## License & scope

Personal research project. Not investment advice. Free-tier APIs only;
paid services (Cortex) used selectively.
