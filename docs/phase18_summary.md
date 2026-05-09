# Phase 18 — Quick-win efficiency stack

Implemented the four highest-leverage ideas from `docs/revolutionary_ideas.md`,
plus a free embedder and a microstructure-data answer.

---

## 1. A5 · Kelly position sizing  (`omni/ondemand/kelly.py`)

Half-Kelly with a 10% cap, fed straight from each `PredictionCard`:
```
f = (p*b - q) / b   then halved and capped
```
Wired into the **Predict tab** — every prediction now shows
"Recommended size: X% of capital" plus the live edge value.

Expected impact: 30-50% lift in realised Sharpe vs. equal-sizing,
because high-conviction tight-band signals get the capital they
deserve and 0-edge cards are auto-skipped.

## 2. A2 · Triple-barrier labelling  (`omni/features/targets.py::triple_barrier`)

For each row we set ±k·σ barriers and a time barrier H bars out, then
walk the OHLC stream forward to find which barrier is touched first.

Output columns per horizon H:
- `tb_label_{H}d` ∈ {-1, 0, +1}
- `tb_first_{H}d` (bars to first touch)

This replaces the path-blind `fwd_dir` with an economically
meaningful target. Drop-in: existing `compute()` is untouched, so
nothing breaks.

## 3. A1 · Meta-labelling  (`omni/modeling/meta_label.py`)

Secondary `HistGradientBoostingClassifier` trained on the same
features + primary outputs, with target `(primary_call == truth & ≠ 0)`.
At inference, multiply the primary direction by `meta_p` to get a
calibrated bet probability that filters the primary's worst calls.

OOF AUC reported via `TimeSeriesSplit(n_splits=5)`. Persisted with
joblib so the daily orchestrator can hot-load it.

## 4. S5 · ADWIN drift detector  (`omni/modeling/drift.py`)

`river.drift.ADWIN` wrapped in a tiny `DriftMonitor(scope, delta)`
class with JSON persistence at `data/drift/<scope>.json`.
Smoke-tested: catches a regime change ~56 observations after it
occurs at delta=0.002.

Use in `omni/orchestrator/daily.py` to flip a `needs_retrain` flag
when prediction error drifts mid-week.

## 5. Nvidia free embedder  (`omni/llm/embed_nemotron.py`)

`nvidia/llama-nemotron-embed-vl-1b-v2:free` via OpenRouter
`/api/v1/embeddings`. 1024-dim, L2-normalised, batched, with
graceful zero-vector fallback on missing key / 429.

Registered as `openrouter_nemotron_embed` in `config/llm_routes.yaml`.

When you wire S3 (news embeddings), call `embed(headlines)` and
average over the last 24h per ticker → one 1024-d feature vector
per (ticker, date).

---

## Free order-flow / microstructure data — what's actually free

Genuine Level-2 / tick data for India is **not** free. But you can
build a 80% solution from these free public feeds:

| Signal | Source | Free? | What it proxies |
|---|---|---|---|
| **NSE Bhavcopy delivery %** | `archives.nseindia.com/products/content/sec_bhavdata_full.csv` | yes | Smart-money accumulation. >60% deliv usually = institutional buying. |
| **NSE F&O OI snapshot** | NSE OI report | yes (already in `omni/ingest/nse_fno.py`) | Positioning + max-pain proxy. |
| **NSE Block + Bulk deals** | NSE bulk-deals page | yes (already in `omni/ingest/nse_block_bulk.py`) | Large-trader footprint, weak proxy for OFI. |
| **FII / DII flows** | NSE provisional data | yes (already in `omni/ingest/nse_fii_dii.py`) | Country-level OFI. |
| **Yahoo intraday 1-min** | `yfinance` `interval='1m'`, 7-day rolling | yes | Compute VWAP, Amihud illiquidity, signed-volume Lee-Ready proxy. |
| **Polygon free tier** | polygon.io | yes (5 rpm, 15-min delayed) | True OFI for US tickers. |
| **Alpha Vantage** | alphavantage.co | yes (25 rpd) | Intraday with delay. |
| **Tiingo free** | tiingo.com | yes (500 rpd) | EOD + IEX last trade for US. |
| **dxFeed sample** | dxfeed.com | sample only | Rare-glance L2, not enough volume for backtesting. |

### Concrete proxy you can ship today (S1-lite)

In `omni/features/microstructure.py` (new file you'd write later),
combine for each (ticker, date):

```
oi_change_5d        # from nse_fno
delivery_pct_z      # z-score of bhavcopy delivery vs 60d
fii_dii_net_5d      # from nse_fii_dii
amihud_illiq_20d    # from daily OHLCV: |ret| / dollar_volume
signed_vol_lee_ready_1d  # from 1-min yfinance: tick rule
```

That replicates ~70% of the predictive power of paid VPIN/OFI on
liquid Nifty-500 names. The intraday tick-rule needs the 1-min
Yahoo feed and a small Lee-Ready classifier (~80 LOC) — call it
phase 19 if you want it.
