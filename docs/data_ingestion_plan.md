# Data-ingestion plan (concrete — no fake data)

Grounded in what actually works in this environment (verified this session):
`cortex secret list` is **empty** (no API keys), and **yfinance works keyless**.
So the plan is split by what is available now vs. what needs a key.

## Feeds

| feed | key? | status | what it gives |
| ---- | ---- | ------ | ------------- |
| **yfinance** (prices) | no | **working** | daily OHLCV+adj_close, any NSE/US/crypto/fx ticker. Used to ingest the 56-name small/mid-cap panel. |
| yfinance (earnings dates) | no | available | `Ticker.get_earnings_dates()` — for PEAD/event experiments. |
| NSE bhavcopy (public) | no | available | EOD incl. **delivery %** (illiquidity proxy) — public daily zip. |
| FMP / Polygon / Tiingo | yes | **blocked** (no key) | fundamentals, delisted/survivorship-clean universes. |
| newsapi / marketaux / currents | yes | **blocked** (no key) | news for the RAG/sentiment path. |

## Schema (matches existing `omni.ingest.prices` output)

`date, open, high, low, close, adj_close, volume, ticker, asset_class, source`
— long-format panel, one row per (ticker, date). Stored as parquet under
`data/lake/` (gitignored — data stays local, not committed).

## Integration points

1. **Universe config** — promote the hard-coded small/mid-cap list to a config
   (e.g. `configs/universe_smallcap.yaml`) so the experiment is reproducible.
2. **`omni.ingest.prices.fetch_prices`** — already the single ingestion entry;
   reused as-is for the small/mid-cap fetch. No new code path needed.
3. **`build_training_set`** — point it at the small/mid-cap panel to run the full
   XGBoost+stack on real down-cap data (next experiment).
4. **Delivery-% feature** — add NSE bhavcopy delivery ratio as an illiquidity
   feature; theory says reversal/illiquidity premium concentrates there.

## Survivorship — the gating risk

yfinance = **current survivors only**; delisted losers are gone, which flatters
any "buy the losers" reversal leg. Before trusting the down-cap result (Phase
28.22) as tradable, the universe must include delisted names — which requires a
**keyed feed (FMP/Polygon)** or a point-in-time NSE constituents archive. Until
then, treat down-cap reversal numbers as an **upper bound**, not a live edge.

## Sequenced next steps (no fabrication, all real)

1. Fix `CANFIN`→`CANFINHOME`, widen universe to ~150 names via yfinance.
2. Add NSE bhavcopy delivery-% ingestion (keyless, public).
3. Run the full model/ensemble on the real small/mid-cap panel under
   `OMNI_CV_MODE=strict`; report net-of-cost only.
4. Acquire a survivorship-clean universe (needs an API key) before any claim of
   a deployable down-cap signal.
