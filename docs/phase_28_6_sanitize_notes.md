# Phase 28.6 — causal OHLCV bad-tick sanitizer notes

## What & why

Added `omni/ingest/sanitize.py::sanitize_ohlcv(df, asset_class)` — a
causal, leak-free cleaner applied to raw daily OHLCV **before** feature
and target computation, at the single funnel both training and inference
share (`build_features_inline`; also `build_for_ticker`). A single corrupt
bar (vendor typo, unadjusted corporate action, phantom non-positive price,
duplicate date) otherwise poisons every rolling feature (returns, vol, RSI,
Bollinger, ...) AND the forward-return labels, since both read the same
price frame.

Rules (all leak-free — no future bar is ever consulted):
1. structural: sort by date, drop duplicate dates (keep last);
2. hard integrity: drop rows with NaN / non-positive prices, or impossible
   bars (`high < low`, `high < max(open,close)`, `low > min(open,close)`);
3. extreme move: drop a bar only when `|log-return vs prev|` breaks BOTH a
   trailing robust band (rolling median ± 8·1.4826·MAD over the prior 60
   bars, `shift(1)`) AND an absolute per-asset-class cap set ABOVE circuit
   limits (india 0.35, us 0.40, crypto 0.70, fx 0.15).

Conservative trade-off (documented in the module): an isolated spike trips
the cap on the move INTO the bad bar and the move OUT of it, so the bad bar
and its immediate neighbour may both drop. Acceptable — bad ticks are rare
and a 10x spike otherwise contaminates ~60 bars of rolling windows.

## Verification

**Unit test (synthetic):** a clean 300-bar series passes through unchanged
and idempotently; injected defects (10x spike, negative price, `high<low`,
duplicate date) are all caught; a legitimate 18% circuit-size move SURVIVES
(below the 0.35 cap) — no false positive.

**Full-lake census (207 tickers, 671,043 bars):**

| group | what it catches | examples |
| ----- | --------------- | -------- |
| integrity | 1,973 bars | fx/USDJPY 238, crypto/SHIB 218, fx/USDINR 206, fx/EURJPY 207 |
| extreme  | 125 bars | crypto/UNI 44, crypto/ARB 23, us/MARA 16, us/SNAP 3, us/NFLX 2 |
| dup      | 0 | — |

Total **2,098 bad bars across 37 tickers** (0.31% of all bars). FX feeds
carry the most integrity defects (weekend/holiday phantom bars); crypto and
US equities carry genuine extreme ticks. CIANAGRO live (2,077 bars) cleaned
0 — no false positives on a flaky BSE-only micro-cap.

**A/B retrain (india-only, --no-news, single tb purged, half-life 1.5):**

| run | OOF AUC |
| --- | ------: |
| control (sanitizer off) | 0.5261 |
| treatment (sanitizer on) | 0.5262 |
| delta | **+0.0001** |

Flat, as expected: only 3 of the 59 India training tickers are touched (a
handful of bars), so the clean NIFTY-large-cap training set is essentially
unchanged. **The value is not in the india-training AUC** — it is in
(a) inference robustness on the long-tail / micro-cap path, and (b) the
FX/crypto/US bars that a future multi-asset retrain would otherwise ingest
dirty.

## Decision

KEEP (unlike the Tier B features, which were dropped for null value). The
sanitizer does demonstrable data-quality work (2,098 real bad bars), takes
effect at inference immediately (it runs in `build_features_inline`), and
is harmless where data is already clean (no-op, idempotent, no false
positives). The production model artifact was restored to its pre-experiment
state (oof_auc=0.5156); the sanitizer ships as source and will clean the
training data on the next real (multi-asset / news-included) retrain.

Gate: an env var `OMNI_DISABLE_OHLCV_SANITIZE=1` disables it (used only for
the A/B control); on by default.

## Follow-up (deferred, researched not implemented)

* **News relevance filter** — `ticker_hint` is the query label, not a
  relevance guarantee; loose `contains` matching counts off-topic articles.
  Cheap first step: require the company name/ticker token in title/desc
  before counting; stronger: reuse the RAG embedder for article↔ticker
  similarity. Attacks noise at the sentiment-pillar source.
* **Feature pruning** — efficiency only (XGBoost already ignores irrelevant
  features; Tier B proved adding collinear features is inert). Defer until
  input-quality levers are exhausted.
