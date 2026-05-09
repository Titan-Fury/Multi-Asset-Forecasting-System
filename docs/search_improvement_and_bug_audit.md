# Search Improvement Plan + Deep Bug Audit

Last updated: 2026-05-08

This document covers:

1. A practical plan to make stock search seamless and reliable.
2. A deep bug list from the current codebase (beyond the already-known stock lookup issue).

---

## Part A — Search Improvement Plan

## A1) Current pain points

- Unknown queries are currently treated as US tickers by fallback logic.
- Search is mostly exact match + simple substring; no robust ranking/confidence.
- Curated universe is hardcoded and not a proper instrument master.
- UI does not strongly enforce candidate confirmation on low-confidence matches.
- Resolver telemetry is limited, so quality issues are hard to measure.

## A2) Recommended architecture

Build a deterministic `SearchResolver` with staged flow:

1. Normalize query (casefold, punctuation/space cleanup, suffix handling).
2. Retrieve candidates from instrument index.
3. Score/rank candidates with deterministic rules.
4. Calibrate confidence and apply resolution policy:
   - High confidence: auto-select
   - Medium confidence: user confirmation
   - Low confidence: unresolved + suggestions
5. Emit resolver telemetry event.

## A3) LLM vs internet/live search (what to use)

- Primary resolver should be deterministic (instrument master + fuzzy ranking).
- Internet/live sources are best for refreshing instrument-master data, not ad-hoc per keystroke.
- LLM should be secondary only for query rewriting/alias hints in low-confidence cases.
- Do not let LLM be final symbol authority.

## A4) Phased implementation plan

### Phase 1 (1-2 days) — Safety + UX quick wins

- Remove silent US fallback for unknown input.
- Add deterministic score + confidence to resolver.
- Add explicit candidate picker for medium/low confidence.
- Surface `match_reason` and confidence in Predict UI.
- Add resolver event logging (query, selected ticker, confidence, latency).

Likely files:

- `omni/ondemand/universe.py`
- `omni/dashboard/app.py`
- `omni/dashboard/components.py`
- `omni/monitoring/predict_log.py` (or new search log module)

### Phase 2 (1-2 weeks) — Robust search core

- Introduce canonical instrument master (symbol, exchange, aliases, status, provider symbols).
- Build in-memory indexed lookup and cache by master version hash.
- Add ranking features: exactness, prefix, alias hit, token overlap, edit distance, exchange prior.
- Add strict ambiguous/unresolved policy (no silent remap).

Likely files:

- `config/universe.yaml`
- `omni/ingest/instrument_master.py` (new)
- `omni/ondemand/universe.py`
- `omni/ondemand/cache.py`

### Phase 3 — Production-grade quality

- Versioned daily master build + validation checks.
- Feedback loop from user correction events.
- Search quality dashboard metrics:
  - unresolved rate
  - ambiguity rate
  - wrong-market rate
  - manual correction rate
  - p95 resolver latency

## A5) Search test plan

- Unit tests: exact, alias, suffix, typo, ambiguity, deterministic tie-break.
- Integration tests: query -> resolver -> lookup behavior for resolved/ambiguous/unresolved.
- Telemetry tests: every resolve attempt logs structured event.

---

## Part B — Deep Bug Audit (Beyond Search)

Severity order: highest first.

## B1) HIGH — Daily flow likely uses historical OOF predictions, not live current-day scoring

- **Symptom:** Dashboard/signal outputs may reflect historical out-of-fold rows rather than today-only actionable rows.
- **Root cause:** Daily orchestration uses `walkforward_fit(...).predictions_frame()` and downstream signal assembly inner-joins on historical `(date, ticker)`.
- **Impact:** Potentially stale/incorrect “daily” signals.
- **Code areas:** `omni/orchestrator/daily.py`, `omni/backtest/runner.py`, `omni/orchestrator/persist.py`.
- **Fix direction:** Split backtest/OOF path from live scoring path; for daily run, load saved model artifact and score latest row per ticker only.

## B2) HIGH — Train/live feature skew for cross-sectional rank features

- **Symptom:** Live inference quality can degrade unpredictably.
- **Root cause:** `build_features_inline` computes cross-sectional ranks on single ticker (degenerate percentile behavior), while training recomputes ranks after pooling.
- **Impact:** Feature distribution mismatch between training and live inference.
- **Code areas:** `omni/ondemand/features_inline.py`, `omni/features/cross_sectional.py`, `scripts/build_training_set.py`.
- **Fix direction:** Do not compute cross-sectional ranks in single-ticker live path unless peer cross-section is provided; align training/live feature contract.

## B3) HIGH — Prediction cache key can return stale cards after retraining

- **Symptom:** New model artifacts may not reflect immediately in predictions.
- **Root cause:** Cache key for `predict_ticker` uses only `{ticker}:{asset_class}` with long TTL.
- **Impact:** Users see stale predictions post-retrain.
- **Code areas:** `omni/ondemand/predict.py`, `omni/ondemand/cache.py`.
- **Fix direction:** Include model version/hash/mtime in cache key or invalidate cache on model update.

## B4) MEDIUM-HIGH — Recency weighting CLI option appears non-functional

- **Symptom:** Changing recency half-life may have no effect.
- **Root cause:** Weights are computed in trainer but not passed through fit/calibration.
- **Impact:** Expected recency bias not applied.
- **Code areas:** `scripts/train_universal_model.py`.
- **Fix direction:** Wire sample weights into model training path or remove the option until implemented.

## B5) MEDIUM — Daily backtest docs/options claim Brier drift alerting, but alert logic only checks AUC

- **Symptom:** Calibration drift can go undetected.
- **Root cause:** `--brier-degrade-pct` argument exists but is not used in alert condition.
- **Impact:** Monitoring blind spot.
- **Code areas:** `scripts/daily_backtest.py`.
- **Fix direction:** Store baseline Brier and compare rolling/current Brier against threshold.

## B6) MEDIUM — Correlation tab likely passes `read_prices` arguments in wrong order

- **Symptom:** Correlation graph may appear empty/incomplete despite available data.
- **Root cause:** `read_prices` signature is `(ticker, asset_class)` but one call appears as `(asset_class, ticker)`.
- **Impact:** UI inconsistency/missing graph data.
- **Code areas:** `omni/lake/io.py`, `omni/dashboard/app.py`.
- **Fix direction:** Use keyword args or fix positional order.

## B7) MEDIUM — Crypto source provenance can be mislabeled after fallback

- **Symptom:** Provenance may say `ccxt` even when yfinance fallback provided data.
- **Root cause:** `source` is set by asset class, not by actual successful provider.
- **Impact:** Data lineage/debug confusion.
- **Code areas:** `omni/ingest/prices.py`.
- **Fix direction:** Track actual backend used and assign source accordingly.

---

## Part C — Priority action list

### Immediate (today/tomorrow)

1. Fix daily live-vs-OOF scoring separation.
2. Fix train/live feature parity for cross-sectional ranks.
3. Add model-version-aware prediction cache invalidation.

### Next (this week)

4. Implement search Phase 1 safety changes (no unsafe fallback + confidence + confirm flow).
5. Fix monitoring gap (`brier` alert).
6. Fix dashboard correlation argument order.

### Next sprint

7. Instrument master + indexed resolver (search Phase 2).
8. Resolver telemetry dashboard + automated resolver tests.

---

## Part D — Residual risk and test gaps

- This audit is static/read-only; runtime confirmation should be done via targeted E2E checks.
- Biggest remaining unknowns:
  - artifact/data shape variance in real lake snapshots
  - feature parity under edge-market conditions
  - cache staleness under frequent retraining cycles

Recommended verification gates:

- Daily run smoke: `signals/latest` contains latest date rows only.
- Feature parity check: live feature distribution against training reference.
- Cache test: retrain model then ensure new prediction differs when expected.
- Monitoring test: forced Brier degradation emits alert file.

