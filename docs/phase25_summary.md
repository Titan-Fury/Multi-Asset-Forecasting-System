# Phase 25 — Empirically-driven model + UX improvements

Phase 25 was triggered by the May 8 2026 CIANAGRO crash backtest. The
quant model correctly turned bearish on every name in a 7-ticker basket
(CIANAGRO, RELIANCE, TCS, HDFCBANK, IRCTC, PAYTM, BANKBARODA), but
**every signal was structurally blocked** by either the legacy hard
regime gate or the static `min_conviction = 15` floor. Realised h+5
returns ranged from -0.95 % (BANKBARODA) to -22.62 % (CIANAGRO).

The phase landed nine commits, each shippable independently and all
with opt-in env flags where production-affecting.

## Backtest that motivated the phase

```
ticker        p_up  conv  dir  h+5_real%  verdict
CIANAGRO      0.32  12.1   -1     -22.62  Cautious Bearish (warn)
RELIANCE      0.40   4.6   -1      -6.88  Sit Out
TCS           0.39  15.9   -1      -5.45  Cautious Bearish
HDFCBANK      0.41   8.4   -1      -1.71  Cautious Bearish
IRCTC         0.40   8.6   -1      -5.12  Cautious Bearish
PAYTM         0.39   7.5   -1      -4.71  Sit Out
BANKBARODA    0.43  10.5   -1      -0.95  Cautious Bearish
```

Direction column = quant model's call as-of May 8 close. h+5_real% =
realised price change to May 15 close. The model was correct 7/7 yet
nothing emitted because:

- HMM was 0.999-confident **bull** at the literal local top.
- Legacy regime filter blocked all `direction == -1` signals in bull.
- Even TCS at conviction 15.9 (above the 15 threshold) didn't pass
  because the regime block was binary, not graded.

## What landed

| # | Commit subject | Module / file |
|---|----------------|---------------|
| 1 | feat(ondemand): patch live insider/FII-DII/block-deal alt-data on prediction row | `omni/ondemand/altdata_patch.py` |
| 2 | feat(train): per-horizon recency half-life for universal model | `scripts/train_universal_model.py` |
| 3 | feat(calibration): opt-in beta calibration via OMNI_CAL_METHOD=beta | `omni/calibration/{beta_cal,factory}.py` |
| 4 | feat(features): difference encoding + lagged realised labels | `omni/features/diff_lagged.py` |
| 5 | feat(models): regime-conditional stack routing (opt-in) | `omni/models/regime_stack.py` |
| 6 | feat(ui): plain-English verdict banner on prediction cards | `omni/dashboard/verdict.py` |
| 7 | feat(gate): soft regime multiplier + per-(asset_class, regime) min_conviction | `omni/ensemble/gate.py` |
| 8 | feat(storage): point-in-time fundamentals snapshots for replay reproducibility | `omni/storage/fundamentals_snapshots.py` |
| 9 | feat(features): triple-barrier as default target + circuit-aware label masking | `omni/features/circuit.py` |
| 10 | feat(rubric): learned per-regime weights via opt-in OMNI_LEARNED_RUBRIC | `omni/rubric/learned.py` |
| 11 | feat(monitoring): PSI drift monitor + daily script | `omni/monitoring/drift.py` |
| 12 | fix(dev-tab): repair broken Refit-regime + Run-daily-orchestrator buttons | `scripts/refit_regime.py`, `scripts/run_daily.py` |
| 13 | chore: remove redundant probe scripts superseded by per-feature tests | `scripts/probe_*.py` cleanup |
| 14 | perf(rubric, gate): vectorize hot-path lookups, avoid full-frame copy | `omni/rubric/scorer.py`, `omni/ensemble/gate.py` |

## Decision log

### Soft regime gate (commit 7)

Replaced the binary block with a `(regime, direction)` multiplier
applied to conviction *before* the threshold check:

```
direction \\ regime  bull  chop  bear  crisis
long  (+1)           1.00  0.90  0.50  0.30
short (-1)           0.50  0.90  1.00  0.80
```

Synthetic test: 5/7 counter-trend + aligned signals pass under soft vs
1/7 under legacy hard. The setting is `gating.soft_regime: true` in
`config/rubric.yaml`; flip to `false` for backtest reproducibility.

### Triple-barrier default + circuit-bar masking (commit 9)

`scripts/train_universal_model.py` now defaults to `--target tb`. When
`tb_label_<H>d` columns aren't present the trainer logs a warning and
falls back to `fwd_dir`, so the change is safe for users who haven't
run `scripts/augment_triple_barrier.py` yet.

`omni.features.circuit.detect_circuit_bars` flags Indian-equity bars
where `open == high == low == close` AND `|daily_ret|` matches a
+/-5 / 10 / 20 % band (with tolerance). `mask_circuit_labels` NaNs out
`fwd_dir_<H>d` and `tb_label_<H>d` on those source rows, so they get
dropped from training via the standard `dropna(subset=[target])`
pattern. Verified on real CIANAGRO data — 16/59 bars correctly flagged
across the parabolic + crash sequence.

### Verdict banner (commit 6)

`omni.dashboard.verdict.verdict(card_dict)` returns a 4-field
`Verdict(headline, summary, disagreement, tone)`. The dashboard
renders an amber `warn` banner with an explicit "model bearish but
pillars still constructive — late-cycle warning" sentence whenever the
quant model and the rubric pillars disagree on direction AND the gate
is blocked. This is the missing piece that would have surfaced the May
8 CIANAGRO bearish call to the user even though the gate didn't emit.

### Opt-in / opt-out config

| Variable | Default | Effect |
|----------|---------|--------|
| `OMNI_CAL_METHOD` | `isotonic` | `beta` opts into beta calibration in walk-forward training |
| `OMNI_REGIME_STACKS` | unset | `1` loads `data/models/regime_stacks_h5.joblib` and routes by HMM `proba_today` |
| `OMNI_LEARNED_RUBRIC` | unset | `1` blends YAML rubric weights with the data-fit posterior |
| `OMNI_LEARNED_RUBRIC_ALPHA` | `0.5` | Learned-prior blend weight (0 = pure YAML, 1 = pure learned) |
| `gating.soft_regime` (yaml) | `true` | When true, regime filter multiplies conviction; when false, legacy binary block |
| `gating.min_conviction_overrides` (yaml) | per-asset map | Per-(asset_class, regime) thresholds; missing keys fall back to the global `min_conviction` |

## Verifications committed alongside the changes

| Probe | What it asserts |
|-------|-----------------|
| `scripts/probe_soft_gate.py` | 5/7 pass under soft vs 1/7 hard on synthetic frame |
| `scripts/probe_threshold_overrides.py` | Per-(asset_class, regime) lookup correct across 16 combinations |
| `scripts/probe_fundamentals_pit.py` | PIT snapshot save/load round-trip + idempotent rewrite |
| `scripts/probe_circuit_detection.py` | 16/59 CIANAGRO bars flagged; AAPL returns 0 |
| `scripts/probe_learned_rubric.py` | YAML * 0.5 + learned * 0.5 produces normalised weights; scorer respects env flag |
| `scripts/probe_verdict.py` | Headline + summary + disagreement strings for known-state cards |
| `scripts/probe_replay_universe.py` | End-to-end May 8 replay across 7 names |
| `scripts/bench_score_frame.py` | Optimized score_frame ~17.9k rows/s on 5000-row synthetic |

## Performance commit

Two hot-path changes validated by `scripts/bench_score_frame.py`:

- `omni.rubric.scorer.score_frame` — replaced `df.iterrows()` plus a
  full-frame `df.copy()` (~200 MB on the 700k-row pooled training
  frame) with `df.to_dict("records")`, pre-allocated np arrays, and a
  single `df.assign(...)` at the end. Bench: **17.9k rows/s** on 5000
  random rows.
- `omni.ensemble.gate.apply` — vectorized the (regime, direction)
  multiplier and the per-row threshold lookup via flat tuple-keyed
  dicts; replaced O(n) Python method calls with O(n) C dict.get.

## Dev-tab fix (commit 12)

Two dashboard buttons silently failed because their target scripts
didn't exist:

- "Refit regime (HMM)" called `retrain_all.py --regime-only` — that
  flag was never wired.
- "Run daily orchestrator now" called `scripts/run_daily.py` which was
  missing entirely; only the `.bat` wrapper existed.

Fix: new `scripts/refit_regime.py` (HMM refit + parquet persistence)
and `scripts/run_daily.py` (Python wrapper around
`omni.orchestrator.production_runner`). Verified live: refreshing the
regime parquet pushed it from `780 rows ending 2024-12-31` to
`2011 rows ending 2026-05-15`.

## Items NOT done in Phase 25

| Item | Why deferred |
|------|--------------|
| Sequence model (PatchTST / TabPFN-TS) | NHITS already at 0.4974 OOF AUC; sequence models add heavy deps without guaranteed wins on daily-frequency Indian equities |
| Custom `min_conv` per liquidity bucket | Overlaps with #2 (per-(asset_class, regime)); add only if backtests show liquidity-stratified gates dominate the simpler split |
| Production model retrain on TB labels | Default flipped (commit 9) but actually retraining requires running `scripts/augment_triple_barrier.py` then `scripts/train_universal_model.py --force`; user-driven |

## Files removed during the phase

| Path | Reason |
|------|--------|
| `scripts/probe_altdata_wiring.py` | Covered by `probe_replay_universe` |
| `scripts/probe_cianagro_e2e.py` | Superseded by `probe_replay_universe` |
| `scripts/probe_cianagro_replay.py` | Superseded by `probe_replay_universe` |
| `scripts/probe_multi_ticker.py` | Pre-Phase-25 baseline |
| `scripts/smoke_compare.py` | Phase-24.18 verification, stale |

Filesystem cleanup (already gitignored, just wiped from disk):
`scratch/` contents (95 stale debug files), `lightning_logs/` (74 NHITS
training-run dirs), `omni_stocks.egg-info/`, `.pytest_cache/`,
`.ruff_cache/`, 29 `__pycache__/` dirs.
