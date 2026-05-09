# Phase 20 — finish what's started + product UX (Tier 1 + Tier 3 minus alerts)

Local-only commit. **Not pushed** (per the standing rule).

## What landed

| # | Task | File(s) | Status |
|---|---|---|---|
| 1.1 | Plumb microstr data sources through `lookup.py` | `omni/ondemand/lookup.py`, `omni/ondemand/oi_archive.py` (new) | live |
| 1.3 | ADWIN drift gate wired into daily backtest | `scripts/daily_backtest.py` | live |
| 1.4 | `OPENROUTER_API_KEY` already declared in `.env.example` | — | doc-only |
| 3.9 | Portfolio mode tab (Kelly + sector cap) | `omni/dashboard/app.py::_render_portfolio_tab` | live |
| 3.10 | Backtest replay slider in History tab | `omni/monitoring/replay.py` (new) + `_render_replay_section` | live |
| 3.11 | Earnings-day specials tab | `omni/ingest/earnings_calendar.py` (new) + `_render_earnings_tab` | live |
| 1.2 | Retrain — **deferred to user** (will run with GPU tomorrow) | `scripts/retrain_all.py` (existing) | ready |
| 3.12 | Live alerts — **dropped from this phase** per user | — | future |

Tab count: **9 → 11** (added Portfolio, Earnings).

## The breakthrough — real microstructure signal on SBIN

After Phase 19, microstructure features were NaN in production because
`lookup.py` didn't plumb the data sources. Phase 20 fixes that.

**Verified live on SBIN (2026-05-09):**

| Feature | Phase 19 (NaN) | Phase 20 (live) |
|---|---|---|
| `delivery_pct_z_60d` | 0/60 | **43/60 non-null**, latest = `-0.78` |
| `amihud_illiq_20d` | 60/60 | 60/60 (already worked) |
| `microstr_score` | 0/60 | **43/60 non-null**, latest = `-0.78` |
| `bhavcopy` rows in bundle | 0 | 60 |
| `oi_history` rows | 0 | 1 (today's first snapshot) |
| `fii_dii` rows | 0 | 1 |

Note: `oi_change_pct_5d` still 0 because OI archive has only 1 day of
history. Will populate naturally as `oi_archive.snapshot_and_history()`
runs daily. The on-disk archive at `data/lake/oi_snapshots/<TICKER>.parquet`
appends one row per ticker per day.

## How to use the new tabs

### Portfolio
1. Star a few tickers in the Predict tab.
2. Open the **Portfolio** tab.
3. Adjust the conviction slider (default 70) and per-sector cap (default 30%).
4. See: total exposure, long/short split, avg Kelly edge, the recommended book sized to capital.

### History → Replay
1. Open the **History** tab — existing batch-signals table at the top.
2. Scroll past the divider for the new **Backtest replay** section.
3. Drag the date slider over the last 180 days; toggle "Only direction != FLAT".
4. See: hit rate, mean realised %, every logged prediction with its realised forward return.

### Earnings
1. Open the **Earnings** tab.
2. Top section auto-shows tracked tickers reporting in next 7 days.
3. Bottom section is a per-ticker drill-down with the last 4-quarter surprise % history.

## Deferred to next session (with GPU)

- **Retrain universal model** with the 38 new feature columns. The data
  pipe is now live, so `scripts/retrain_all.py` will pick up
  microstructure + embeddings naturally next time it runs.

## Pending pushes

Local-only commit awaiting your sign-off. Run `git push origin main` when ready.
