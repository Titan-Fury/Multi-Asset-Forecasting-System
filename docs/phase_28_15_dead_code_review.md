# Phase 28.15 — dead / redundant code review

Review for "redundant code, code which doesn't make sense, uselessly used".
Tools: `pyflakes` (unused imports/vars, bad f-strings) + `vulture`
(unused functions/vars) over `omni/` and `scripts/`.

## Removed (commit `a222cad`) — verified zero callers, compile + imports clean

| item | file | why |
| ---- | ---- | --- |
| `walkforward_splits()` (33-line fn) | `omni/models/xgboost_5d.py` | Superseded expanding-window splitter, no callers anywhere. Main CV uses `purged_kfold_indices`; honest-WF now uses `strict_walkforward_splits`. |
| `cov` "sanity stub" + orphaned import | `scripts/train_conformal.py` | Called `empirical_coverage` on dummy `[0],[0],[1]` then never used; real coverage is `emp_cov`. Made no sense. |
| `severity` import | `scripts/monitor_drift.py` | Imported, never used. |
| `joblib` import | `scripts/train_tb_and_meta.py` | Imported, never used. |
| `f"[4/4] done."` | `scripts/train_conformal.py` | f-string with no placeholders. |

## Deliberately KEPT — tool false positives (intentional, NOT dead)

These look "unused" to pyflakes/vulture but are load-bearing; removing them would
break feature detection or the public namespace:

- **Availability-probe imports inside `try/except`** that set a `_HAS_*` flag or
  fall back to `None`: `ingest/prices.py` (`_ccxt` → `_HAS_CCXT`),
  `storage/mount.py` (`google.colab` for Colab detection),
  `models/nhits.py` + `models/patchtst.py` (`neuralforecast` — optional deep
  models), `llm/rag/retriever.py` (`BM25Okapi` probe). The *import attempt itself*
  is the check.
- **`omni/lake/__init__.py`** `from . import paths, io, backfill` — intentional
  namespace re-exports (`omni.lake.paths` access).
- **`features/build.py`** `from pathlib import Path  # noqa: F401` — explicit
  re-export (pyflakes doesn't honor `noqa`; flake8 does).
- **`production_runner.py` `signum`** — POSIX signal-handler signature
  `(signum, frame)`; the arg is required even when unused.

## Remaining low-priority (left as-is; cosmetic, no behavior impact)

A handful of `f-string is missing placeholders` in `scripts/` (backfill_news,
benchmark_sbi, probe_predictions, train_circuit_risk, train_regime_stacks,
train_universal_model). Harmless logging strings; not worth churn-commits across
six files. Flagged here for a future sweep if a lint gate is added.

## Bottom line

The codebase is **lean** — one genuinely dead function and a few stray
imports/vars, now removed; everything else flagged by the tools is intentional
infrastructure. No large redundant subsystems or duplicate logic found. A
`flake8`/`ruff` CI gate would keep it that way and auto-catch the cosmetic
f-string items.
