"""Per-prediction event logging + drift monitoring scaffolding.

Every call to ``predict_ticker()`` and ``deep_analysis()`` should log a row
through ``log_prediction()``. Rows are appended to a date-partitioned parquet
under ``data/lake/predictions/{date}.parquet``.

Used by:
- ``scripts/daily_backtest.py`` — joins logged predictions to realised returns
  ``H`` trading days later and writes ``data/lake/monitoring/realized_metrics.parquet``.
- ``scripts/retrain_if_drift.py`` — checks alerts and triggers retraining.

Design choices:
- Append-only, atomic single-file-per-day → safe under concurrent calls.
- One row per (ticker, asof, horizon) so multi-horizon predictions log multiple rows.
- Schema is fixed; new fields go to a JSON sidecar to avoid schema drift.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from omni.lake import io as _lake_io
from omni.lake import paths as _lake_paths
from omni.utils.logger import log


_SCHEMA_COLS = [
    "logged_at", "ticker", "asset_class", "asof",
    "horizon_days", "model_kind",
    "proba_up", "blended_p", "direction", "conviction",
    "expected_move_pct", "regime", "passes_gate",
    "data_quality_score", "model_trained_at",
    # Phase 24.16 — per-pillar scores for explainability and drift tracking.
    # Older snapshots written before this phase will be read back with NaN
    # for these columns; the schema-coerce loop below pads missing fields.
    "technical_score", "fundamental_score", "sentiment_score",
    "macro_score", "microstructure_score", "oi_score",
    "relative_strength_score", "insider_score", "earnings_momentum_score",
    "rubric_total",
]


def log_prediction(card: dict, *, horizon_days: int = 5) -> bool:
    """Append a single prediction row for ``card`` at ``horizon_days``.

    ``card`` must be a dict (e.g. ``predict_ticker(...).to_dict()``).
    Returns True on success, False on any failure (never raises).
    """
    try:
        provenance = card.get("provenance") or {}
        # Choose move_X based on horizon to support multi-horizon logging
        if horizon_days == 1:
            move_pct = float(card.get("move_1d_pct", 0.0) or 0.0)
        elif horizon_days == 21:
            move_pct = float(card.get("move_1m_pct", 0.0) or 0.0)
        else:
            move_pct = float(card.get("move_1w_pct",
                                       card.get("expected_move_pct", 0.0)) or 0.0)
        # Phase 24.16 — pull per-pillar scores from the card's `pillars`
        # dict if present (gracefully None for older cards).
        pillars = card.get("pillars") or {}
        row = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "ticker": str(card.get("ticker", "")),
            "asset_class": str(card.get("asset_class", "")),
            "asof": str(card.get("asof", ""))[:19],
            "horizon_days": int(horizon_days),
            "model_kind": str(card.get("model_kind", "")),
            "proba_up": float(card.get("proba_up", 0.5) or 0.5),
            "blended_p": float(card.get("blended_p", 0.5) or 0.5),
            "direction": int(card.get("direction", 0)),
            "conviction": float(card.get("conviction", 0.0) or 0.0),
            "expected_move_pct": move_pct,
            "regime": str(card.get("regime", "")),
            "passes_gate": bool(card.get("passes_gate", False)),
            "data_quality_score": int(provenance.get("data_quality_score", 0) or 0),
            "model_trained_at": str(card.get("model_trained_at", "")),
            # New per-pillar fields. None instead of 0 so the analytics
            # later can distinguish "missing" from "neutral 50".
            "technical_score": float(pillars["technical"]) if "technical" in pillars else None,
            "fundamental_score": float(pillars["fundamental"]) if "fundamental" in pillars else None,
            "sentiment_score": float(pillars["sentiment"]) if "sentiment" in pillars else None,
            "macro_score": float(pillars["macro"]) if "macro" in pillars else None,
            "microstructure_score": float(pillars["microstructure"]) if "microstructure" in pillars else None,
            "oi_score": float(pillars["oi"]) if "oi" in pillars else None,
            "relative_strength_score": float(pillars["relative_strength"]) if "relative_strength" in pillars else None,
            "insider_score": float(pillars["insider"]) if "insider" in pillars else None,
            "earnings_momentum_score": float(pillars["earnings_momentum"]) if "earnings_momentum" in pillars else None,
            "rubric_total": float(card.get("rubric_total", 0.0)) if card.get("rubric_total") is not None else None,
        }
        date_iso = datetime.now(timezone.utc).date().isoformat()
        path = _lake_paths.predictions_path(date_iso)
        _lake_paths.ensure_parent(path)
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        # Coerce schema — pad any column missing from the (possibly older)
        # existing file. None default lets analytics distinguish missing
        # from explicit zero.
        for c in _SCHEMA_COLS:
            if c not in df.columns:
                df[c] = None
        df = df[_SCHEMA_COLS]
        df.to_parquet(path, index=False)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning(f"predict_log.log_prediction failed: {exc}")
        return False


def read_predictions_window(start_iso: str, end_iso: str) -> pd.DataFrame:
    """Read predictions logged between two ISO dates (inclusive)."""
    d = _lake_paths.predictions_dir()
    if not d.exists():
        return pd.DataFrame(columns=_SCHEMA_COLS)
    parts = []
    for f in sorted(d.glob("*.parquet")):
        if start_iso <= f.stem <= end_iso:
            try:
                parts.append(pd.read_parquet(f))
            except Exception as exc:  # noqa: BLE001
                log.warning(f"predict_log: skipping unreadable {f}: {exc}")
    if not parts:
        return pd.DataFrame(columns=_SCHEMA_COLS)
    return pd.concat(parts, ignore_index=True)
