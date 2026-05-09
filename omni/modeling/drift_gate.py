"""A6 — Conditional drift gating.

Wraps `omni.modeling.drift.DriftMonitor` with a per-horizon scope keyed
to live prediction error. Designed to be called from the daily backtest
(`scripts/daily_backtest.py`) where realised vs predicted is already
computed.

Public API:
    update_from_predictions(preds_df, realised_lookup, *, horizons=(1,5,21))
        -> dict[scope, dict]   # per-scope summary + drift bool

    needs_retrain(scope) -> bool

State written to `data/lake/monitoring/drift_<scope>.json`. The presence
of a `data/lake/monitoring/needs_retrain.json` file with any True value
is the cue for cron to schedule an emergency retrain.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from omni.modeling.drift import DriftMonitor

_MON_DIR = Path("data") / "lake" / "monitoring"
_FLAG = _MON_DIR / "needs_retrain.json"


def _brier(prob_up: float, realised_dir: int) -> float:
    """Per-row Brier component. realised_dir in {-1,0,+1}; UP -> 1, else 0."""
    y = 1.0 if realised_dir > 0 else 0.0
    return float((prob_up - y) ** 2)


def update_from_predictions(preds_df: pd.DataFrame,
                            realised_dir: pd.Series,
                            *,
                            horizons: Iterable[int] = (1, 5, 21),
                            delta: float = 0.002) -> dict:
    """Push every (pred, realised) pair through ADWIN.

    Args:
        preds_df: must contain columns
                  [horizon, prob_up, asof, ...].
        realised_dir: aligned series of realised direction (±1) per row.
        horizons: which horizons to maintain a separate monitor for.
        delta: ADWIN confidence (smaller = stricter).

    Returns:
        {scope: {"drifted": bool, "n_seen": int, "last_drift": str|None}}.
    """
    out: dict[str, dict] = {}
    if preds_df is None or preds_df.empty:
        return out
    df = preds_df.copy()
    df["__y__"] = pd.Series(realised_dir, index=df.index)
    df = df.dropna(subset=["__y__"])

    any_drift = False
    for h in horizons:
        scope = f"h{int(h)}"
        sub = df[df.get("horizon", h).astype(int) == int(h)] if "horizon" in df.columns else df
        if sub.empty:
            continue
        mon = DriftMonitor.load(scope, delta=delta)
        drifted = False
        for _, row in sub.iterrows():
            try:
                p = float(row.get("prob_up", row.get("blended_p", 0.5)))
                y = int(row["__y__"])
            except Exception:
                continue
            b = _brier(p, y)
            if mon.update(b):
                drifted = True
                mon.last_drift = datetime.now(tz=timezone.utc).isoformat()
        mon.save()
        if drifted:
            any_drift = True
        out[scope] = {
            "drifted": drifted,
            "n_seen": mon.n_seen,
            "last_drift": mon.last_drift,
        }

    # Consolidated retrain flag
    _MON_DIR.mkdir(parents=True, exist_ok=True)
    _FLAG.write_text(json.dumps({
        "any_drift": any_drift,
        "per_scope": out,
        "checked_at": datetime.now(tz=timezone.utc).isoformat(),
    }, indent=2))
    return out


def needs_retrain(scope: str | None = None) -> bool:
    """Read the consolidated retrain flag.

    If `scope` is given, returns True only if that scope drifted;
    otherwise returns True if any scope drifted.
    """
    if not _FLAG.exists():
        return False
    try:
        obj = json.loads(_FLAG.read_text())
    except Exception:
        return False
    if scope is None:
        return bool(obj.get("any_drift", False))
    return bool(obj.get("per_scope", {}).get(scope, {}).get("drifted", False))
