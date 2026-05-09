"""Atomic artifact writers for the dashboard tier.

The dashboard reads from data/{predictions,signals,regime,backtest,features}/.
This module owns the write path. All writes are atomic via tmp-file + rename
so the dashboard never sees a half-written parquet.

Layout:
  data/predictions/latest.parquet     -- per-row OOF predictions
  data/predictions/<run_date>.parquet -- daily snapshot
  data/signals/latest.parquet         -- gated + ungated rows for today
  data/signals/<run_date>.parquet
  data/signals/history.parquet        -- append-only emitted-signal log
  data/regime/latest.parquet          -- date -> regime label
  data/backtest/summary.json          -- BacktestReport.summary
  data/backtest/fold_metrics.json
  data/features/latest.parquet        -- last-day features (small)

Persisting writes a manifest json so the dashboard can show a "last run at" stamp.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from omni.utils.logger import log


_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "data"


def _atomic_write_parquet(df: pd.DataFrame, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=str(dst.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        # pandas.to_parquet serializes df.attrs as JSON; non-JSON-safe values
        # (DataFrames stashed for cross-stage data) blow up. Strip before write.
        if df.attrs:
            df = df.copy()
            df.attrs = {}
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            try: tmp_path.unlink()
            except OSError: pass


def _atomic_write_json(payload, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(dst.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text(json.dumps(payload, default=str, indent=2))
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            try: tmp_path.unlink()
            except OSError: pass


@dataclass
class PersistResult:
    run_date: str
    paths: dict
    n_predictions: int
    n_signals: int
    n_emitted: int


def persist_run(*,
                predictions: pd.DataFrame | None = None,
                signals: pd.DataFrame | None = None,
                regime_labels: pd.DataFrame | None = None,
                backtest_summary: dict | None = None,
                fold_metrics: list[dict] | None = None,
                features_latest: pd.DataFrame | None = None,
                run_date: str | None = None) -> PersistResult:
    run_date = run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    paths: dict[str, str] = {}

    if predictions is not None and not predictions.empty:
        _atomic_write_parquet(predictions, _DATA / "predictions" / "latest.parquet")
        _atomic_write_parquet(predictions, _DATA / "predictions" / f"{run_date}.parquet")
        paths["predictions"] = str(_DATA / "predictions" / "latest.parquet")

    if signals is not None and not signals.empty:
        _atomic_write_parquet(signals, _DATA / "signals" / "latest.parquet")
        _atomic_write_parquet(signals, _DATA / "signals" / f"{run_date}.parquet")
        paths["signals"] = str(_DATA / "signals" / "latest.parquet")
        # append emitted to history
        emitted = signals[signals.get("passes_gate", False)] if "passes_gate" in signals.columns else signals.iloc[0:0]
        if not emitted.empty:
            hist_path = _DATA / "signals" / "history.parquet"
            if hist_path.exists():
                old = pd.read_parquet(hist_path)
                combined = pd.concat([old, emitted], ignore_index=True)
                # de-dupe on (date, ticker)
                combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
            else:
                combined = emitted
            _atomic_write_parquet(combined, hist_path)
            paths["history"] = str(hist_path)

    if regime_labels is not None and not regime_labels.empty:
        _atomic_write_parquet(regime_labels, _DATA / "regime" / "latest.parquet")
        paths["regime"] = str(_DATA / "regime" / "latest.parquet")

    if backtest_summary is not None:
        _atomic_write_json(backtest_summary, _DATA / "backtest" / "summary.json")
        paths["backtest_summary"] = str(_DATA / "backtest" / "summary.json")
    if fold_metrics is not None:
        _atomic_write_json(fold_metrics, _DATA / "backtest" / "fold_metrics.json")

    if features_latest is not None and not features_latest.empty:
        _atomic_write_parquet(features_latest, _DATA / "features" / "latest.parquet")
        paths["features"] = str(_DATA / "features" / "latest.parquet")

    manifest = {
        "run_date": run_date,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "paths": paths,
        "n_predictions": int(len(predictions)) if predictions is not None else 0,
        "n_signals": int(len(signals)) if signals is not None else 0,
        "n_emitted": int(signals["passes_gate"].sum()) if signals is not None and "passes_gate" in signals.columns else 0,
    }
    _atomic_write_json(manifest, _DATA / "manifest.json")
    log.info(f"persisted run {run_date}: {manifest['n_signals']} signals, "
             f"{manifest['n_emitted']} emitted")
    return PersistResult(
        run_date=run_date, paths=paths,
        n_predictions=manifest["n_predictions"],
        n_signals=manifest["n_signals"],
        n_emitted=manifest["n_emitted"],
    )
