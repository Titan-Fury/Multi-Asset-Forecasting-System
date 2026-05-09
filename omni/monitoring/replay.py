"""Backtest replay helpers — read logged predictions, attach realised
forward returns from the price lake.

Used by:
  - scripts/daily_backtest.py (originally inlined; this is the
    refactored shared core).
  - omni/dashboard/app.py — the History-tab replay slider.
"""
from __future__ import annotations

import pandas as pd

from omni.lake import io as lake_io
from omni.monitoring.predict_log import read_predictions_window


def realised_return(ticker: str, asset_class: str,
                    asof: pd.Timestamp,
                    horizon_days: int) -> float | None:
    """Realised forward return for (ticker, asof, horizon).

    Returns None if no prices available or horizon hasn't elapsed yet.
    """
    px = lake_io.read_prices(ticker, asset_class)
    if px is None or px.empty:
        return None
    px = px.sort_values("date").reset_index(drop=True)
    px["date"] = pd.to_datetime(px["date"])
    asof_n = pd.Timestamp(asof).normalize()
    on_or_after = px[px["date"] >= asof_n]
    if len(on_or_after) <= horizon_days:
        return None
    p0 = float(on_or_after.iloc[0]["adj_close"])
    p1 = float(on_or_after.iloc[horizon_days]["adj_close"])
    if p0 <= 0:
        return None
    return p1 / p0 - 1.0


def replay_window(start_iso: str, end_iso: str,
                  *, attach_realised: bool = True) -> pd.DataFrame:
    """Load logged predictions in [start, end] and attach realised returns.

    Returns a DataFrame ready for table rendering with columns:
      asof, ticker, asset_class, horizon_days, model_kind,
      proba_up, blended_p, direction, conviction, expected_move_pct,
      regime, passes_gate, realised_ret, y_true, hit
    """
    preds = read_predictions_window(start_iso, end_iso)
    if preds.empty:
        return preds
    if not attach_realised:
        return preds
    realised = []
    y_true = []
    for _, p in preds.iterrows():
        r = realised_return(p["ticker"], p["asset_class"],
                            pd.to_datetime(p["asof"]),
                            int(p["horizon_days"]))
        realised.append(r)
        y_true.append(int(r > 0) if r is not None else None)
    out = preds.copy()
    out["realised_ret"] = realised
    out["y_true"] = y_true
    out["hit"] = [
        1 if (yt is not None and int(d) != 0 and int(d) == (1 if yt else -1))
        else (0 if (yt is not None and int(d) != 0) else None)
        for d, yt in zip(out["direction"], y_true)
    ]
    return out


def replay_summary(rdf: pd.DataFrame) -> dict:
    """One-line metrics for the table footer."""
    if rdf is None or rdf.empty:
        return {"n": 0, "hit_rate": None, "mean_realised_pct": None}
    realised_mask = rdf["realised_ret"].notna()
    if not realised_mask.any():
        return {"n": int(len(rdf)), "hit_rate": None,
                "mean_realised_pct": None}
    hits = rdf.loc[realised_mask & rdf["hit"].notna(), "hit"]
    return {
        "n": int(realised_mask.sum()),
        "n_active": int((rdf["direction"] != 0).sum()),
        "hit_rate": float(hits.mean()) if len(hits) else None,
        "mean_realised_pct": float(rdf.loc[realised_mask, "realised_ret"].mean() * 100),
    }
