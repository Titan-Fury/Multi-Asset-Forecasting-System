"""Intraday microstructure features.

Pure functions over a session-bars DataFrame (schema from
omni.ingest.intraday). Returns dicts of float features; safe-default zeros
when bars are missing or thin.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

_KEYS = (
    "intraday_return_pct",
    "gap_open_pct",
    "vwap_dev_bps",
    "opening_range_breakout",
    "intraday_realized_vol",
    "last_hour_return_pct",
    "volume_z_today",
    "range_compression",
    "intraday_bar_count",
)


def _zeros() -> dict:
    return {k: 0.0 for k in _KEYS}


def compute_intraday_features(
    bars: pd.DataFrame,
    last_daily_close: float | None,
    *,
    open_range_minutes: int = 30,
) -> dict:
    """Compute microstructure features from intraday 5-minute bars.

    `bars` is the *full* multi-day intraday frame from fetch_intraday_bars.
    Volume-z compares today's cumulative volume vs the trailing 20-session
    median. Returns a dict; zero-valued on insufficient data.
    """
    out = _zeros()
    if bars is None or bars.empty:
        return out

    bars = bars.sort_values("ts").reset_index(drop=True)
    bars["day"] = bars["ts"].dt.normalize()
    days = bars["day"].unique()
    if len(days) == 0:
        return out

    today = days[-1]
    today_bars = bars[bars["day"] == today].reset_index(drop=True)
    if today_bars.empty:
        return out

    out["intraday_bar_count"] = float(len(today_bars))

    open_px = float(today_bars["open"].iloc[0])
    last_px = float(today_bars["close"].iloc[-1])

    # 1) intraday return: open -> latest
    if open_px > 0:
        out["intraday_return_pct"] = (last_px / open_px - 1.0) * 100.0

    # 2) gap: yesterday close -> today open
    if last_daily_close and last_daily_close > 0:
        out["gap_open_pct"] = (open_px / last_daily_close - 1.0) * 100.0

    # 3) VWAP deviation in bps
    vwap = float(today_bars["vwap"].iloc[-1])
    if vwap > 0:
        out["vwap_dev_bps"] = (last_px / vwap - 1.0) * 1e4

    # 4) opening-range breakout
    if len(today_bars) >= 2:
        # take roughly first `open_range_minutes` of bars; bar period inferred from spacing
        if len(today_bars) >= 3:
            dt = (today_bars["ts"].iloc[1] - today_bars["ts"].iloc[0]).total_seconds() / 60.0
            n_or = max(1, int(round(open_range_minutes / max(dt, 1.0))))
        else:
            n_or = 1
        or_block = today_bars.iloc[:n_or]
        or_high = float(or_block["high"].max())
        or_low = float(or_block["low"].min())
        if last_px > or_high:
            out["opening_range_breakout"] = 1.0
        elif last_px < or_low:
            out["opening_range_breakout"] = -1.0

    # 5) intraday realized vol: std of bar log-returns, scaled to a session
    if len(today_bars) >= 5:
        rets = np.log(today_bars["close"]).diff().dropna()
        if len(rets) >= 3 and rets.std() > 0:
            # ~78 5-min bars in a US session; we give a unitless intra-session scale
            n_bars_full_session = 78.0
            out["intraday_realized_vol"] = float(rets.std() * math.sqrt(n_bars_full_session) * 100.0)

    # 6) last-hour return
    if len(today_bars) >= 2:
        cutoff = today_bars["ts"].iloc[-1] - pd.Timedelta(hours=1)
        last_hr = today_bars[today_bars["ts"] >= cutoff]
        if len(last_hr) >= 2:
            a = float(last_hr["close"].iloc[0])
            b = float(last_hr["close"].iloc[-1])
            if a > 0:
                out["last_hour_return_pct"] = (b / a - 1.0) * 100.0

    # 7) today's volume vs trailing 20-session median
    today_vol = float(today_bars["volume"].sum())
    prior = bars[bars["day"] != today]
    if not prior.empty:
        per_day = prior.groupby("day")["volume"].sum()
        if len(per_day) >= 3:
            med = float(per_day.median())
            mad = float((per_day - med).abs().median()) or 1.0
            out["volume_z_today"] = (today_vol - med) / (1.4826 * mad + 1e-9)

    # 8) range compression: today H-L vs prior-day H-L median.
    # Use two simple grouped aggregations + subtract — same result as
    # `groupby.apply(lambda g: g.high.max() - g.low.min())` but ~10x faster
    # and avoids the pandas 2.2 DeprecationWarning about apply including
    # the grouping columns.
    today_range = float(today_bars["high"].max() - today_bars["low"].min())
    if not prior.empty:
        per_day_high = prior.groupby("day")["high"].max()
        per_day_low = prior.groupby("day")["low"].min()
        per_day_range = (per_day_high - per_day_low).dropna()
        if len(per_day_range) >= 3 and per_day_range.median() > 0:
            out["range_compression"] = today_range / float(per_day_range.median())

    # Final sanitisation
    for k, v in list(out.items()):
        if not np.isfinite(v):
            out[k] = 0.0
    return out
