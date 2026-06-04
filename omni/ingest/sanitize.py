"""Causal OHLCV bad-tick / outlier sanitizer (Phase 28.6).

Cleans raw daily OHLCV BEFORE feature + target computation so a single
corrupt bar (data-vendor typo, unadjusted corporate action, phantom
zero/negative price, duplicated date) doesn't poison every downstream
rolling feature (returns, vol, RSI, Bollinger, ...) AND the forward-return
labels — both of which read the same price frame.

Leak-free by construction:
  * structural + hard-integrity checks are per-row (no neighbours).
  * the extreme-move filter uses ONLY trailing statistics (rolling
    median / MAD shifted by one bar) plus a fixed per-asset-class
    absolute cap set ABOVE the exchange circuit limits, so legitimate
    limit-up/down days and earnings gaps survive while catastrophic
    vendor ticks are removed.

No future bar is ever consulted, so the live (last) inference bar is
sanitised by exactly the same rules as historical training bars.

Conservative trade-off: an isolated bad value at bar T produces a large
return INTO T (spike) and a large return OUT of T (revert). Both can trip
the cap, so the bad bar and its immediate neighbour may both be dropped.
Bad ticks are rare, so losing one adjacent bar is negligible versus the
~60-bar contamination a 10x spike inflicts on rolling windows.

Public API:
    sanitize_ohlcv(df, asset_class) -> (clean_df, stats)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_OHLC = ("open", "high", "low", "close", "adj_close")

# Absolute single-day |log-return| caps by asset class. Deliberately set
# ABOVE exchange circuit limits so real limit days / earnings gaps are
# preserved; only catastrophic ticks exceed them. India equities circuit
# at +/-20% (ln 1.20 ~ 0.18) -> 0.35 leaves wide headroom; crypto is
# genuinely volatile -> 0.70; fx is tight -> 0.15.
_ABS_LOGRET_CAP = {
    "india": 0.35,
    "us": 0.40,
    "crypto": 0.70,
    "fx": 0.15,
}
_DEFAULT_CAP = 0.50

_MAD_K = 8.0            # trailing robust-band multiplier (sigma-equivalents)
_MAD_WINDOW = 60        # trailing window for the robust band
_MAD_MINP = 20          # min trailing bars before the band is trusted


def _empty_stats() -> dict:
    return {
        "n_total": 0, "n_dup": 0, "n_dropped_integrity": 0,
        "n_dropped_extreme": 0, "n_clean": 0,
    }


def sanitize_ohlcv(df: pd.DataFrame, asset_class: str = "us"
                   ) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, stats). Idempotent and never raises on bad input.

    ``df`` must have a ``date`` column and at least ``adj_close``. Rows are
    sorted by date; duplicate dates collapse to the last; rows failing hard
    integrity (NaN / non-positive prices, impossible OHLC) are dropped; and
    bars whose trailing log-return breaks BOTH the trailing robust band AND
    the absolute asset-class cap are dropped as bad ticks.
    """
    stats = _empty_stats()
    if df is None or len(df) == 0:
        return df, stats
    if "date" not in df.columns or "adj_close" not in df.columns:
        return df, stats

    out = df.copy()
    stats["n_total"] = len(out)

    # --- structural: sort + dedup dates (keep last) ---
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    before = len(out)
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    stats["n_dup"] = before - len(out)

    # --- hard integrity: NaN / non-positive / impossible bars ---
    price_cols = [c for c in _OHLC if c in out.columns]
    bad = pd.Series(False, index=out.index)
    for c in price_cols:
        v = pd.to_numeric(out[c], errors="coerce")
        bad |= v.isna() | (v <= 0)
    if "high" in out.columns and "low" in out.columns:
        hi = pd.to_numeric(out["high"], errors="coerce")
        lo = pd.to_numeric(out["low"], errors="coerce")
        o = pd.to_numeric(out["open"], errors="coerce") if "open" in out.columns else hi
        cl = pd.to_numeric(out["close"], errors="coerce") if "close" in out.columns else hi
        # high must dominate open/close; low must be dominated by them.
        bad |= (hi < lo) | (hi < np.maximum(o, cl)) | (lo > np.minimum(o, cl))
    n0 = len(out)
    out = out[~bad.values].reset_index(drop=True)
    stats["n_dropped_integrity"] = n0 - len(out)

    # --- extreme single-bar move (causal: trailing band + absolute cap) ---
    if len(out) > _MAD_MINP:
        ac = pd.to_numeric(out["adj_close"], errors="coerce")
        lr = np.log(ac / ac.shift(1)).replace([np.inf, -np.inf], np.nan)
        # Trailing robust band built strictly from PAST returns (shift(1)).
        past = lr.shift(1)
        med = past.rolling(_MAD_WINDOW, min_periods=_MAD_MINP).median()
        mad = (past - med).abs().rolling(_MAD_WINDOW, min_periods=_MAD_MINP).median()
        band = (_MAD_K * 1.4826 * mad).clip(lower=0)          # MAD -> sigma
        cap = _ABS_LOGRET_CAP.get(asset_class, _DEFAULT_CAP)
        # Flag only when the move breaks BOTH the trailing band AND the cap,
        # so a benign drifting-vol regime never trips it.
        flagged = ((lr.abs() > band) & (lr.abs() > cap)).fillna(False)
        stats["n_dropped_extreme"] = int(flagged.sum())
        if flagged.any():
            out = out[~flagged.values].reset_index(drop=True)

    stats["n_clean"] = len(out)
    return out, stats
