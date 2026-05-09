"""Tiny daily archive for NSE option-chain OI snapshots.

`nse_oi.option_chain()` returns a fresh snapshot but doesn't keep
history, so `oi_change_pct_5d` is always 0-non-null. This module
appends one row per (ticker, today) to a per-ticker parquet, then
returns the trailing window for feature computation.

File layout:
    data/lake/oi_snapshots/<TICKER>.parquet

Append is idempotent: if today's row already exists for this ticker,
we replace it with the latest snapshot.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from omni.ingest.sources.nse_oi import oi_features
from omni.utils.logger import log

ARCHIVE_DIR = Path("data") / "lake" / "oi_snapshots"


def _path(ticker: str) -> Path:
    safe = str(ticker).replace("/", "_").replace("\\", "_").upper()
    return ARCHIVE_DIR / f"{safe}.parquet"


def snapshot_and_history(ticker: str, *, lookback_days: int = 30
                          ) -> pd.DataFrame:
    """Take today's OI snapshot, append to disk, return the trailing window.

    Returns DataFrame with columns:
      [date, ticker, oi_pcr, max_pain_strike, oi_total_lakhs,
       ce_oi_lakhs, pe_oi_lakhs, spot]
    Empty DataFrame on any failure (so callers can degrade gracefully).
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    nse_sym = str(ticker).upper().replace(".NS", "").replace(".BO", "")
    today = pd.Timestamp.now().normalize()

    # Read existing history (if any)
    p = _path(nse_sym)
    if p.exists():
        try:
            hist = pd.read_parquet(p)
            hist["date"] = pd.to_datetime(hist["date"]).astype("datetime64[ns]")
        except Exception:
            hist = pd.DataFrame()
    else:
        hist = pd.DataFrame()

    # Snapshot today (best-effort)
    try:
        feats = oi_features(nse_sym) or {}
    except Exception as e:  # noqa: BLE001
        log.debug(f"oi_archive: snapshot failed for {nse_sym}: {e}")
        feats = {}

    if feats:
        row = {
            "date": today,
            "ticker": ticker,
            "oi_pcr": feats.get("oi_pcr"),
            "max_pain_strike": feats.get("max_pain_strike"),
            "oi_total_lakhs": feats.get("oi_total_lakhs"),
            "ce_oi_lakhs": feats.get("ce_oi_lakhs"),
            "pe_oi_lakhs": feats.get("pe_oi_lakhs"),
            "spot": feats.get("spot"),
        }
        if not hist.empty and (hist["date"] == today).any():
            hist = hist[hist["date"] != today]
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        try:
            hist.sort_values("date").to_parquet(p, index=False)
        except Exception as e:  # noqa: BLE001
            log.debug(f"oi_archive: write failed: {e}")

    if hist.empty:
        return hist
    cutoff = today - pd.tseries.offsets.BDay(lookback_days)
    return hist[hist["date"] >= cutoff].sort_values("date").reset_index(drop=True)
