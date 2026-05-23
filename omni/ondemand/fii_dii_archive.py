"""Daily FII/DII history archive.

`omni.ingest.sources.nse_flows.fii_dii_activity()` returns the latest
single trading day from NSE's `/api/fiidiiTradeReact` endpoint — not
historical. To compute a meaningful 20-session z-score (used by
`altdata_patch._fii_dii_z_live`) we need to accumulate snapshots
ourselves.

Storage layout:
    data/lake/fii_dii_history.parquet     # 1 row per trading date

Schema:
    [date, fii_buy_cr, fii_sell_cr, fii_net_cr,
     dii_buy_cr, dii_sell_cr, dii_net_cr]

Idempotent: re-running on the same day replaces today's row, never
duplicates. Crash-safe: write failures are logged but never raise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from omni.utils.logger import log

ARCHIVE_PATH = Path("data") / "lake" / "fii_dii_history.parquet"


def _read_history() -> pd.DataFrame:
    if not ARCHIVE_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(ARCHIVE_PATH)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"fii_dii_archive read failed: {exc}")
        return pd.DataFrame()


def _write_history(df: pd.DataFrame) -> None:
    """Persist sorted, deduped history."""
    try:
        ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = df.drop_duplicates(subset=["date"], keep="last") \
                .sort_values("date").reset_index(drop=True)
        df.to_parquet(ARCHIVE_PATH, index=False)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"fii_dii_archive write failed: {exc}")


def snapshot_now() -> pd.DataFrame:
    """Pull live FII/DII snapshot, append to history, return updated history.

    The NSE endpoint returns multiple recent days when warm; we union
    them all into the archive (replacing any existing rows for those
    dates). Best-effort: returns whatever's on disk if live fetch fails.
    """
    try:
        from omni.ingest.sources.nse_flows import fii_dii_activity
        live = fii_dii_activity()
    except Exception as exc:  # noqa: BLE001
        log.debug(f"fii_dii_archive: live fetch failed: {exc}")
        live = pd.DataFrame()

    hist = _read_history()
    if live is not None and not live.empty:
        live = live.copy()
        live["date"] = pd.to_datetime(live["date"], errors="coerce")
        live = live.dropna(subset=["date"])
        if hist.empty:
            combined = live
        else:
            combined = pd.concat([hist, live], ignore_index=True)
        _write_history(combined)
        log.info(
            f"fii_dii_archive: appended {len(live)} live rows; "
            f"history size now {len(combined.drop_duplicates(['date']))}"
        )
        return combined.drop_duplicates(["date"], keep="last") \
                       .sort_values("date").reset_index(drop=True)
    return hist


def get_history(*, lookback_days: int = 30) -> pd.DataFrame:
    """Return the trailing `lookback_days` of history (no live call)."""
    hist = _read_history()
    if hist.empty:
        return hist
    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=lookback_days)
    return hist[hist["date"] >= cutoff].reset_index(drop=True)


def fii_dii_net_z_20d_from_history() -> float | None:
    """Compute the latest 20d z-score of (fii_net_cr + dii_net_cr).

    Reads the archive on disk (does NOT make a live call). Returns None
    if there are <5 sessions of data. Used by `altdata_patch._fii_dii_z_live`.
    """
    hist = _read_history()
    if hist is None or hist.empty:
        return None
    if "fii_net_cr" not in hist.columns:
        return None
    d = hist.copy()
    d["net"] = d.get("fii_net_cr", 0).fillna(0) + d.get("dii_net_cr", 0).fillna(0)
    d = d.sort_values("date")
    if len(d) < 5:
        return None
    mu = d["net"].rolling(20, min_periods=5).mean().iloc[-1]
    sd = d["net"].rolling(20, min_periods=5).std().iloc[-1]
    if not np.isfinite(sd) or sd == 0:
        return None
    return float((d["net"].iloc[-1] - mu) / sd)
