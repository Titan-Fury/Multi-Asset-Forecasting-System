"""Point-in-time fundamentals snapshot tier.

Live `_fundamentals_combined` returns whatever yfinance/FMP say *right
now*. That breaks replay reproducibility — re-running the model on a
past date returns today's fundamentals (the May 8 -> May 15 CIANAGRO
replay showed earnings_momentum jump 50 -> 82 between two same-day
runs because yfinance returned a fresher quarterly).

This module persists each successful live fundamentals fetch, dated by
when we received it. Replay-mode callers ask for the snapshot whose
`snapshot_date <= asof` — that's a leakage-safe approximation of what
the model would have seen on `asof`.

Layout:
    data/lake/fundamentals_pit/<asset_class>/<TICKER>.parquet
    columns: snapshot_date + (every fundamentals field returned)

Idempotency: snapshots for the same (ticker, snapshot_date) are
overwritten so re-running the orchestrator on the same trading day
doesn't accumulate duplicates.

API:
    snapshot_save(asset_class, ticker, df_fundamentals)  # adds today's row
    load_pit(asset_class, ticker, asof)                    # latest <= asof
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from omni.utils.logger import log
from omni.utils.paths import local_data_root


def _path(asset_class: str, ticker: str) -> Path:
    safe = str(ticker).replace("/", "_").replace("\\", "_").upper()
    return (local_data_root() / "lake" / "fundamentals_pit"
            / asset_class.lower() / f"{safe}.parquet")


def snapshot_save(asset_class: str, ticker: str,
                   df: pd.DataFrame, *,
                   snapshot_date: pd.Timestamp | None = None) -> Path | None:
    """Append today's fundamentals row to the per-ticker PIT parquet.

    Returns the parquet path, or None if `df` is empty / write failed.
    Idempotent: if the snapshot for `snapshot_date` exists it is replaced.
    """
    if df is None or df.empty:
        return None
    sd = pd.to_datetime(snapshot_date or pd.Timestamp.now()).normalize()
    p = _path(asset_class, ticker)
    p.parent.mkdir(parents=True, exist_ok=True)

    new = df.copy()
    new["snapshot_date"] = sd

    if p.exists():
        try:
            old = pd.read_parquet(p)
            # Drop any existing row with the same date; keep newest write
            old = old[pd.to_datetime(old["snapshot_date"]).dt.normalize() != sd]
            combined = pd.concat([old, new], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"fundamentals_pit: read failed {p}: {exc}; overwriting")
            combined = new
    else:
        combined = new

    combined = combined.sort_values("snapshot_date").reset_index(drop=True)
    try:
        combined.to_parquet(p, index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"fundamentals_pit: write failed {p}: {exc}")
        return None
    return p


def load_pit(asset_class: str, ticker: str,
              asof: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """Return the latest fundamentals snapshot whose `snapshot_date <= asof`.

    When `asof` is None or no snapshot exists at/before `asof`, returns
    an empty DataFrame.
    """
    p = _path(asset_class, ticker)
    if not p.exists():
        return pd.DataFrame()
    try:
        hist = pd.read_parquet(p)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"fundamentals_pit: read failed {p}: {exc}")
        return pd.DataFrame()
    if hist.empty or "snapshot_date" not in hist.columns:
        return pd.DataFrame()
    hist["snapshot_date"] = pd.to_datetime(hist["snapshot_date"])
    if asof is None:
        return hist.sort_values("snapshot_date").tail(1).reset_index(drop=True)
    cut = pd.to_datetime(asof).normalize()
    sub = hist[hist["snapshot_date"] <= cut]
    if sub.empty:
        return pd.DataFrame()
    return sub.sort_values("snapshot_date").tail(1).reset_index(drop=True)


def list_snapshots(asset_class: str, ticker: str) -> pd.DataFrame:
    """Return all snapshots for a ticker (date + columns). Empty when none."""
    p = _path(asset_class, ticker)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p).sort_values("snapshot_date").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"fundamentals_pit: read failed {p}: {exc}")
        return pd.DataFrame()
