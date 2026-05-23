"""Daily history archive for NSE bulk/block deals + SEBI insider disclosures.

The NSE `snapshot-capital-market-largedeal` and `corporates-pit`
endpoints both return single-day or short-window snapshots. To compute
trailing-window aggregates for `altdata_patch._insider_net_z_20d_live`
and `_block_deal_net_cr_live` we accumulate snapshots ourselves.

Storage layout:
    data/lake/bulk_deals_history.parquet
    data/lake/block_deals_history.parquet
    data/lake/sebi_insider_history.parquet

Each is a long-form table of event-level rows. Dedup key:
    bulk/block: (date, ticker, client_name, side, qty)
    sebi:       (date, ticker, person, transaction_type, qty)

Idempotent: re-running on the same day replaces those date+key rows,
never duplicates. Crash-safe: write failures logged, never raise.

Phase 25.1 — feeds altdata_patch.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from omni.utils.logger import log

LAKE = Path("data") / "lake"
BULK_PATH = LAKE / "bulk_deals_history.parquet"
BLOCK_PATH = LAKE / "block_deals_history.parquet"
SEBI_PATH = LAKE / "sebi_insider_history.parquet"

_BULK_KEY = ["date", "ticker", "client_name", "side", "qty"]
_BLOCK_KEY = ["date", "ticker", "client_name", "side", "qty"]
_SEBI_KEY = ["date", "ticker", "person", "transaction_type", "qty"]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception as exc:  # noqa: BLE001
        log.debug(f"deals_archive read {path.name} failed: {exc}")
        return pd.DataFrame()


def _write(df: pd.DataFrame, path: Path, key: list[str]) -> None:
    """Dedup + sort + persist. Best-effort."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Coerce key columns to string for stable dedup (qty/price can be float)
        if not df.empty:
            valid_key = [c for c in key if c in df.columns]
            if valid_key:
                df = df.drop_duplicates(subset=valid_key, keep="last")
            df = df.sort_values("date" if "date" in df.columns else valid_key[0])
        df.reset_index(drop=True).to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"deals_archive write {path.name} failed: {exc}")


def _append(live: pd.DataFrame, path: Path, key: list[str], label: str) -> pd.DataFrame:
    if live is None or live.empty:
        return _read(path)
    hist = _read(path)
    combined = pd.concat([hist, live], ignore_index=True) if not hist.empty else live.copy()
    _write(combined, path, key)
    log.info(f"deals_archive[{label}]: appended {len(live)} live rows; "
             f"history size now {len(combined.drop_duplicates(subset=[c for c in key if c in combined.columns]))}")
    return _read(path)


def snapshot_now() -> dict:
    """Pull live bulk-deals + block-deals + SEBI insider, append to archives.

    Returns a dict of size summaries. Best-effort: failures are logged
    and the affected archive is left alone.
    """
    out = {"bulk": 0, "block": 0, "sebi": 0}

    # 1. Bulk deals
    try:
        from omni.ingest.sources.nse_flows import bulk_deals
        bd = bulk_deals()
        if bd is not None and not bd.empty:
            bd = bd.copy()
            bd["date"] = pd.to_datetime(bd["date"], errors="coerce")
            hist = _append(bd, BULK_PATH, _BULK_KEY, "bulk")
            out["bulk"] = len(hist)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"deals_archive bulk snapshot failed: {exc}")

    # 2. Block deals
    try:
        from omni.ingest.sources.nse_flows import block_deals
        bk = block_deals()
        if bk is not None and not bk.empty:
            bk = bk.copy()
            bk["date"] = pd.to_datetime(bk["date"], errors="coerce")
            hist = _append(bk, BLOCK_PATH, _BLOCK_KEY, "block")
            out["block"] = len(hist)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"deals_archive block snapshot failed: {exc}")

    # 3. SEBI insider
    try:
        from omni.ingest.sources.sebi_insider import insider_disclosures
        si = insider_disclosures(days=30)
        if si is not None and not si.empty:
            si = si.copy()
            si["date"] = pd.to_datetime(si["date"], errors="coerce")
            hist = _append(si, SEBI_PATH, _SEBI_KEY, "sebi")
            out["sebi"] = len(hist)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"deals_archive sebi snapshot failed: {exc}")

    return out


# ----- Per-ticker rolling-window aggregators for altdata_patch -----------

def block_deal_net_cr_5d(ticker: str) -> float | None:
    """Net institutional-deal value (INR crores) over the last 5 sessions.

    Combines BOTH bulk-deal and block-deal archives:
      - Block deals: counterparty-matched (BUY+SELL of same trade) → net 0
        when paired (correct: matched trade has no directional signal).
      - Bulk deals: asymmetric flows → produce non-zero net when one side
        dominates (e.g. FII accumulating without matching seller-side
        bulk-deal of the same day).

    Sums signed (BUY=+1, SELL=-1) qty * avg_price / 1e7 across both
    archives. Returns None when neither archive has any rows for
    `ticker` in the last 5 sessions.
    """
    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=10)  # ~5 trading days ≈ 7 calendar; pad to 10
    nse_sym = str(ticker).upper().replace(".NS", "").replace(".BO", "")

    total = 0.0
    found_any = False
    for path in (BULK_PATH, BLOCK_PATH):
        hist = _read(path)
        if hist is None or hist.empty or "ticker" not in hist.columns:
            continue
        sub = hist[hist["ticker"].astype(str).str.upper() == nse_sym].copy()
        if sub.empty:
            continue
        sub = sub[sub["date"] >= cutoff]
        if sub.empty:
            continue
        side = sub.get("side", pd.Series([], dtype=str)).astype(str).str.upper()
        sign = side.map({"BUY": 1, "SELL": -1}).fillna(0).astype(float)
        qty = pd.to_numeric(sub.get("qty", 0), errors="coerce").fillna(0).astype(float)
        px = pd.to_numeric(sub.get("avg_price", 0), errors="coerce").fillna(0).astype(float)
        value_cr = (qty * px * sign) / 1e7
        if not np.isfinite(value_cr.sum()):
            continue
        total += float(value_cr.sum())
        found_any = True

    return total if found_any else None


def insider_net_z_20d(ticker: str) -> float | None:
    """20d z-score of per-ticker SEBI insider net-buy value.

    Reads from `sebi_insider_history.parquet`. Builds the daily signed
    value series for the ticker, computes its 20-session rolling z-score,
    returns the latest value. Needs >= 5 days of non-zero daily data
    for that ticker.
    """
    hist = _read(SEBI_PATH)
    if hist is None or hist.empty or "ticker" not in hist.columns:
        return None
    if "value_lakhs" not in hist.columns or "transaction_type" not in hist.columns:
        return None
    nse_sym = str(ticker).upper().replace(".NS", "").replace(".BO", "")
    sub = hist[hist["ticker"].astype(str).str.upper() == nse_sym].copy()
    if sub.empty:
        return None

    ttype = sub["transaction_type"].astype(str).str.lower()
    is_buy = ttype.str.contains("buy|acqui|purchase").astype(int)
    is_sell = ttype.str.contains("sell|dispos|sale").astype(int)
    sign = (is_buy - is_sell).astype(float)
    val = pd.to_numeric(sub["value_lakhs"], errors="coerce").fillna(0).astype(float)
    sub["signed_value"] = val * sign

    daily = sub.groupby(sub["date"].dt.normalize())["signed_value"].sum().sort_index()
    if len(daily) < 5:
        return None
    mu = daily.rolling(20, min_periods=5).mean().iloc[-1]
    sd = daily.rolling(20, min_periods=5).std().iloc[-1]
    if not np.isfinite(sd) or sd == 0:
        return None
    return float((daily.iloc[-1] - mu) / sd)
