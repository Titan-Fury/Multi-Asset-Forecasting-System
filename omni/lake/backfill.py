"""Backfill orchestrator for the data lake.

Two modes:
- ``mode="full"``: fresh pull of N years per ticker (slow, 1-time).
- ``mode="incremental"``: pull only since-last-date already on disk (fast, daily).

The orchestrator is idempotent — re-running it does no harm. Each ticker
checkpoints to its own parquet. A crashed run can resume by simply re-running
in incremental mode.

Public entry points:
  backfill_prices(tickers, *, asset_class, lookback_years=15, mode="incremental")
  backfill_fundamentals(tickers, *, asset_class, date_iso=None)
  backfill_universe(asset_classes, *, limit=None, lookback_years=15, mode)
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from omni.ingest.fundamentals import fetch_fundamentals
from omni.ingest.prices import fetch_prices
from omni.lake import io as _io
from omni.lake import paths as _paths
from omni.ondemand.universe import _CURATED  # type: ignore[attr-defined]
from omni.utils.logger import log


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _start_for(ticker: str, asset_class: str, lookback_years: int,
                 mode: str) -> str:
    """Decide the start date for a fetch based on lake state + mode."""
    if mode == "full":
        return (date.today() - timedelta(days=int(lookback_years * 365.25))).isoformat()
    last = _io.last_price_date(ticker, asset_class)
    if last is None:
        # No history yet → fall back to full lookback
        return (date.today() - timedelta(days=int(lookback_years * 365.25))).isoformat()
    return (last + pd.Timedelta(days=1)).date().isoformat()


# ---------------------------------------------------------------- prices ---
def backfill_prices_one(ticker: str, asset_class: str, *,
                          lookback_years: int = 15,
                          mode: str = "incremental") -> dict:
    """Fetch + persist prices for ONE ticker. Returns a stats dict."""
    start = _start_for(ticker, asset_class, lookback_years, mode)
    end = _today_iso()
    if start >= end:
        return {"ticker": ticker, "asset_class": asset_class,
                 "skipped": True, "reason": "up-to-date",
                 "rows_added": 0, "rows_total": len(_io.read_prices(ticker, asset_class))}
    try:
        df = fetch_prices(asset_class, ticker, start=start, end=end)
    except Exception as e:
        log.warning(f"backfill prices failed {asset_class}/{ticker}: {e}")
        return {"ticker": ticker, "asset_class": asset_class,
                 "skipped": True, "reason": f"fetch_error: {e}",
                 "rows_added": 0,
                 "rows_total": len(_io.read_prices(ticker, asset_class))}
    if df is None or df.empty:
        return {"ticker": ticker, "asset_class": asset_class,
                 "skipped": True, "reason": "empty",
                 "rows_added": 0,
                 "rows_total": len(_io.read_prices(ticker, asset_class))}
    rows_total = (_io.write_prices(df, ticker, asset_class) if mode == "full"
                  else _io.append_prices(df, ticker, asset_class))
    return {"ticker": ticker, "asset_class": asset_class, "skipped": False,
             "rows_added": len(df), "rows_total": rows_total,
             "first_new": str(pd.Timestamp(df["date"].min()).date()),
             "last_new":  str(pd.Timestamp(df["date"].max()).date())}


def backfill_prices(tickers: Iterable[tuple[str, str]], *,
                      lookback_years: int = 15,
                      mode: str = "incremental",
                      sleep_between: float = 0.0) -> list[dict]:
    """Backfill many (ticker, asset_class) pairs sequentially."""
    out: list[dict] = []
    pairs = list(tickers)
    for i, (ticker, ac) in enumerate(pairs, 1):
        stats = backfill_prices_one(ticker, ac, lookback_years=lookback_years,
                                       mode=mode)
        msg = (f"[{i}/{len(pairs)}] {ac}/{ticker}: "
                f"{'skip' if stats['skipped'] else '+' + str(stats['rows_added'])} rows"
                f"  total={stats['rows_total']}")
        log.info(msg)
        out.append(stats)
        if sleep_between > 0 and i < len(pairs):
            time.sleep(sleep_between)
    return out


# ---------------------------------------------------------- fundamentals ---
def backfill_fundamentals_snapshot(tickers: Iterable[tuple[str, str]], *,
                                       date_iso: str | None = None) -> dict:
    """Pull fundamentals for ``tickers`` and persist as one snapshot for ``date_iso``."""
    date_iso = date_iso or _today_iso()
    rows = []
    for ticker, ac in tickers:
        try:
            df = fetch_fundamentals(ac, ticker)
            if df is not None and not df.empty:
                rows.append(df.iloc[0].to_dict())
        except Exception as e:
            log.warning(f"backfill_fund {ac}/{ticker}: {e}")
    if not rows:
        return {"date": date_iso, "rows": 0}
    snap = pd.DataFrame(rows)
    n = _io.write_fundamentals_snapshot(snap, date_iso)
    log.info(f"fundamentals snapshot {date_iso}: {n} rows -> "
              f"{_paths.fundamentals_path(date_iso)}")
    return {"date": date_iso, "rows": n}


# -------------------------------------------------------------- universe ---
def universe_pairs(asset_classes: Iterable[str] | None = None,
                     limit: int | None = None) -> list[tuple[str, str]]:
    """Return the curated universe filtered by asset class with optional ``limit``."""
    pairs = [(t, ac) for (t, ac, _name) in _CURATED]
    if asset_classes:
        wanted = set(asset_classes)
        pairs = [p for p in pairs if p[1] in wanted]
    if limit:
        pairs = pairs[:limit]
    return pairs


def backfill_universe(asset_classes: Iterable[str] | None = None, *,
                        limit: int | None = None,
                        lookback_years: int = 15,
                        mode: str = "incremental",
                        do_fundamentals: bool = True,
                        sleep_between: float = 0.0) -> dict:
    """Run a full lake backfill across the curated universe."""
    pairs = universe_pairs(asset_classes, limit=limit)
    log.info(f"backfill_universe: {len(pairs)} pairs, mode={mode}, "
              f"lookback={lookback_years}y")
    price_stats = backfill_prices(pairs, lookback_years=lookback_years,
                                       mode=mode, sleep_between=sleep_between)
    fund_stats = (backfill_fundamentals_snapshot(pairs)
                    if do_fundamentals else {"rows": 0})
    summary = {
        "n_tickers": len(pairs),
        "mode": mode,
        "lookback_years": lookback_years,
        "price_total_rows": sum(s["rows_total"] for s in price_stats),
        "price_added_rows": sum(s["rows_added"] for s in price_stats),
        "tickers_skipped": sum(1 for s in price_stats if s["skipped"]),
        "fundamentals_rows": fund_stats.get("rows", 0),
        "asset_classes": sorted({ac for _, ac in pairs}),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _io.update_manifest({"last_backfill": summary})
    return summary
