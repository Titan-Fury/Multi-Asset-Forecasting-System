"""NSE Bhavcopy daily snapshot — delivery percentage is the smart-money proxy.

NSE publishes a daily 'sec_bhavdata_full_<DDMMYYYY>.csv' on the public
archives endpoint with one row per (SYMBOL, SERIES, DATE) including:
  - DELIV_QTY    : number of shares actually settled (vs intraday turnover)
  - DELIV_PER    : delivery percentage = DELIV_QTY / TTL_TRD_QNTY * 100

A high delivery % on rising volume is the cleanest free proxy we have
for institutional accumulation: anyone willing to take physical
delivery is NOT day-trading. Empirically the z-score of delivery_pct
vs its 60-day rolling baseline correlates with 5-20d forward returns.

Endpoint: https://archives.nseindia.com/products/content/sec_bhavdata_full_<DDMMYYYY>.csv

Free, no auth, served from CDN. Some weekends / holidays return 404 —
we handle that quietly. Cached at TTL.fundamentals (24h) per date.
"""
from __future__ import annotations

from datetime import date, datetime
from io import StringIO
from typing import Iterable

import pandas as pd
import requests

from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log

_BHAV_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/csv,*/*",
    "Referer": "https://www.nseindia.com/",
}
_TIMEOUT = 8.0


def _fmt_date(d: date | str | pd.Timestamp) -> str:
    ts = pd.Timestamp(d)
    return ts.strftime("%d%m%Y")


@cached(ttl=TTL.fundamentals, key="nse_bhavcopy:{ddmmyyyy}")
def _fetch_one(ddmmyyyy: str) -> pd.DataFrame:
    """Internal: fetch the bhavcopy CSV for a single date string."""
    url = _BHAV_URL.format(ddmmyyyy=ddmmyyyy)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code != 200 or not r.text.strip():
            log.debug(f"bhavcopy {ddmmyyyy}: status={r.status_code}")
            return pd.DataFrame()
        df = pd.read_csv(StringIO(r.text))
    except Exception as e:  # noqa: BLE001
        log.debug(f"bhavcopy {ddmmyyyy} fetch failed: {e}")
        return pd.DataFrame()

    # Column names sometimes have leading whitespace
    df.columns = [c.strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns or "DELIV_PER" not in df.columns:
        return pd.DataFrame()

    # Filter to EQ series only (drops bonds, ETFs in BE/BZ etc.)
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"].copy()

    # Coerce numerics
    for c in ("DELIV_QTY", "DELIV_PER", "TTL_TRD_QNTY", "TURNOVER_LACS",
              "CLOSE_PRICE"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Date column from filename (file's DATE field is sometimes blank)
    df["date"] = pd.Timestamp(datetime.strptime(ddmmyyyy, "%d%m%Y"))

    out = df.rename(columns={
        "SYMBOL": "ticker_nse",
        "DELIV_QTY": "deliv_qty",
        "DELIV_PER": "deliv_pct",
        "TTL_TRD_QNTY": "total_qty",
        "TURNOVER_LACS": "turnover_lacs",
        "CLOSE_PRICE": "close_price",
    })[["date", "ticker_nse", "deliv_qty", "deliv_pct",
        "total_qty", "turnover_lacs", "close_price"]]
    return out


def fetch_bhavcopy(d: date | str | pd.Timestamp) -> pd.DataFrame:
    """Public single-day fetch.

    Returns an empty DataFrame on 404 / weekend / holiday — the caller
    should treat missing days as a no-op, not a failure.
    """
    return _fetch_one(_fmt_date(d))


def fetch_range(start: date | str | pd.Timestamp,
                end: date | str | pd.Timestamp,
                *,
                tickers_nse: Iterable[str] | None = None) -> pd.DataFrame:
    """Fetch bhavcopy for every business day in [start, end] (inclusive).

    Skips missing days quietly. Optional `tickers_nse` filter (e.g.
    ['SBIN', 'RELIANCE']) lets the caller narrow the result without
    pulling 1,500 names per day.

    Returns concatenated DataFrame keyed by (date, ticker_nse).
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    days = pd.bdate_range(start, end)
    frames: list[pd.DataFrame] = []
    for d in days:
        df = fetch_bhavcopy(d)
        if df.empty:
            continue
        if tickers_nse is not None:
            wanted = {str(t).upper() for t in tickers_nse}
            df = df[df["ticker_nse"].astype(str).str.upper().isin(wanted)]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker_nse", "deliv_pct",
                                       "deliv_qty", "total_qty",
                                       "turnover_lacs", "close_price"])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker_nse", "date"]).reset_index(drop=True)
    return out
