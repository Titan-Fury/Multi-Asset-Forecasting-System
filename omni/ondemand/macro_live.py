"""Live macro feature builder for the on-demand path.

Mirrors omni.features.macro.compute() schema but pulls FRED+yfinance
on the fly, with no tier read. Cached 24h via diskcache.

Schema returned (one row per business day):
  date,
  vix_level, vix_chg_5d,
  dxy_level, dxy_chg_5d,
  us10y, us10y_chg_5d,
  yield_curve_2_10,
  oil_chg_20d,
  gold_chg_20d,
  cpi_yoy,
  unrate_z_24m
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from omni.ingest.sources import fred_live
from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log

_FRED_IDS = ["DGS10", "DGS2", "VIXCLS", "DCOILWTICO", "CPIAUCSL", "UNRATE"]
_YF_MACRO = {"vix": "^VIX", "dxy": "DX-Y.NYB", "us10y": "^TNX",
             "gold": "GC=F", "oil": "CL=F"}


def _fetch_yf_series(symbol: str, *, lookback_years: int = 10) -> pd.DataFrame:
    """Fetch a single yfinance macro series. Returns DataFrame[date, value]."""
    import yfinance as yf
    end = date.today()
    start = end - timedelta(days=int(lookback_years * 365.25))
    try:
        df = yf.download(symbol, start=start.isoformat(), end=end.isoformat(),
                         progress=False, auto_adjust=False, threads=False)
    except Exception as e:
        log.warning(f"yf macro fetch failed {symbol}: {e}")
        return pd.DataFrame(columns=["date", "value"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "value"])
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "date", "Close": "value"})
    return df[["date", "value"]].dropna()


@cached(ttl=TTL.macro, key="macro_live:v2:{lookback_years}")
def compute_live(*, lookback_years: int = 10) -> pd.DataFrame:
    """Build the wide macro feature frame from live sources.

    Falls back to whatever subset of sources are available. If FRED is missing
    (no key), yfinance covers vix/dxy/us10y/gold/oil but we lose CPI, UNRATE,
    DGS2 -> no yield_curve_2_10. The caller should not assume any column.
    """
    fred = fred_live.many(_FRED_IDS, lookback_years=lookback_years)
    yf_rows = []
    for name, sym in _YF_MACRO.items():
        s = _fetch_yf_series(sym, lookback_years=lookback_years)
        if s.empty:
            continue
        s = s.assign(series_id=name)[["date", "series_id", "value"]]
        yf_rows.append(s)
    yfm = pd.concat(yf_rows, ignore_index=True) if yf_rows else \
        pd.DataFrame(columns=["date", "series_id", "value"])
    raw = pd.concat([fred, yfm], ignore_index=True)
    if raw.empty:
        return pd.DataFrame(columns=["date"])
    raw["date"] = pd.to_datetime(raw["date"])

    wide = raw.pivot_table(index="date", columns="series_id",
                            values="value", aggfunc="last").sort_index()
    bdays = pd.bdate_range(wide.index.min(), wide.index.max())
    wide = wide.reindex(bdays).ffill()

    out = pd.DataFrame({"date": wide.index})

    def _col(names: list[str]) -> pd.Series | None:
        for n in names:
            if n in wide.columns:
                return wide[n]
        return None

    vix = _col(["VIXCLS", "vix"])
    dxy = _col(["dxy"])
    us10 = _col(["DGS10", "us10y"])
    us2 = _col(["DGS2"])
    oil = _col(["DCOILWTICO", "oil"])
    gold = _col(["gold"])
    cpi = _col(["CPIAUCSL"])
    unrate = _col(["UNRATE"])

    if vix is not None:
        out["vix_level"] = vix.values
        out["vix_chg_5d"] = vix.pct_change(5).values
    if dxy is not None:
        out["dxy_level"] = dxy.values
        out["dxy_chg_5d"] = dxy.pct_change(5).values
    if us10 is not None:
        out["us10y"] = us10.values
        out["us10y_chg_5d"] = us10.diff(5).values
    if us10 is not None and us2 is not None:
        out["yield_curve_2_10"] = (us10 - us2).values
    if oil is not None:
        out["oil_chg_20d"] = oil.pct_change(20).values
    if gold is not None:
        out["gold_chg_20d"] = gold.pct_change(20).values
    if cpi is not None:
        # CPI is monthly; pct_change(12) on the daily-ffilled series compares
        # today vs ~12 *business days* ago, not 12 months. Compute YoY on the
        # deduplicated monthly values and reindex back to bday with ffill.
        cpi_monthly = cpi.dropna().drop_duplicates()
        cpi_yoy_m = cpi_monthly.pct_change(12)
        out["cpi_yoy"] = cpi_yoy_m.reindex(wide.index, method="ffill").values
    if unrate is not None:
        # Same monthly-vs-daily mismatch as CPI. 24-month z-score must be
        # computed on the monthly series, not on a 24-bday rolling window.
        unrate_monthly = unrate.dropna().drop_duplicates()
        roll_m = unrate_monthly.rolling(24, min_periods=12)
        z_m = (unrate_monthly - roll_m.mean()) / roll_m.std(ddof=0)
        out["unrate_z_24m"] = z_m.reindex(wide.index, method="ffill").values

    return out
