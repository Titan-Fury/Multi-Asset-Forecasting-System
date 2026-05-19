"""Macro features.

Loads the FRED series and yfinance macro tickers from raw tier, pivots to
wide form, computes simple lag/diff/z-score features, and forward-fills
non-trading days so it can be joined to per-ticker daily features.

Output schema (one row per business day):
  date,
  vix_level, vix_chg_5d,
  dxy_level, dxy_chg_5d,
  us10y, us10y_chg_5d,
  yield_curve_2_10,           # DGS10 - DGS2
  oil_chg_20d,
  gold_chg_20d,
  cpi_yoy,                    # if present
  unrate_z_24m
"""
from __future__ import annotations

import pandas as pd

from omni.storage import drive as _drive
from omni.utils.logger import log


def _load_fred(series_ids: list[str]) -> pd.DataFrame:
    rows = []
    for sid in series_ids:
        parts = ("raw", "macro_fred", "macro", f"{sid}.parquet")
        try:
            df = _drive.read_raw(*parts)
        except (RuntimeError, FileNotFoundError):
            log.debug(f"FRED {sid} not available")
            continue
        rows.append(df[["date", "series_id", "value"]])
    if not rows:
        return pd.DataFrame(columns=["date", "series_id", "value"])
    return pd.concat(rows, ignore_index=True)


def _load_yf_macro(names: list[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        parts = ("raw", "prices", "macro", f"{n}.parquet")
        try:
            df = _drive.read_raw(*parts)
        except (RuntimeError, FileNotFoundError):
            log.debug(f"macro yf {n} not available")
            continue
        rows.append(df[["date", "ticker", "close"]].rename(columns={"close": "value", "ticker": "series_id"}))
    if not rows:
        return pd.DataFrame(columns=["date", "series_id", "value"])
    return pd.concat(rows, ignore_index=True)


def compute(*, fred_ids: list[str] | None = None,
            yf_names: list[str] | None = None) -> pd.DataFrame:
    """Build the wide macro feature frame across all available business days."""
    fred_ids = fred_ids or ["DGS10", "DGS2", "VIXCLS", "DCOILWTICO",
                              "CPIAUCSL", "UNRATE", "DEXINUS"]
    yf_names = yf_names or ["vix", "dxy", "us10y", "gold", "oil"]

    fred = _load_fred(fred_ids)
    yfm = _load_yf_macro(yf_names)
    raw = pd.concat([fred, yfm], ignore_index=True)
    if raw.empty:
        return pd.DataFrame(columns=["date"])

    raw["date"] = pd.to_datetime(raw["date"])
    wide = raw.pivot_table(index="date", columns="series_id",
                           values="value", aggfunc="last").sort_index()

    # Build a daily business-day index covering the data range, ffill to cover holidays
    bdays = pd.bdate_range(wide.index.min(), wide.index.max())
    wide = wide.reindex(bdays).ffill()

    out = pd.DataFrame({"date": wide.index})

    def _col(name_options: list[str]) -> pd.Series | None:
        for n in name_options:
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
        # Phase 24.22 — CPI is a MONTHLY series; after the bday-reindex+ffill
        # above it repeats the same value for ~22 business days. Calling
        # `.pct_change(12)` on the ffilled daily series compares today's CPI
        # to the value 12 BUSINESS DAYS ago (~2.5 weeks), which almost always
        # produces 0 (same ffilled value). Compute YoY on the deduplicated
        # monthly series first (12 monthly observations = 1 year), then
        # forward-fill back onto the bday index.
        cpi_monthly = cpi.dropna().drop_duplicates()
        cpi_yoy_m = cpi_monthly.pct_change(12)
        out["cpi_yoy"] = cpi_yoy_m.reindex(wide.index).ffill().values
    if unrate is not None:
        # Phase 24.22 — same fix as CPI above. UNRATE is monthly; rolling(24)
        # on the daily-ffilled series captured ~24 business days (~1 month)
        # of distinct values, not 24 months. Compute on monthly series, ffill
        # the resulting z-score back onto the bday index.
        unrate_monthly = unrate.dropna().drop_duplicates()
        roll_m = unrate_monthly.rolling(24, min_periods=12)
        unrate_z_m = (unrate_monthly - roll_m.mean()) / roll_m.std(ddof=0)
        out["unrate_z_24m"] = unrate_z_m.reindex(wide.index).ffill().values

    # Phase 25.1 — cross-asset broadcast features. Z-scored over 20
    # sessions so trees can split on regime tails (vix_z > 2 = panic,
    # us10y bp shock = duration risk). Broadcast across every ticker
    # via the existing build.py macro merge — same value for all rows
    # of a given date, but trees pick up the interaction with other
    # per-ticker features.
    inr = _col(["DEXINUS"])
    if vix is not None:
        roll = vix.rolling(20, min_periods=10)
        out["vix_z_20d"] = ((vix - roll.mean()) / roll.std(ddof=0)).values
    if us10 is not None:
        # Convert pct-points-per-day diff to basis points (×100)
        out["us10y_chg_5d_bp"] = (us10.diff(5) * 100).values
        # Z-score for shape consistency with vix_z_20d
        roll = us10.diff(5).rolling(20, min_periods=10)
        out["us10y_chg_5d_z"] = ((us10.diff(5) - roll.mean()) / roll.std(ddof=0)).values
    if inr is not None:
        out["usd_inr_level"] = inr.values
        out["usd_inr_chg_5d_pct"] = (inr.pct_change(5) * 100).values
        roll = inr.rolling(20, min_periods=10)
        out["usd_inr_z_20d"] = ((inr - roll.mean()) / roll.std(ddof=0)).values

    return out
