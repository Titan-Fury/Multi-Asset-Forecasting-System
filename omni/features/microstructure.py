"""S1-lite microstructure features (free-feed proxy for VPIN / OFI / Kyle's lambda).

We don't have paid Level-2 / tick data, but we can recreate ~70% of the
predictive signal on liquid Indian names by combining feeds we already
ingest plus the new bhavcopy delivery %:

  delivery_pct_z_60d   z-score of NSE bhavcopy DELIV_PER vs per-ticker 60d
  oi_change_pct_5d     5-day change in F&O total OI (when option chain is live)
  fii_dii_net_z_20d    z-score of country-level FII+DII net cash flows (20d)
  amihud_illiq_20d     log of Amihud illiquidity = |ret| / dollar_volume, 20d mean
  signed_vol_proxy     Lee-Ready tick-rule signed volume aggregated to daily
  microstr_score       robust z-sum of the five above (single dense signal)

All inputs are India-equities-only. For other asset classes the function
returns NaN columns so the universal feature schema stays stable.

Wiring:
    from omni.features.microstructure import compute_microstructure
    out = compute_microstructure(out, asset_class, ticker, prices,
                                  bhavcopy_df=..., flows_df=...,
                                  intraday_bars=...)

Each kwarg is optional — missing inputs degrade to NaN for the columns
they would have populated. The composite `microstr_score` averages
whatever IS available so a partial feed still helps the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MICROSTR_COLS = (
    "delivery_pct_z_60d",
    "oi_change_pct_5d",
    "fii_dii_net_z_20d",
    "amihud_illiq_20d",
    "signed_vol_proxy",
    "microstr_score",
)


def _empty_columns(out: pd.DataFrame) -> pd.DataFrame:
    for c in _MICROSTR_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out


def _delivery_z(prices: pd.DataFrame,
                bhav: pd.DataFrame | None,
                ticker_nse: str) -> pd.Series:
    if bhav is None or bhav.empty or "ticker_nse" not in bhav.columns:
        return pd.Series(np.nan, index=prices.index)
    b = bhav[bhav["ticker_nse"].astype(str).str.upper()
             == ticker_nse.upper()].copy()
    if b.empty:
        return pd.Series(np.nan, index=prices.index)
    b["date"] = pd.to_datetime(b["date"]).astype("datetime64[ns]")
    b = b.sort_values("date")
    b["mu60"] = b["deliv_pct"].rolling(60, min_periods=15).mean()
    b["sd60"] = b["deliv_pct"].rolling(60, min_periods=15).std().replace(0, np.nan)
    b["delivery_pct_z_60d"] = (b["deliv_pct"] - b["mu60"]) / b["sd60"]
    merged = prices[["date"]].merge(
        b[["date", "delivery_pct_z_60d"]], on="date", how="left"
    )
    return merged["delivery_pct_z_60d"]


def _fii_dii_z(prices: pd.DataFrame, flows: pd.DataFrame | None) -> pd.Series:
    if flows is None or flows.empty:
        return pd.Series(np.nan, index=prices.index)
    f = flows.copy()
    f["date"] = pd.to_datetime(f["date"]).astype("datetime64[ns]")
    if "fii_net_cr" not in f.columns:
        return pd.Series(np.nan, index=prices.index)
    f["net"] = f.get("fii_net_cr", 0).fillna(0) + f.get("dii_net_cr", 0).fillna(0)
    f = f.sort_values("date")
    f["mu20"] = f["net"].rolling(20, min_periods=5).mean()
    f["sd20"] = f["net"].rolling(20, min_periods=5).std().replace(0, np.nan)
    f["fii_dii_net_z_20d"] = (f["net"] - f["mu20"]) / f["sd20"]
    merged = prices[["date"]].merge(
        f[["date", "fii_dii_net_z_20d"]], on="date", how="left"
    )
    return merged["fii_dii_net_z_20d"]


def _amihud(prices: pd.DataFrame) -> pd.Series:
    """Amihud (2002) illiquidity = mean(|ret| / dollar_volume) over 20d.
    Higher = less liquid = bigger price impact per unit volume."""
    px = prices["adj_close"].astype(float)
    vol = prices["volume"].astype(float).replace(0, np.nan)
    ret = px.pct_change().abs()
    dollar_vol = (px * vol).replace(0, np.nan)
    raw = ret / dollar_vol
    rolled = raw.rolling(20, min_periods=5).mean()
    # log1p compresses the long tail so trees can split it cleanly
    return np.log1p(rolled * 1e9)  # scale so the typical value is O(1)


def _signed_volume_lee_ready(intraday: pd.DataFrame | None,
                              prices: pd.DataFrame) -> pd.Series:
    """Lee-Ready (1991) tick rule on 5-min bars aggregated to daily.

    For each intraday bar with close c_i, sign = +1 if c_i > c_{i-1},
    -1 if c_i < c_{i-1}, else previous sign. Daily signed volume is
    sum(sign * volume), then normalised by 20d rolling abs mean.
    """
    if intraday is None or intraday.empty or "close" not in intraday.columns:
        return pd.Series(np.nan, index=prices.index)
    bars = intraday.copy()
    if "ts" in bars.columns:
        bars["date"] = pd.to_datetime(bars["ts"]).dt.normalize()
    elif "date" not in bars.columns:
        return pd.Series(np.nan, index=prices.index)
    bars = bars.sort_values("ts" if "ts" in bars.columns else "date")
    diff = bars["close"].diff()
    sign = np.sign(diff).replace(0, np.nan).ffill().fillna(0)
    bars["signed_vol"] = sign * bars["volume"].astype(float)
    daily = bars.groupby("date")["signed_vol"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ns]")
    # Normalise by 20d rolling abs mean so values land in O(1)
    rolling_norm = daily["signed_vol"].abs().rolling(20, min_periods=3).mean()
    daily["signed_vol_proxy"] = daily["signed_vol"] / rolling_norm.replace(0, np.nan)
    merged = prices[["date"]].merge(
        daily[["date", "signed_vol_proxy"]], on="date", how="left"
    )
    return merged["signed_vol_proxy"]


def compute_microstructure(
    out: pd.DataFrame,
    asset_class: str,
    ticker: str,
    prices: pd.DataFrame,
    *,
    bhavcopy_df: pd.DataFrame | None = None,
    flows_df: pd.DataFrame | None = None,
    intraday_bars: pd.DataFrame | None = None,
    oi_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add microstructure columns to `out`. Idempotent; safe for non-India."""
    out = _empty_columns(out)
    if asset_class != "india":
        return out  # leave NaN — model treats these as missing
    if prices is None or prices.empty:
        return out
    p = prices.sort_values("date").reset_index(drop=True).copy()
    p["date"] = pd.to_datetime(p["date"]).astype("datetime64[ns]")

    # NSE ticker without .NS suffix
    nse_sym = str(ticker).upper().replace(".NS", "").replace(".BO", "")

    # -- delivery_pct_z_60d
    deliv_z = _delivery_z(p, bhavcopy_df, nse_sym)
    # -- fii_dii_net_z_20d (country-level, same value broadcast across rows)
    fii_z = _fii_dii_z(p, flows_df)
    # -- amihud illiq (always available from price/vol)
    amihud = _amihud(p)
    # -- signed volume proxy
    sv = _signed_volume_lee_ready(intraday_bars, p)
    # -- oi_change_pct_5d
    if oi_history is not None and not oi_history.empty and "date" in oi_history.columns:
        oi = oi_history[oi_history.get("ticker", ticker) == ticker].copy() \
            if "ticker" in oi_history.columns else oi_history.copy()
        oi["date"] = pd.to_datetime(oi["date"]).astype("datetime64[ns]")
        oi = oi.sort_values("date")
        if "oi_total_lakhs" in oi.columns:
            oi["oi_change_pct_5d"] = oi["oi_total_lakhs"].pct_change(5) * 100
            merged = p[["date"]].merge(
                oi[["date", "oi_change_pct_5d"]], on="date", how="left"
            )
            oi_change = merged["oi_change_pct_5d"]
        else:
            oi_change = pd.Series(np.nan, index=p.index)
    else:
        oi_change = pd.Series(np.nan, index=p.index)

    # Stitch back onto `out` by date (out may have extra rows e.g. weekends)
    micro = pd.DataFrame({
        "date": p["date"],
        "delivery_pct_z_60d": deliv_z.values,
        "oi_change_pct_5d": oi_change.values,
        "fii_dii_net_z_20d": fii_z.values,
        "amihud_illiq_20d": amihud.values,
        "signed_vol_proxy": sv.values,
    })
    # Phase 24.22 — composite was averaging only 2 of the 5 produced signals
    # (the docstring claims "robust z-sum of the five above"). Now we include
    # all five, but z-score the three non-z components first so the mean is
    # over comparable scales. clip(-5, 5) keeps outlier days from dominating.
    z_cols = ["delivery_pct_z_60d", "fii_dii_net_z_20d"]
    extra_cols = ["oi_change_pct_5d", "amihud_illiq_20d", "signed_vol_proxy"]
    z_part = micro[z_cols].clip(-5, 5)
    extras = micro[extra_cols].astype(float)
    # Cross-sectional z-score across the per-ticker time-series so they
    # share a unit (sigma) with the *_z_* columns. ddof=0 to keep zero-vol
    # series finite (returns NaN -> mean(skipna=True) handles it).
    extras_mean = extras.mean(axis=0, skipna=True)
    extras_std = extras.std(axis=0, skipna=True, ddof=0)
    extras_z = ((extras - extras_mean) / extras_std.replace(0, np.nan)).clip(-5, 5)
    composite = pd.concat([z_part, extras_z], axis=1)
    micro["microstr_score"] = composite.mean(axis=1, skipna=True)

    out = out.drop(columns=[c for c in _MICROSTR_COLS if c in out.columns])
    out = out.merge(micro, on="date", how="left")
    return out
