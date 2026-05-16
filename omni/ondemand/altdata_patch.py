"""On-demand alt-data patcher for the live prediction row.

Training-time microstructure features are populated by
`omni/features/microstructure.py`, but the on-demand path (predict.py)
only patches OI fields. That leaves `insider_net_z_20d`,
`fii_dii_net_z_20d`, and `block_deal_net_value_cr` NaN at scoring time,
so the rubric's `insider` and microstructure pillars fall back to the
neutral 50.

This module computes those fields **on-demand from the live free
sources** and patches them onto a single feature row. All source pulls
are wrapped in try/except — any failure leaves the field NaN and the
rubric falls back to neutral 50, matching the existing degradation
contract.

Usage:
    from omni.ondemand.altdata_patch import patch_altdata
    feat_row = patch_altdata(feat_row, ticker, asset_class="india")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from omni.utils.logger import log


def _fii_dii_z_live() -> float | None:
    """Country-level FII+DII net flow, 20d z-score on the most recent value."""
    try:
        from omni.ingest.sources.nse_flows import fii_dii_activity
        df = fii_dii_activity()
        if df is None or df.empty or "fii_net_cr" not in df.columns:
            return None
        d = df.copy()
        d["net"] = d.get("fii_net_cr", 0).fillna(0) + d.get("dii_net_cr", 0).fillna(0)
        d = d.sort_values("date")
        if len(d) < 5:
            return None
        mu = d["net"].rolling(20, min_periods=5).mean().iloc[-1]
        sd = d["net"].rolling(20, min_periods=5).std().iloc[-1]
        if not np.isfinite(sd) or sd == 0:
            return None
        return float((d["net"].iloc[-1] - mu) / sd)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"fii_dii_z_live failed: {exc}")
        return None


def _insider_net_z_20d_live(ticker: str) -> float | None:
    """20d z-score of per-ticker insider net-buy value from SEBI PIT-2015.

    We approximate the z-score from a single window: the disclosures
    pulled cover ~30 days, so we compute the daily net-buy series, then
    z-score the latest day's value against the rolling 20d mean+std of
    that series. When the disclosure stream is sparse (single ticker
    typically gets a handful of filings/year), this returns None and the
    rubric falls back to neutral.
    """
    try:
        from omni.ingest.sources.sebi_insider import insider_disclosures
        df = insider_disclosures(days=30)
        if df is None or df.empty or "ticker" not in df.columns:
            return None
        nse_sym = str(ticker).upper().replace(".NS", "").replace(".BO", "")
        sub = df[df["ticker"].astype(str).str.upper() == nse_sym]
        if sub.empty or "value_lakhs" not in sub.columns or "transaction_type" not in sub.columns:
            return None
        ttype = sub["transaction_type"].astype(str).str.lower()
        signed_value = sub["value_lakhs"].fillna(0).astype(float) * (
            ttype.str.contains("buy|acqui|purchase").map({True: 1, False: 0})
            - ttype.str.contains("sell|dispos|sale").map({True: 1, False: 0})
        )
        # Aggregate to daily
        daily = pd.DataFrame({"date": sub["date"], "v": signed_value}) \
            .groupby("date")["v"].sum().sort_index()
        if len(daily) < 5:
            return None
        mu = daily.rolling(20, min_periods=5).mean().iloc[-1]
        sd = daily.rolling(20, min_periods=5).std().iloc[-1]
        if not np.isfinite(sd) or sd == 0:
            return None
        return float((daily.iloc[-1] - mu) / sd)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"insider_net_z_20d_live({ticker}) failed: {exc}")
        return None


def _block_deal_net_cr_live(ticker: str) -> float | None:
    """Net block-deal value (INR crores) over the last 5 sessions for `ticker`.

    NSE's snapshot endpoint only returns the most recent trading day,
    so this is best-effort: when we have today's deals we return the
    signed total; otherwise None. `value_cr = qty * avg_price / 1e7`.
    """
    try:
        from omni.ingest.sources.nse_flows import block_deals
        df = block_deals()
        if df is None or df.empty or "ticker" not in df.columns:
            return None
        nse_sym = str(ticker).upper().replace(".NS", "").replace(".BO", "")
        sub = df[df["ticker"].astype(str).str.upper() == nse_sym]
        if sub.empty or "qty" not in sub.columns or "avg_price" not in sub.columns:
            return None
        side = sub.get("side", pd.Series([], dtype=str)).astype(str).str.upper()
        sign = side.map({"BUY": 1, "SELL": -1}).fillna(0).astype(float)
        value_cr = (sub["qty"].astype(float) * sub["avg_price"].astype(float) * sign) / 1e7
        return float(value_cr.sum())
    except Exception as exc:  # noqa: BLE001
        log.debug(f"block_deal_net_cr_live({ticker}) failed: {exc}")
        return None


def patch_altdata(feat_row: pd.Series, ticker: str, *,
                   asset_class: str = "india") -> pd.Series:
    """Patch the live feature row with on-demand alt-data fields.

    Only runs for `asset_class == 'india'`. Each field is populated only
    when its current value is NaN — pre-computed values from the offline
    feature builder take priority. Missing fields stay NaN; the rubric
    pillar falls back to neutral 50.
    """
    if asset_class != "india":
        return feat_row

    # FII/DII (country-level, broadcasts across tickers)
    if pd.isna(feat_row.get("fii_dii_net_z_20d", np.nan)):
        v = _fii_dii_z_live()
        if v is not None:
            feat_row["fii_dii_net_z_20d"] = v

    # Insider net-buy z (per ticker)
    if pd.isna(feat_row.get("insider_net_z_20d", np.nan)):
        v = _insider_net_z_20d_live(ticker)
        if v is not None:
            feat_row["insider_net_z_20d"] = v

    # Block-deal net value, last 5 sessions (per ticker)
    if pd.isna(feat_row.get("block_deal_net_value_cr", np.nan)):
        v = _block_deal_net_cr_live(ticker)
        if v is not None:
            feat_row["block_deal_net_value_cr"] = v

    return feat_row
