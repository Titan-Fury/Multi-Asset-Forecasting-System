"""NSE F&O open-interest snapshot — leading indicator of institutional positioning.

NSE publishes daily aggregate OI data per FNO underlying:
  - total OI in calls & puts
  - PCR (put-call ratio) — directional bias
  - max-pain strike
  - sector-wise OI buildup

Endpoint: /api/option-chain-equities?symbol=<NSE_SYMBOL>
This returns the live option chain. We aggregate it down to per-ticker features:
  - oi_pcr               put OI / call OI
  - max_pain_strike      strike that maximizes loss for option writers
  - max_pain_dist_pct    distance from spot to max-pain (signed)
  - oi_total_lakhs       total OI across both sides (in lakhs)
  - oi_change_pct        change vs previous close (when available)

Cached at TTL.news.
"""
from __future__ import annotations

import pandas as pd
import requests

from omni.ingest.sources.nse_flows import _nse_session, _HEADERS_API
from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log


@cached(ttl=TTL.news, key="nse_option_chain:{ticker}")
def option_chain(ticker: str) -> dict:
    """Fetch the live option chain for an NSE FNO symbol. Returns raw JSON
    or {} on any failure.
    """
    s = _nse_session()
    s.headers.update(_HEADERS_API)
    try:
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={ticker}"
        r = s.get(url, timeout=8)
        if r.status_code != 200:
            log.debug(f"NSE option chain {ticker}: status={r.status_code}")
            return {}
        return r.json()
    except Exception as e:  # noqa: BLE001
        log.warning(f"nse option chain fetch failed for {ticker}: {e}")
        return {}


def oi_features(ticker: str) -> dict:
    """Aggregate per-ticker OI features.

    Returns: {oi_pcr, max_pain_strike, max_pain_dist_pct, oi_total_lakhs,
              ce_oi_lakhs, pe_oi_lakhs, spot, has_option_chain}
    """
    out = {
        "oi_pcr": 0.0, "max_pain_strike": 0.0, "max_pain_dist_pct": 0.0,
        "oi_total_lakhs": 0.0, "ce_oi_lakhs": 0.0, "pe_oi_lakhs": 0.0,
        "spot": 0.0, "has_option_chain": False,
    }
    chain = option_chain(ticker)
    if not chain or "records" not in chain:
        return out

    records = chain.get("records", {})
    rows = records.get("data", [])
    if not rows:
        return out

    spot = float(records.get("underlyingValue", 0) or 0)
    out["spot"] = spot
    out["has_option_chain"] = True

    # Build per-strike call/put OI (only nearest expiry)
    expiry_dates = records.get("expiryDates", [])
    if not expiry_dates:
        return out
    nearest = expiry_dates[0]

    strikes: dict[float, dict] = {}
    for r in rows:
        if r.get("expiryDate") != nearest:
            continue
        strike = float(r.get("strikePrice", 0) or 0)
        ce = (r.get("CE") or {}).get("openInterest", 0) or 0
        pe = (r.get("PE") or {}).get("openInterest", 0) or 0
        strikes.setdefault(strike, {"ce": 0, "pe": 0})
        strikes[strike]["ce"] = float(ce)
        strikes[strike]["pe"] = float(pe)

    if not strikes:
        return out

    ce_total = sum(s["ce"] for s in strikes.values())
    pe_total = sum(s["pe"] for s in strikes.values())
    out["ce_oi_lakhs"] = round(ce_total / 1e5, 2)
    out["pe_oi_lakhs"] = round(pe_total / 1e5, 2)
    out["oi_total_lakhs"] = round((ce_total + pe_total) / 1e5, 2)
    out["oi_pcr"] = round(pe_total / ce_total, 3) if ce_total > 0 else 0.0

    # Max pain: strike that minimises total option-writer payoff
    if spot > 0:
        best_strike, best_pain = 0.0, float("inf")
        for k in sorted(strikes.keys()):
            # pain at strike k = sum(max(k - K, 0) * CE_oi) + sum(max(K - k, 0) * PE_oi)
            pain = 0.0
            for K, oi in strikes.items():
                if k > K:
                    pain += (k - K) * oi["ce"]  # call writers lose
                if K > k:
                    pain += (K - k) * oi["pe"]  # put writers lose
            if pain < best_pain:
                best_pain = pain
                best_strike = k
        out["max_pain_strike"] = best_strike
        out["max_pain_dist_pct"] = round((spot - best_strike) / spot * 100.0, 2) \
                                     if spot > 0 else 0.0
    return out
