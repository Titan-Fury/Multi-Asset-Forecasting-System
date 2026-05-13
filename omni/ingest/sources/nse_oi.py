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

Phase 24.16 hardening:
  - Dedicated warmup that visits the option-chain landing page (the
    earlier flows-warmup didn't set the right cookies for this endpoint).
  - Retry once with a fresh session on 401/403/429 (NSE rotates rate
    limits independently for /api/option-chain-equities).
  - Sanity check: if every numeric field comes back zero AND the symbol
    is FNO-listed, emit a degradation event so the silent-empty mode is
    no longer invisible.
"""
from __future__ import annotations

import time

import requests

from omni.ingest.sources.nse_flows import _nse_session, _HEADERS_API
from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log


# Symbols known to have F&O. If any of these comes back empty we know
# the endpoint is broken (vs the symbol just being non-FNO). Used to
# raise a degradation event without false positives on cash-only stocks.
_KNOWN_FNO_SAMPLE = frozenset((
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "ITC", "ICICIBANK",
    "AXISBANK", "BHARTIARTL", "LT", "KOTAKBANK", "HCLTECH",
))


def _option_chain_warmup() -> requests.Session:
    """Build a session with cookies primed specifically for the
    option-chain-equities endpoint. NSE's anti-scraping is per-endpoint;
    a session warmed for /market-data does NOT necessarily work here.
    """
    s = _nse_session()
    s.headers.update(_HEADERS_API)
    # Visit the human option-chain page first; this sets the cookies
    # the JSON endpoint requires (`bm_sv`, `_abck`, etc).
    try:
        s.get("https://www.nseindia.com/option-chain", timeout=8)
    except Exception as e:  # noqa: BLE001
        log.debug(f"nse_oi option-chain warmup soft-fail: {e}")
    return s


@cached(ttl=TTL.news, key="nse_option_chain:{ticker}")
def option_chain(ticker: str) -> dict:
    """Fetch the live option chain for an NSE FNO symbol. Returns raw JSON
    or {} on any failure.

    Retries once with a fresh session on rate-limit / auth errors. Surfaces
    persistent failures to the degradation monitor.
    """
    url = f"https://www.nseindia.com/api/option-chain-equities?symbol={ticker}"
    last_status: int | None = None
    last_err: str | None = None

    for attempt in (1, 2):
        s = _option_chain_warmup()
        try:
            r = s.get(url, timeout=10)
            last_status = r.status_code
            if r.status_code == 200:
                try:
                    j = r.json()
                except ValueError as ve:
                    last_err = f"non-JSON body: {ve}"
                    log.debug(f"NSE option chain {ticker}: {last_err}")
                    j = {}
                if j:
                    return j
            elif r.status_code in (401, 403, 429):
                # Rate-limit or cookie expired — warmup again on retry
                last_err = f"status={r.status_code}"
                log.debug(f"NSE option chain {ticker} attempt {attempt}: {last_err}")
                time.sleep(0.5)
                continue
            else:
                last_err = f"status={r.status_code}"
                log.debug(f"NSE option chain {ticker}: {last_err}")
                break
        except requests.exceptions.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            log.debug(f"NSE option chain {ticker} attempt {attempt}: {last_err}")
            time.sleep(0.5)
            continue
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            log.warning(f"nse option chain unexpected error for {ticker}: {e}")
            break

    # Both attempts failed — surface to ops if this is a known-FNO name
    if ticker.upper() in _KNOWN_FNO_SAMPLE:
        try:
            from omni.llm.rag.degradation import monitor
            monitor.record(
                component="nse_oi",
                severity="warn",
                event="empty_response_on_known_fno",
                details={
                    "ticker": ticker,
                    "last_status": last_status,
                    "last_error": (last_err or "")[:200],
                },
            )
        except Exception:  # noqa: BLE001
            pass
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

