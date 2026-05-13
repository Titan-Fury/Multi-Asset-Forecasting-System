"""SEC EDGAR fetcher (Form 4 insider trades + 8-K material events).

EDGAR is free and requires no API key — only a polite ``User-Agent`` header
identifying the requester per SEC fair-access rules.

Endpoints used:
- ``https://www.sec.gov/files/company_tickers.json`` (ticker -> CIK)
- ``https://data.sec.gov/submissions/CIK{cik:010d}.json`` (recent filings)

Both are cached aggressively. Form 4 = insider buys/sells (often leading);
8-K = material events (M&A, departures, guidance updates) filed within
4 business days of the event — typically hours before the press release.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from omni.ondemand.cache import cached, get_cache
from omni.utils.logger import log


_UA = os.environ.get("SEC_USER_AGENT", "Omni_Stocks research-noreply@example.com")
_HEADERS = {"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"}
_TIMEOUT = 20


def _get_cik_map() -> dict[str, str]:
    """Cache the full ticker->CIK map for 7 days (SEC updates daily)."""
    cache = get_cache()
    key = "edgar:cik_map_v1"
    cached_val = cache.get(key)
    if cached_val is not None:
        return cached_val
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        out: dict[str, str] = {}
        for _, rec in data.items():
            t = str(rec.get("ticker", "")).upper().strip()
            cik = str(rec.get("cik_str", "")).zfill(10)
            if t and cik:
                out[t] = cik
        cache.set(key, out, expire=7 * 86400)
        log.info(f"edgar: cik map cached ({len(out)} tickers)")
        return out
    except Exception as e:
        log.warning(f"edgar cik map failed: {e}")
        return {}


def _ticker_to_cik(ticker: str) -> str | None:
    """Resolve a US ticker (e.g. 'AAPL') to a 10-digit CIK string."""
    return _get_cik_map().get(ticker.upper().replace("-", "."))


@cached(ttl=1800, key="edgar:filings:{ticker}:{days}")
def recent_filings(ticker: str, *, days: int = 180,
                   forms: tuple[str, ...] = ("4", "8-K")) -> pd.DataFrame:
    """Recent EDGAR filings for ``ticker`` within the last ``days`` days.

    Returns a DataFrame with columns:
        ``filing_date``, ``form``, ``primary_doc``, ``description``,
        ``accession``, ``url``, ``ticker``, ``source``, ``title``,
        ``published_at``, ``sent_score``.

    The last four columns let the result be unioned with the news frame
    in ``lookup._news_combined`` without further reshaping.
    """
    cik = _ticker_to_cik(ticker)
    if not cik:
        return pd.DataFrame()
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"edgar submissions failed {ticker}: {e}")
        return pd.DataFrame()

    rec = data.get("filings", {}).get("recent", {})
    if not rec or not rec.get("form"):
        return pd.DataFrame()

    df = pd.DataFrame({
        "form": rec.get("form", []),
        "filing_date": rec.get("filingDate", []),
        "accession": rec.get("accessionNumber", []),
        "primary_doc": rec.get("primaryDocument", []),
        "description": rec.get("primaryDocDescription", []),
    })
    if df.empty:
        return df

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    df = df[df["filing_date"] >= cutoff]
    df = df[df["form"].isin(list(forms))]
    if df.empty:
        return df

    df["ticker"] = ticker.upper()
    df["source"] = "sec_edgar"
    df["title"] = df.apply(
        lambda r: (f"[{r['form']}] "
                    f"{r['description'] or 'EDGAR filing'} "
                    f"({pd.Timestamp(r['filing_date']).date()} · {r['accession']})"),
        axis=1)
    df["published_at"] = df["filing_date"]
    # Heuristic sentiment: 8-K is event-tagged elsewhere; Form 4 polarity needs
    # the actual XBRL transaction parse. Default to 0; refined by `insider_score`.
    df["sent_score"] = 0.0
    df["url"] = df.apply(
        lambda r: f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                  f"{str(r['accession']).replace('-', '')}/{r['primary_doc']}",
        axis=1,
    )
    df = df.sort_values("filing_date", ascending=False).reset_index(drop=True)
    return df[["filing_date", "form", "primary_doc", "description", "accession",
               "url", "ticker", "source", "title", "published_at", "sent_score"]]


@cached(ttl=3600, key="edgar:insider_score:{ticker}")
def insider_score(ticker: str) -> dict:
    """Simple insider-activity rollup over the last 30/90 days.

    Returns counts of Form 4 filings only. A real signed buy/sell aggregator
    would parse the per-filing XBRL — kept out for now to stay key-free
    and rate-limit friendly. Provided as a leading-indicator hint that
    something is happening, not as a signed delta.
    """
    df = recent_filings(ticker, days=90, forms=("4",))
    if df is None or df.empty:
        return {"form4_30d": 0, "form4_90d": 0, "last_form4": None}
    today = pd.Timestamp.utcnow().tz_localize(None)
    d30 = (df["filing_date"] >= today - pd.Timedelta(days=30)).sum()
    d90 = len(df)
    last = df["filing_date"].max()
    return {
        "form4_30d": int(d30),
        "form4_90d": int(d90),
        "last_form4": last.date().isoformat() if pd.notna(last) else None,
    }


def event_filings(ticker: str, *, days: int = 30) -> pd.DataFrame:
    """Just the 8-K material events in the last ``days``."""
    return recent_filings(ticker, days=days, forms=("8-K",))
