"""Screener.in fundamentals fallback for Indian equities (Phase 24.16).

yfinance is sparse for SME / mid-cap NSE stocks (CIANAGRO, OMAXAUTO, etc.) —
many of our names come back with `profit_margin: None`, `roe: None`,
`debt_to_equity: None`. Screener.in renders a structured ratio block on
each company page that's consistently filled even for SME listings.

Endpoint:
    https://www.screener.in/company/<TICKER>/  -> standalone view
    https://www.screener.in/company/<TICKER>/consolidated/  -> consolidated

We try consolidated first (richer for groups), fall back to standalone.

Page structure (as of late 2025):
    <ul id="top-ratios"> contains the headline ratios (PE, ROE, etc.).
    Each <li> has `<span class="name">` and `<span class="number">`.

We parse only the headline ratios — those map directly to our schema:
    pe_ratio, market_cap, book_value, dividend_yield, roe, roce,
    debt_to_equity (derived from D/E line if present)

Robustness: html.parser (stdlib), so no lxml dependency. Returns {} on
any failure — caller treats as "fallback also unavailable" and keeps the
yfinance result (or None) for those fields.
"""
from __future__ import annotations

import re

import requests

from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36")
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
_TIMEOUT = 8.0

# Map Screener.in ratio labels -> our canonical fundamental field names.
# Keys are lowercase + stripped (we normalise the raw label that way).
_LABEL_MAP: dict[str, str] = {
    "market cap":          "market_cap",          # in INR crores on screener
    "stock p/e":           "pe_ratio",
    "p/e":                 "pe_ratio",
    "book value":          "book_value",
    "dividend yield":      "dividend_yield",
    "roce":                "roce",
    "roe":                 "return_on_equity",
    "face value":          "face_value",
    "industry pe":         "industry_pe",
    "debt to equity":      "debt_to_equity",
    "promoter holding":    "promoter_holding_pct",
    "eps":                 "eps_ttm",
    "earnings yield":      "earnings_yield",
}


def _parse_number(s: str) -> float | None:
    """Parse '1,234.50%', 'Rs. 9,800 Cr.', '2.34' -> float (or None)."""
    if not s:
        return None
    # Strip Rs., currency, commas, percent, "Cr."
    cleaned = re.sub(r"[^\d.\-]", "", s)
    if cleaned in ("", "-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _scrape(url: str) -> dict[str, float]:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    except requests.exceptions.RequestException as e:
        log.debug(f"screener fetch failed {url}: {e}")
        return {}
    if r.status_code != 200:
        log.debug(f"screener status={r.status_code} for {url}")
        return {}

    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        log.warning("bs4 not installed; screener fallback disabled")
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    out: dict[str, float] = {}

    # Top ratios are inside <ul id="top-ratios">. Each <li> has a
    # `.name` span and a `.number` span.
    top = soup.find(id="top-ratios")
    if top is None:
        return out
    for li in top.find_all("li"):
        name_el = li.find("span", class_="name")
        num_el = li.find("span", class_="number")
        if not name_el or not num_el:
            continue
        label = name_el.get_text(strip=True).lower()
        # Strip trailing icons/help glyphs sometimes appended
        label = re.sub(r"[^a-z0-9 /%]+", "", label).strip()
        canonical = _LABEL_MAP.get(label)
        if canonical is None:
            continue
        val = _parse_number(num_el.get_text(strip=True))
        if val is None:
            continue
        out[canonical] = val
    return out


@cached(ttl=TTL.fundamentals if hasattr(TTL, "fundamentals") else 86400,
        key="screener_resolve:{ticker}")
def _resolve_url(ticker: str) -> str | None:
    """Resolve an NSE ticker to its Screener URL path.

    Screener uses internal IDs (BSE scrip codes) as URL slugs, not NSE
    tickers. Major NSE names (RELIANCE, TCS, ITC) often have a vanity
    redirect at `/company/<TICKER>/`; smaller / SME names don't and have
    to be found via the search API.

    Returns the URL path like '/company/519477/consolidated/' or None.
    """
    # Try the search API first — works for both major and SME names.
    try:
        r = requests.get(
            f"https://www.screener.in/api/company/search/?q={ticker}",
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            results = r.json()
            if results and isinstance(results, list):
                # Take the first result whose name starts with the ticker
                # OR if no exact prefix match, the very first result.
                tk_up = ticker.upper()
                for item in results:
                    name = str(item.get("name", "")).upper()
                    if name.startswith(tk_up) or tk_up in name.split()[0]:
                        return item.get("url")
                return results[0].get("url")
    except Exception as e:  # noqa: BLE001
        log.debug(f"screener resolve failed for {ticker}: {e}")
    # Final fallback: assume the ticker IS the slug
    return f"/company/{ticker.upper()}/"


@cached(ttl=TTL.fundamentals if hasattr(TTL, "fundamentals") else 86400,
        key="screener_in:{ticker}")
def fetch(ticker: str) -> dict[str, float]:
    """Return a dict of canonical fundamental fields for an NSE ticker.

    Tries consolidated view first (for groups with subsidiaries), falls
    back to standalone. Returns {} if both fail OR if the page contains
    no recognizable ratio block.
    """
    url_path = _resolve_url(ticker)
    if not url_path:
        return {}
    base = f"https://www.screener.in{url_path.rstrip('/')}"
    # Search-API URLs may already include /consolidated/; if not, append.
    if "consolidated" not in base:
        out = _scrape(f"{base}/consolidated/")
        if not out:
            out = _scrape(f"{base}/")
    else:
        out = _scrape(f"{base}/")
        if not out:
            # Strip /consolidated and try standalone
            standalone = base.rsplit("/consolidated", 1)[0] + "/"
            out = _scrape(standalone)

    # Normalise units. Screener returns market cap in crores; our schema
    # uses raw INR. PE / ROE / D/E are already in our units (no scaling).
    if "market_cap" in out:
        out["market_cap"] = out["market_cap"] * 1e7  # crores -> INR

    # ROE / ROCE / dividend yield are reported as percentages (e.g.
    # 18.5 = 18.5%). Convert to decimals to match yfinance.
    for k in ("return_on_equity", "roce", "dividend_yield",
              "earnings_yield", "promoter_holding_pct"):
        if k in out:
            out[k] = out[k] / 100.0

    return out
