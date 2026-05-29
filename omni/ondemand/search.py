"""Multi-source live ticker search.

Beyond the 150 curated names in `universe.py`, this module hits three
public search endpoints to discover ANY ticker the user types:

  - NSE  : https://www.nseindia.com/api/search/autocomplete?q=...
              (cookie-warmed, returns Indian listings with both
                NSE and BSE symbols + sector + ISIN)
  - Yahoo: https://query1.finance.yahoo.com/v1/finance/search?q=...
              (universal: US / NASDAQ / NYSE / LSE / TSE / NSE / BSE /
                crypto / FX / indices). No auth needed.
  - CoinGecko: https://api.coingecko.com/api/v3/search?query=...
              (deep crypto coverage, free).

Each result is normalised to:
    SearchResult(ticker, asset_class, display_name, exchange, source, score)

The router prioritises results so India-flavoured queries surface NSE
matches first, US-flavoured queries surface NASDAQ first, etc. Cached
30s in-process to avoid hammering APIs while the user types.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests

from omni.ingest.sources.nse_flows import _nse_session, _HEADERS_API
from omni.utils.logger import log


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36")
_YH_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}


@dataclass(frozen=True)
class SearchResult:
    ticker: str          # canonical symbol (no Yahoo suffix)
    asset_class: str     # india | us | crypto | fx | etf | index
    display_name: str
    exchange: str = ""   # NSE | BSE | NASDAQ | NYSE | LSE | crypto | etc.
    source: str = ""     # which API returned this (yahoo|nse|coingecko)
    score: float = 0.0   # 0..100 ranking score


# ---- in-process TTL cache (30s) so typing doesn't spam APIs ----
_CACHE: dict[str, tuple[float, list[SearchResult]]] = {}
_CACHE_TTL = 30.0
# Phase 28.4 — guard `_CACHE` mutations against concurrent search
# calls. The eviction-by-clear pattern in `_cache_set` is racy: a
# reader between `.clear()` and the new write sees an empty cache
# and triggers a duplicate fan-out to Yahoo / NSE / Coingecko.
import threading as _search_threading
_CACHE_LOCK = _search_threading.Lock()


def _cache_get(k: str) -> list[SearchResult] | None:
    hit = _CACHE.get(k)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL:
        return hit[1]
    return None


def _cache_set(k: str, v: list[SearchResult]) -> None:
    # Phase 28.4 — atomic clear + write under the lock so concurrent
    # readers don't observe a transiently empty cache.
    with _CACHE_LOCK:
        if len(_CACHE) > 256:
            _CACHE.clear()
        _CACHE[k] = (time.monotonic(), v)


# ---- Yahoo Finance Search ----
def _yahoo_search(q: str, limit: int = 8) -> list[SearchResult]:
    if not q or len(q) < 1:
        return []
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": q, "quotesCount": limit, "newsCount": 0},
            headers=_YH_HEADERS, timeout=4,
        )
        if r.status_code != 200:
            return []
        quotes = r.json().get("quotes", [])
    except Exception as e:  # noqa: BLE001
        log.debug(f"yahoo search failed: {e}")
        return []
    out: list[SearchResult] = []
    for i, q_ in enumerate(quotes):
        sym_full = str(q_.get("symbol", ""))
        if not sym_full:
            continue
        # Map exchange / suffix to our asset-class taxonomy
        if sym_full.endswith(".NS"):
            ac, exch, ticker = "india", "NSE", sym_full[:-3]
        elif sym_full.endswith(".BO"):
            ac, exch, ticker = "india", "BSE", sym_full[:-3]
        elif sym_full.endswith("-USD"):
            ac, exch, ticker = "crypto", "crypto", sym_full[:-4]
        elif sym_full.endswith("=X"):
            ac, exch, ticker = "fx", "fx", sym_full[:-2]
        elif sym_full.startswith("^"):
            ac, exch, ticker = "index", "index", sym_full
        else:
            ac, exch, ticker = "us", str(q_.get("exchange", "")), sym_full
        name = (q_.get("longname") or q_.get("shortname")
                or q_.get("symbol") or ticker)
        # Score: top result ~ 95, descending. Prefer EQUITY > ETF > INDEX
        kind = q_.get("quoteType", "EQUITY")
        score = max(20.0, 95.0 - i * 5.0)
        if kind != "EQUITY":
            score *= 0.85
        out.append(SearchResult(
            ticker=ticker, asset_class=ac, display_name=str(name),
            exchange=exch, source="yahoo", score=score,
        ))
    return out


# ---- NSE autocomplete (cookie-warmed) ----
def _nse_search(q: str, limit: int = 6) -> list[SearchResult]:
    if not q or len(q) < 2:
        return []
    try:
        s = _nse_session()
        s.headers.update(_HEADERS_API)
        r = s.get(
            "https://www.nseindia.com/api/search/autocomplete",
            params={"q": q}, timeout=4,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        symbols = data.get("symbols", []) if isinstance(data, dict) else []
    except Exception as e:  # noqa: BLE001
        log.debug(f"nse search failed: {e}")
        return []
    out: list[SearchResult] = []
    for i, item in enumerate(symbols[:limit]):
        sym = str(item.get("symbol_info", item.get("symbol", "")))
        if not sym:
            continue
        # NSE returns either a clean string in `symbol_suggest` or a JSON
        # blob (a list of {input, weight} dicts). Pull the nicest available.
        raw_name = item.get("symbol_suggest") or item.get("desc") or sym
        if isinstance(raw_name, list) and raw_name:
            # First entry usually has the long company name in 'input'
            name = next((x.get("input") for x in raw_name
                          if isinstance(x, dict) and x.get("input")), sym)
        else:
            name = str(raw_name)
        # symbol_info often *is* the company name when it's not the ticker
        ticker_fld = str(item.get("symbol", sym)).upper()
        # NSE search returns equities + ETFs; treat all as india
        out.append(SearchResult(
            ticker=ticker_fld, asset_class="india",
            display_name=str(name).title(), exchange="NSE",
            source="nse", score=max(30.0, 92.0 - i * 4.0),
        ))
    return out


# ---- CoinGecko search (crypto deep coverage) ----
def _coingecko_search(q: str, limit: int = 5) -> list[SearchResult]:
    if not q or len(q) < 2:
        return []
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/search",
            params={"query": q}, timeout=4,
        )
        if r.status_code != 200:
            return []
        coins = r.json().get("coins", [])
    except Exception as e:  # noqa: BLE001
        log.debug(f"coingecko search failed: {e}")
        return []
    out: list[SearchResult] = []
    for i, c in enumerate(coins[:limit]):
        sym = str(c.get("symbol", "")).upper()
        if not sym:
            continue
        name = str(c.get("name", sym))
        out.append(SearchResult(
            ticker=sym, asset_class="crypto",
            display_name=name, exchange="crypto",
            source="coingecko", score=max(25.0, 80.0 - i * 4.0),
        ))
    return out


# ---- Public router ----
def search_tickers(query: str, *, limit: int = 10,
                    india_first: bool = True) -> list[SearchResult]:
    """Multi-source live ticker search.

    Returns up to `limit` deduplicated results ranked by relevance.
    India-flavoured queries surface NSE first when india_first is True.
    """
    q = (query or "").strip()
    if not q:
        return []
    cache_key = f"{q.lower()}|{india_first}|{limit}"
    hit = _cache_get(cache_key)
    if hit is not None:
        return hit[:limit]

    looks_indian = bool(re.search(r"\.(ns|bo)$", q, re.IGNORECASE)) or india_first
    looks_crypto = bool(re.fullmatch(r"(btc|eth|sol|doge|ada|xrp|bnb|matic|"
                                       r"avax|dot|ltc|link|atom|near).*", q.lower()))

    results: list[SearchResult] = []

    # Run NSE (India-first) and Yahoo in parallel-ish (sequential is fine,
    # both have <= 4s timeouts).
    if looks_indian:
        results.extend(_nse_search(q))
    results.extend(_yahoo_search(q))
    if looks_crypto:
        results.extend(_coingecko_search(q))

    # Dedupe by (ticker.upper(), asset_class), keep highest-score
    best: dict[tuple[str, str], SearchResult] = {}
    for r in results:
        key = (r.ticker.upper(), r.asset_class)
        if key not in best or r.score > best[key].score:
            best[key] = r

    out = sorted(best.values(), key=lambda x: -x.score)
    _cache_set(cache_key, out)
    return out[:limit]
