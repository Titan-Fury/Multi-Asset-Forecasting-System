"""GDELT 2.0 DOC API client — global event news (no key).

Free, no auth. Indexed every 15 minutes from 60k+ news outlets worldwide.
Best leading-indicator source for breaking news. We use the DOC search
API with a per-ticker query string.

Schema:
  published_at (UTC), source, title, description (= ""),
  url, ticker_hint, lang
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from omni.utils.logger import log

_COLS = ["published_at", "source", "title", "description", "url",
         "ticker_hint", "lang"]

# Phase 24.21 — GDELT enforces "one request per 5 seconds" globally and replies
# 429 when violated. We were generating 100+ rate-limit warnings per ingest run
# because the caller fires tickers back-to-back. Track last-call time at module
# scope (thread-safe via lock) and sleep just enough to honor the 5s gap before
# every request. This eliminates the 429 storm and the cascade of connection
# errors that GDELT throws after sustained abuse.
_RATE_LIMIT_SECONDS = 5.0
_last_call_ts: float = 0.0
_rate_lock = threading.Lock()


def _throttle() -> None:
    """Block until at least _RATE_LIMIT_SECONDS have elapsed since the last
    GDELT call. Safe to call from multiple threads — the lock serializes.
    """
    global _last_call_ts
    with _rate_lock:
        now = time.monotonic()
        wait = _RATE_LIMIT_SECONDS - (now - _last_call_ts)
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.monotonic()


def news(ticker: str, display_name: str | None = None,
         *, lookback_days: int = 3, max_records: int = 75) -> pd.DataFrame:
    """Search GDELT for articles mentioning the ticker or company name."""
    q_terms = [ticker]
    if display_name and display_name.lower() != ticker.lower():
        q_terms.append(f'"{display_name}"')
    q = " OR ".join(q_terms)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    params = {
        "query": f"{q} sourcelang:eng",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": min(max_records, 250),
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
    }
    _throttle()
    try:
        # GDELT is picky about user-agent. Phase 24.18 — split timeout into
        # (connect, read) tuple. GDelt is intermittently unreachable globally
        # so a 5s connect timeout cuts wait when the API is down; 25s read
        # timeout still tolerates slow responses on big queries.
        r = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params=params, timeout=(5, 25),
            headers={"User-Agent": "Mozilla/5.0 Omni_Stocks/0.1"},
        )
        if r.status_code != 200:
            # Phase 24.20 — 429 / 5xx are transient and frequent; demote to
            # DEBUG so the daily logs aren't spammed with 100+ identical
            # warnings. We already throttle to ~5s between calls upstream.
            level = log.warning if r.status_code not in (429, 502, 503, 504) else log.debug
            level(f"gdelt {r.status_code}: {r.text[:200]}")
            return pd.DataFrame(columns=_COLS)
        try:
            arts = r.json().get("articles", [])
        except ValueError:
            # GDELT occasionally returns malformed JSON; treat as empty
            return pd.DataFrame(columns=_COLS)
    except (requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.RetryError) as e:
        # Transient network failures, slow reads, and urllib3 retry exhaustion
        # against GDelt are common and self-recover on the next ingest cycle.
        # Phase 24.21 — added RetryError because urllib3 wraps repeated
        # ConnectTimeouts in a MaxRetryError surfaced as RetryError, which was
        # previously falling through to the generic except below and emitting
        # a misleading "fetch failed" warning instead of the quieter debug.
        log.debug(f"gdelt unreachable for {ticker}: {e}")
        return pd.DataFrame(columns=_COLS)
    except Exception as e:
        log.warning(f"gdelt fetch failed for {ticker}: {e}")
        return pd.DataFrame(columns=_COLS)

    rows = []
    for a in arts:
        rows.append({
            "published_at": a.get("seendate"),
            "source": a.get("domain", "gdelt"),
            "title": a.get("title", ""),
            "description": "",
            "url": a.get("url", ""),
            "ticker_hint": ticker,
            "lang": "en",
        })
    df = pd.DataFrame(rows, columns=_COLS)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce",
                                             format="%Y%m%dT%H%M%SZ", utc=True)
    return df
