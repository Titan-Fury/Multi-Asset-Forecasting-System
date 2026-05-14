"""GDELT 2.0 DOC API client — global event news (no key).

Free, no auth. Indexed every 15 minutes from 60k+ news outlets worldwide.
Best leading-indicator source for breaking news. We use the DOC search
API with a per-ticker query string.

Schema:
  published_at (UTC), source, title, description (= ""),
  url, ticker_hint, lang
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from omni.utils.logger import log

_COLS = ["published_at", "source", "title", "description", "url",
         "ticker_hint", "lang"]


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
            log.warning(f"gdelt {r.status_code}: {r.text[:200]}")
            return pd.DataFrame(columns=_COLS)
        try:
            arts = r.json().get("articles", [])
        except ValueError:
            # GDELT occasionally returns malformed JSON; treat as empty
            return pd.DataFrame(columns=_COLS)
    except (requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError) as e:
        # Transient network failures and slow reads against GDelt are common
        # and self-recover on the next ingest cycle. Log at DEBUG so the
        # daily logs stay quiet.
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
