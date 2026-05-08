"""NewsAPI live client (no tier writes).

Free tier: 100 requests/day. Returns recent headlines mentioning a query.
Schema:
  published_at (UTC), source, title, description, url, ticker_hint, lang
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from omni.utils.logger import log

_COLS = ["published_at", "source", "title", "description", "url",
         "ticker_hint", "lang"]


def _key() -> str | None:
    k = os.getenv("NEWSAPI_KEY", "").strip()
    return k or None


def news(ticker: str, display_name: str | None = None,
         *, lookback_days: int = 3, page_size: int = 50) -> pd.DataFrame:
    key = _key()
    if not key:
        return pd.DataFrame(columns=_COLS)

    q = display_name or ticker
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": q, "from": since, "language": "en",
                     "sortBy": "publishedAt", "pageSize": page_size,
                     "apiKey": key}, timeout=30,
        )
        if r.status_code != 200:
            log.warning(f"newsapi {r.status_code}: {r.text[:200]}")
            return pd.DataFrame(columns=_COLS)
        arts = r.json().get("articles", [])
    except Exception as e:
        log.warning(f"newsapi fetch failed for {ticker}: {e}")
        return pd.DataFrame(columns=_COLS)

    rows = [{
        "published_at": a.get("publishedAt"),
        "source": (a.get("source") or {}).get("name", "newsapi"),
        "title": a.get("title", ""),
        "description": a.get("description", ""),
        "url": a.get("url", ""),
        "ticker_hint": ticker,
        "lang": "en",
    } for a in arts if a.get("url")]
    df = pd.DataFrame(rows, columns=_COLS)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    return df
