"""Currents News API live client.

Free tier: 600 requests/day, 15-day max date range per request. The
15-day window is still ~15x what NewsAPI's free tier offers (24h), so
small-cap tickers on quiet news days find articles via Currents.

Sign up: https://currentsapi.services
Key in env as CURRENTS_API_KEY.

Schema mirrors the news-union contract used by lookup._news_combined:
    published_at (UTC datetime), source, title, description, url,
    ticker_hint, lang
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from omni.utils.logger import log

_COLS = ["published_at", "source", "title", "description", "url",
         "ticker_hint", "lang"]

_BASE = "https://api.currentsapi.services/v1/search"


def _key() -> str | None:
    k = os.getenv("CURRENTS_API_KEY", "").strip()
    return k or None


def news(ticker: str, display_name: str | None = None,
         *, lookback_days: int = 14, page_size: int = 50,
         language: str = "en") -> pd.DataFrame:
    """Fetch articles mentioning the ticker / display name.

    `lookback_days` defaults to 14 (kept under the free-tier 15-day cap
    by one day to avoid edge-case 400s). The query string prefers
    `display_name` because Currents is full-text-indexed and a name
    match is more selective than a ticker symbol that may collide with
    random words.
    """
    key = _key()
    if not key:
        return pd.DataFrame(columns=_COLS)

    q = display_name or ticker
    end = datetime.now(timezone.utc)
    # Hard-cap at 15 days (free-tier limit); allow caller to ask for less.
    lookback = max(1, min(int(lookback_days), 15))
    start = end - timedelta(days=lookback)

    params = {
        "keywords": q,
        "language": language,
        "start_date": start.strftime("%Y-%m-%dT%H:%M:%S.000+00:00"),
        "end_date": end.strftime("%Y-%m-%dT%H:%M:%S.000+00:00"),
        "page_size": page_size,
        "apiKey": key,
    }

    try:
        r = requests.get(_BASE, params=params, timeout=30)
        if r.status_code == 429:
            log.warning("currents 429 rate-limit; backing off")
            return pd.DataFrame(columns=_COLS)
        if r.status_code != 200:
            log.warning(f"currents {r.status_code}: {r.text[:200]}")
            return pd.DataFrame(columns=_COLS)
        payload = r.json() or {}
        if str(payload.get("status", "")).lower() != "ok":
            log.warning(f"currents non-ok status: {str(payload)[:200]}")
            return pd.DataFrame(columns=_COLS)
        arts = payload.get("news", [])
    except Exception as exc:  # noqa: BLE001
        log.warning(f"currents fetch failed for {ticker}: {exc}")
        return pd.DataFrame(columns=_COLS)

    rows = []
    for a in arts:
        url = a.get("url") or ""
        if not url:
            continue
        # Currents puts the publisher domain in `author` sometimes and a
        # human-readable site in the URL host otherwise. Prefer the
        # domain so provenance aggregation matches NewsAPI / GDELT's
        # convention.
        try:
            from urllib.parse import urlparse
            host = urlparse(url).netloc
            domain = host.lower().lstrip("www.") if host else "currents"
        except Exception:  # noqa: BLE001
            domain = "currents"
        rows.append({
            "published_at": a.get("published"),
            "source": domain or "currents",
            "title": a.get("title", "") or "",
            "description": a.get("description", "") or "",
            "url": url,
            "ticker_hint": ticker,
            "lang": (a.get("language") or language)[:2],
        })

    df = pd.DataFrame(rows, columns=_COLS)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"],
                                              errors="coerce", utc=True)
        # Drop rows where the date didn't parse — keeps downstream merges sane.
        df = df.dropna(subset=["published_at"]).reset_index(drop=True)
    return df
