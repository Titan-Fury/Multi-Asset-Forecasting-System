"""Marketaux client — stock-tagged news with built-in sentiment.

Free tier: 100 requests/day. Returns articles tagged with the queried
ticker plus a sentiment_score in [-1, 1] computed by Marketaux's NLP.

Schema:
  published_at (UTC), source, title, description, url, ticker_hint,
  lang, sent_score (float in [-1, 1] or 0.0 if not provided)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from omni.utils.logger import log

_COLS = ["published_at", "source", "title", "description", "url",
         "ticker_hint", "lang", "sent_score"]


def _key() -> str | None:
    k = os.getenv("MARKETAUX_API_KEY", "").strip()
    return k or None


def news(ticker: str, *, asset_class: str = "us", lookback_days: int = 3,
         limit: int = 50) -> pd.DataFrame:
    key = _key()
    if not key:
        return pd.DataFrame(columns=_COLS)
    if asset_class not in ("us", "india"):
        return pd.DataFrame(columns=_COLS)

    sym = (ticker if asset_class == "us" or "." in ticker or ticker.startswith("^")
           else f"{ticker}.NS")
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M")
    try:
        r = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params={"symbols": sym, "filter_entities": "true",
                     "language": "en", "published_after": since,
                     "limit": min(limit, 100), "api_token": key}, timeout=30,
        )
        if r.status_code != 200:
            log.warning(f"marketaux {r.status_code}: {r.text[:200]}")
            return pd.DataFrame(columns=_COLS)
        data = r.json().get("data", [])
    except Exception as e:
        log.warning(f"marketaux fetch failed for {ticker}: {e}")
        return pd.DataFrame(columns=_COLS)

    rows = []
    for a in data:
        # Marketaux gives per-entity sentiment; use first matching entity's score.
        score = 0.0
        for ent in a.get("entities", []):
            if ent.get("symbol", "").upper().startswith(ticker.upper()):
                try:
                    score = float(ent.get("sentiment_score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                break
        rows.append({
            "published_at": a.get("published_at"),
            "source": a.get("source", "marketaux"),
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "url": a.get("url", ""),
            "ticker_hint": ticker,
            "lang": "en",
            "sent_score": score,
        })
    df = pd.DataFrame(rows, columns=_COLS)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    return df
