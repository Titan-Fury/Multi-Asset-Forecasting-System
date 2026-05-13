"""Indian financial blogs — Finshots, Trade Brains, Smallcase.

These are slow-moving narrative-quality content (1-3 articles/day) but
each piece is highly editorialized and surface real retail attention.
Lower frequency than ET/Mint but useful as a long-form sentiment proxy.

Output schema: [title, url, source, published_at, sent_score].
"""
from __future__ import annotations

import re

import pandas as pd
import requests

from omni.ingest.sources.indian_rss import _parse_rss
from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"}

_FEEDS = {
    "finshots": [
        "https://finshots.in/feed/",
        "https://finshots.in/markets/feed/",
    ],
    "trade_brains": [
        "https://tradebrains.in/feed/",
    ],
    "smallcase": [
        "https://www.smallcase.com/blog/feed/",
    ],
    "the_ken": [
        "https://the-ken.com/feed/",
    ],
}


def _fetch_one(url: str, source: str) -> list[dict]:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=4)
        if r.status_code != 200:
            return []
        return _parse_rss(r.text, source)
    except Exception as e:  # noqa: BLE001
        log.debug(f"{source} {url}: {e}")
        return []


@cached(ttl=TTL.news, key="indian_blogs:{ticker}:{display_name}")
def fetch_blog_news(ticker: str, display_name: str = "") -> pd.DataFrame:
    """Pull Finshots/TradeBrains/Smallcase/Ken posts mentioning the ticker."""
    rows: list[dict] = []
    for source, urls in _FEEDS.items():
        for u in urls:
            rows.extend(_fetch_one(u, source))
    if not rows:
        return pd.DataFrame(columns=["title", "url", "source", "published_at", "sent_score"])

    root = re.sub(r"\.(NS|BO)$", "", ticker, flags=re.IGNORECASE).lower()
    tokens = []
    if len(root) >= 3:
        tokens.append(root)
    if display_name:
        clean = re.sub(r"\b(ltd|limited|inc|corp|corporation|company|co)\b",
                        "", display_name.lower())
        for t in re.split(r"[^a-z0-9]+", clean):
            if len(t) >= 4:
                tokens.append(t)

    matched = []
    for r in rows:
        haystack = (r.get("title", "") + " " + r.get("summary", "")).lower()
        if any(t in haystack for t in tokens):
            matched.append(r)

    if not matched:
        return pd.DataFrame(columns=["title", "url", "source", "published_at", "sent_score"])

    df = pd.DataFrame(matched).drop(columns=["summary"], errors="ignore")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = (df.dropna(subset=["title"])
            .drop_duplicates(subset=["title"])
            .sort_values("published_at", ascending=False, na_position="last")
            .reset_index(drop=True))
    df["published_at"] = df["published_at"].astype(str)
    return df.head(20)
