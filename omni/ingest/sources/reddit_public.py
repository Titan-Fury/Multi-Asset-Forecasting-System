"""Reddit public-JSON fetcher (no auth, no API key).

Reddit exposes its listing/search pages as JSON simply by appending ``.json``
to the URL. These endpoints are public read-only views — same data a
logged-out browser sees — so we never authenticate and your Reddit account
is *not* used or implicated.

Polite-use rules we follow:
- Realistic ``User-Agent`` identifying the project (Reddit rate-limits
  generic UAs like ``python-requests/2.x`` aggressively).
- 1-second floor between calls.
- On 429, return empty and cache the empty result briefly so we don't retry-storm.
- Cap volume to ~2 requests per ticker (one search per relevant subreddit).
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

from omni.ondemand.cache import cached
from omni.utils.logger import log


_UA = os.environ.get(
    "REDDIT_USER_AGENT",
    "Omni_Stocks/0.1 research-only (contact: research-noreply@example.com)",
)
_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}
_TIMEOUT = 15

# Subreddits that produce trader/investor signal worth looking at.
_DEFAULT_SUBS_US = ("wallstreetbets", "stocks", "investing", "StockMarket")
_DEFAULT_SUBS_INDIA = ("IndianStockMarket", "IndiaInvestments", "DalalStreetTalks")
_DEFAULT_SUBS_CRYPTO = ("CryptoCurrency", "CryptoMarkets")

_RATE_SLEEP = 1.0
_LAST_CALL = {"t": 0.0}


def _polite_sleep() -> None:
    """Ensure at least ``_RATE_SLEEP`` seconds between outgoing calls."""
    elapsed = time.monotonic() - _LAST_CALL["t"]
    if elapsed < _RATE_SLEEP:
        time.sleep(_RATE_SLEEP - elapsed)
    _LAST_CALL["t"] = time.monotonic()


def _subs_for(asset_class: str) -> tuple[str, ...]:
    if asset_class == "india":
        return _DEFAULT_SUBS_INDIA
    if asset_class == "crypto":
        return _DEFAULT_SUBS_CRYPTO
    return _DEFAULT_SUBS_US


def _search_one(sub: str, query: str, limit: int = 25) -> list[dict]:
    """Fetch search.json for a single subreddit. Empty on any failure."""
    url = (f"https://www.reddit.com/r/{sub}/search.json"
           f"?q={requests.utils.quote(query)}"
           f"&restrict_sr=1&sort=new&limit={limit}&t=month")
    try:
        _polite_sleep()
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code == 429:
            log.warning(f"reddit 429 on r/{sub} — backing off")
            return []
        r.raise_for_status()
        data = r.json() or {}
        children = data.get("data", {}).get("children", []) or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except Exception as e:
        log.warning(f"reddit search r/{sub} q={query!r} failed: {e}")
        return []


def _score_post(p: dict) -> float:
    """Cheap heuristic sentiment from upvote ratio + score.

    Real per-post sentiment is computed downstream by the LLM router on
    the unioned news frame; this just gives a polarity hint to break ties.
    """
    ratio = float(p.get("upvote_ratio", 0.5) or 0.5)
    # Map 0.5 ratio -> 0.0 sentiment, 1.0 ratio -> +1.0, 0.0 ratio -> -1.0
    return max(-1.0, min(1.0, (ratio - 0.5) * 2.0))


@cached(ttl=1800, key="reddit_pub:{ticker}:{display_name}:{asset_class}")
def posts(ticker: str, *, display_name: str = "",
          asset_class: str = "us") -> pd.DataFrame:
    """Recent Reddit posts mentioning ``ticker`` or ``display_name``.

    Returns a DataFrame compatible with the news union schema:
        ``title``, ``url``, ``source``, ``published_at``, ``sent_score``.
    """
    queries: list[str] = [ticker.upper()]
    dn = (display_name or "").strip()
    if dn and len(dn) >= 4 and dn.upper() != ticker.upper():
        # Use the head of the display name (e.g. "Garden Reach" not the full legal name)
        head = " ".join(dn.split()[:2])
        queries.append(head)

    rows: list[dict] = []
    for sub in _subs_for(asset_class):
        for q in queries:
            for p in _search_one(sub, q, limit=15):
                title = (p.get("title") or "").strip()
                if len(title) < 5:
                    continue
                created = p.get("created_utc")
                if not created:
                    continue
                rows.append({
                    "title": f"r/{sub}: {title}",
                    "url": "https://www.reddit.com" + (p.get("permalink") or ""),
                    "source": f"reddit_{sub}",
                    "published_at": pd.to_datetime(int(created), unit="s", utc=True),
                    "sent_score": _score_post(p),
                    # Extras kept around in case downstream wants them
                    "score": int(p.get("score", 0) or 0),
                    "num_comments": int(p.get("num_comments", 0) or 0),
                    "upvote_ratio": float(p.get("upvote_ratio", 0.5) or 0.5),
                })
            # Only one query per sub if first returned plenty
            if len([r for r in rows if r["source"] == f"reddit_{sub}"]) >= 10:
                break

    if not rows:
        return pd.DataFrame(columns=["title", "url", "source",
                                      "published_at", "sent_score"])
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["title"]).sort_values(
        "published_at", ascending=False).reset_index(drop=True)
    return df


@cached(ttl=1800, key="reddit_pub:summary:{ticker}:{asset_class}")
def crowd_intensity(ticker: str, *, asset_class: str = "us") -> dict:
    """Lightweight rollup: post count + avg sentiment + comment volume."""
    df = posts(ticker, asset_class=asset_class)
    if df is None or df.empty:
        return {"reddit_post_count_30d": 0, "reddit_avg_sent": 0.0,
                "reddit_total_comments": 0}
    return {
        "reddit_post_count_30d": int(len(df)),
        "reddit_avg_sent": float(df["sent_score"].mean()),
        "reddit_total_comments": int(df.get("num_comments", pd.Series([0])).sum()),
    }
