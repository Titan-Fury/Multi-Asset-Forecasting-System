"""Indian financial news RSS feeds — fast, free, no-auth.

Polls multiple Indian outlets in parallel for ticker-relevant headlines.
Output schema matches other news fetchers:
    title, url, source, published_at, sent_score (0 → caller computes)

Outlets:
    moneycontrol     — most widely-read Indian retail finance site
    economictimes    — ET Markets section (BSE/NSE focus)
    livemint         — Mint markets
    business_std     — Business Standard markets
    ndtv_profit      — NDTV Profit (formerly Gadgets360 Now)

Each fetch is best-effort; failures are logged and yield empty rows. Total
budget across outlets is ~6 seconds with 3-second per-outlet timeout.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"}
_TIMEOUT = 4.0


# Per-outlet feed list. Multiple feeds per outlet improves recall.
_FEEDS = {
    "moneycontrol": [
        "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "https://www.moneycontrol.com/rss/buzzingstocks.xml",
        "https://www.moneycontrol.com/rss/results.xml",
        "https://www.moneycontrol.com/rss/marketreports.xml",
    ],
    "economictimes": [
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    ],
    "livemint": [
        "https://www.livemint.com/rss/markets",
        "https://www.livemint.com/rss/money",
        "https://www.livemint.com/rss/companies",
    ],
    "hindu_businessline": [
        "https://www.thehindubusinessline.com/markets/feeder/default.rss",
        "https://www.thehindubusinessline.com/companies/feeder/default.rss",
        "https://www.thehindubusinessline.com/markets/stock-markets/feeder/default.rss",
    ],
}


def _strip_html(s: str) -> str:
    s = s or ""
    # Remove CDATA wrappers (ET, LiveMint, BS use these heavily)
    s = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", s, flags=re.DOTALL)
    # Decode common entities BEFORE stripping tags
    s = (s.replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')
            .replace("&lt;", "<").replace("&gt;", ">").replace("&#x27;", "'")
            .replace("&nbsp;", " "))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_rss(xml: str, source: str) -> list[dict]:
    """Tiny RSS / Atom parser. Handles CDATA sections used by ET/Mint/BS."""
    out: list[dict] = []
    if not xml:
        return out
    items = re.findall(r"<item[^>]*>(.*?)</item>", xml, flags=re.DOTALL | re.IGNORECASE)
    if not items:
        items = re.findall(r"<entry[^>]*>(.*?)</entry>", xml, flags=re.DOTALL | re.IGNORECASE)
    for item in items[:80]:
        # Strip CDATA at the item level so the inner regexes don't trip on `]]>`
        item_clean = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", item, flags=re.DOTALL)
        title_m = re.search(r"<title[^>]*>(.*?)</title>",
                             item_clean, flags=re.DOTALL | re.IGNORECASE)
        link_m = re.search(r"<link[^>]*>([^<]+)</link>",
                            item_clean, flags=re.IGNORECASE)
        if not link_m:
            link_m = re.search(r'<link[^>]*href="([^"]+)"',
                                item_clean, flags=re.IGNORECASE)
        date_m = re.search(r"<(?:pubDate|published|updated|dc:date)[^>]*>([^<]+)</",
                            item_clean, flags=re.IGNORECASE)
        desc_m = re.search(r"<(?:description|summary|content)[^>]*>(.*?)</(?:description|summary|content)>",
                            item_clean, flags=re.DOTALL | re.IGNORECASE)
        title = _strip_html(title_m.group(1)) if title_m else ""
        link = (link_m.group(1).strip() if link_m else "")
        if not title:
            continue
        try:
            ts = (pd.to_datetime(date_m.group(1).strip(), utc=True, errors="coerce")
                   if date_m else None)
        except Exception:
            ts = None
        out.append({
            "title": title,
            "url": link,
            "source": source,
            "published_at": (ts.isoformat() if ts is not None and pd.notna(ts)
                              else datetime.now(timezone.utc).isoformat()),
            "summary": _strip_html(desc_m.group(1)) if desc_m else "",
            "sent_score": 0.0,
        })
    return out


def _fetch_one(url: str, source: str) -> list[dict]:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code != 200:
            log.debug(f"{source} {url} -> {r.status_code}")
            return []
        return _parse_rss(r.text, source)
    except Exception as e:  # noqa: BLE001
        log.debug(f"{source} {url} fetch failed: {e}")
        return []


def _matches_ticker(item: dict, *, ticker: str, display_name: str) -> bool:
    """Loose match: ticker root or any token of display_name >=3 chars."""
    haystack = (item.get("title", "") + " " + item.get("summary", "")).lower()
    if not haystack:
        return False
    root = re.sub(r"\.(NS|BO)$", "", ticker, flags=re.IGNORECASE).lower()
    if len(root) >= 3 and root in haystack:
        return True
    if display_name:
        # Strip "Ltd" / "Limited" / "Inc" suffixes; any token >=4 chars matches
        clean = re.sub(r"\b(ltd|limited|inc|corp|corporation|company|co)\b", "",
                        display_name.lower())
        toks = [t for t in re.split(r"[^a-z0-9]+", clean) if len(t) >= 4]
        for t in toks:
            if t in haystack:
                return True
    return False


@cached(ttl=TTL.news, key="indian_rss:{ticker}:{display_name}")
def fetch_indian_news(ticker: str, display_name: str = "") -> pd.DataFrame:
    """Pull Indian-financial-press headlines and filter to those mentioning the ticker.

    Returns a DataFrame with [title, url, source, published_at, sent_score]
    sorted newest-first. Empty DataFrame on total failure.
    """
    t0 = time.time()
    rows: list[dict] = []
    for source, urls in _FEEDS.items():
        for u in urls:
            rows.extend(_fetch_one(u, source))
            if time.time() - t0 > 8.0:  # global budget
                break
        if time.time() - t0 > 8.0:
            break
    if not rows:
        return pd.DataFrame(columns=["title", "url", "source", "published_at", "sent_score"])

    matched = [r for r in rows if _matches_ticker(r, ticker=ticker, display_name=display_name)]
    if not matched:
        return pd.DataFrame(columns=["title", "url", "source", "published_at", "sent_score"])

    df = pd.DataFrame(matched).drop(columns=["summary"], errors="ignore")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = (df.dropna(subset=["title"])
            .drop_duplicates(subset=["title"])
            .sort_values("published_at", ascending=False, na_position="last")
            .reset_index(drop=True))
    df["published_at"] = df["published_at"].astype(str)
    return df.head(40)


def market_pulse() -> pd.DataFrame:
    """Top Indian-press headlines unfiltered — used for macro-flavour features.

    Returns a small DataFrame of recent headlines from all Indian outlets.
    Cached at the news TTL.
    """
    return _market_pulse_cached()


@cached(ttl=TTL.news, key="indian_rss_pulse:v1")
def _market_pulse_cached() -> pd.DataFrame:
    rows: list[dict] = []
    for source, urls in _FEEDS.items():
        for u in urls[:1]:
            rows.extend(_fetch_one(u, source))
    if not rows:
        return pd.DataFrame(columns=["title", "url", "source", "published_at"])
    df = pd.DataFrame(rows).drop(columns=["summary", "sent_score"], errors="ignore")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = (df.dropna(subset=["title"])
            .drop_duplicates(subset=["title"])
            .sort_values("published_at", ascending=False, na_position="last")
            .head(50)
            .reset_index(drop=True))
    df["published_at"] = df["published_at"].astype(str)
    return df
