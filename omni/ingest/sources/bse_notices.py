"""BSE corporate announcements & notices.

BSE publishes a public RSS of corporate notices (board meetings, results,
insider disclosures, AGM/EGM, dividends, splits, buybacks, etc.). This is
a reliable, free, no-auth feed — much friendlier than NSE.

URL: https://www.bseindia.com/data/xml/notices.xml
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd
import requests

from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA, "Accept": "*/*"}
_NOTICES_URL = "https://www.bseindia.com/data/xml/notices.xml"


def _strip(s: str) -> str:
    s = s or ""
    s = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", s, flags=re.DOTALL)
    s = (s.replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')
            .replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " "))
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@cached(ttl=TTL.news, key="bse_notices:v1")
def notices() -> pd.DataFrame:
    """Latest BSE notices. Schema: [title, url, source, published_at, sent_score]."""
    try:
        r = requests.get(_NOTICES_URL, headers=_HEADERS, timeout=6)
        if r.status_code != 200:
            log.debug(f"bse notices status {r.status_code}")
            return pd.DataFrame()
        items = re.findall(r"<item[^>]*>(.*?)</item>", r.text, flags=re.DOTALL | re.IGNORECASE)
        out = []
        for it in items[:80]:
            title_m = re.search(r"<title[^>]*>(.*?)</title>", it, flags=re.DOTALL | re.IGNORECASE)
            link_m = re.search(r"<link[^>]*>([^<]+)</link>", it, flags=re.IGNORECASE)
            date_m = re.search(r"<pubDate[^>]*>([^<]+)</pubDate>", it, flags=re.IGNORECASE)
            title = _strip(title_m.group(1)) if title_m else ""
            link = link_m.group(1).strip() if link_m else ""
            if not title:
                continue
            try:
                ts = pd.to_datetime(date_m.group(1).strip(), utc=True, errors="coerce") if date_m else None
            except Exception:
                ts = None
            out.append({
                "title": title, "url": link, "source": "bse_notices",
                "published_at": (ts.isoformat() if ts is not None and pd.notna(ts)
                                  else datetime.now(timezone.utc).isoformat()),
                "sent_score": 0.0,
            })
        if not out:
            return pd.DataFrame()
        df = pd.DataFrame(out)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df = (df.dropna(subset=["title"])
                .drop_duplicates(subset=["title"])
                .sort_values("published_at", ascending=False, na_position="last")
                .reset_index(drop=True))
        df["published_at"] = df["published_at"].astype(str)
        return df.head(60)
    except Exception as e:  # noqa: BLE001
        log.warning(f"bse notices fetch failed: {e}")
        return pd.DataFrame()


def matching_notices(ticker: str, display_name: str = "") -> pd.DataFrame:
    """Filter notices() to those mentioning the ticker or company-name tokens."""
    df = notices()
    if df is None or df.empty:
        return df
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
    if not tokens:
        return df.head(0)
    pattern = "|".join(re.escape(t) for t in tokens)
    mask = df["title"].astype(str).str.lower().str.contains(pattern, na=False)
    return df[mask].reset_index(drop=True)
