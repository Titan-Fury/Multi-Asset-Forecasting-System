"""Telegram public-channel news fetcher.

Telegram exposes a *public web view* at https://t.me/s/<channel> that
returns recent posts as HTML — no auth, no API key, no Bot token. We
parse the post divs and surface them as news rows.

Note: only works for channels that have set "public" + posting messages
visible without app login. Channels that require subscription are
silently skipped.

Default channels (Indian finance retail focus):
  marketsmojo_official     — institutional-style mid/small cap
  bombaydalal              — large dalal-street community
  sharemarketnews          — generic market news aggregator
  niftybnf_intraday        — Nifty / Bank Nifty intraday
  stockmarket_official     — generic
  squareoff_in             — squareoff.in's channel

Output schema: [title, url, source, published_at, sent_score].
"""
from __future__ import annotations

import re
from datetime import UTC, datetime

import pandas as pd
import requests

from omni.ingest.sources.indian_rss import _strip_html
from omni.ondemand.cache import TTL, cached
from omni.utils.logger import log

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"}

DEFAULT_CHANNELS = (
    "marketsmojo_official",
    "bombaydalal",
    "sharemarketnews",
    "niftybnf_intraday",
    "stockmarket_official",
    "squareoff_in",
)


def _parse_channel(html: str, channel: str) -> list[dict]:
    """Extract recent post text from t.me/s/<channel> HTML."""
    out = []
    # Each post is in <div class="tgme_widget_message_text js-message_text">...</div>
    blocks = re.findall(
        r'<div class="tgme_widget_message_text js-message_text"[^>]*>(.*?)</div>',
        html, flags=re.DOTALL,
    )
    times = re.findall(r'<time[^>]*datetime="([^"]+)"', html)
    for i, body in enumerate(blocks[:30]):
        text = _strip_html(body)
        if not text or len(text) < 10:
            continue
        ts = None
        if i < len(times):
            try:
                ts = pd.to_datetime(times[i], utc=True, errors="coerce")
            except Exception:
                ts = None
        out.append({
            "title": text[:200],
            "url": f"https://t.me/{channel}",
            "source": f"telegram_{channel}",
            "published_at": (ts.isoformat() if ts is not None and pd.notna(ts)
                              else datetime.now(UTC).isoformat()),
            "sent_score": 0.0,
        })
    return out


@cached(ttl=TTL.news, key="telegram_pulse:v1")
def channel_pulse(channels: tuple[str, ...] = DEFAULT_CHANNELS) -> pd.DataFrame:
    """Latest posts across the configured Telegram channels."""
    rows = []
    for c in channels:
        try:
            r = requests.get(f"https://t.me/s/{c}", headers=_HEADERS, timeout=4)
            if r.status_code == 200:
                rows.extend(_parse_channel(r.text, c))
        except Exception as e:  # noqa: BLE001
            log.debug(f"telegram {c} failed: {e}")
    if not rows:
        return pd.DataFrame(columns=["title", "url", "source", "published_at", "sent_score"])
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = (df.dropna(subset=["title"])
            .drop_duplicates(subset=["title"])
            .sort_values("published_at", ascending=False, na_position="last")
            .reset_index(drop=True))
    df["published_at"] = df["published_at"].astype(str)
    return df.head(60)


def fetch_telegram_news(ticker: str, display_name: str = "") -> pd.DataFrame:
    """Filter latest channel posts to those mentioning ticker or company tokens."""
    df = channel_pulse()
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
    haystack = df["title"].astype(str).str.lower()
    mask = pd.Series(False, index=df.index)
    for t in tokens:
        mask = mask | haystack.str.contains(re.escape(t), na=False)
    return df[mask].reset_index(drop=True).head(20)
