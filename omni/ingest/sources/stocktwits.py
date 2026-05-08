"""StockTwits client — retail sentiment from cashtags.

No API key required. ~200 requests/hour (rate-limited by IP).
Each "message" carries a sentiment annotation: Bullish / Bearish / null.

Schema returned:
  published_at (UTC), source, title (= message body), description (=""),
  url, ticker_hint, lang, sent_score (in {-1, 0, +1})
"""
from __future__ import annotations

import pandas as pd
import requests

from omni.utils.logger import log

_COLS = ["published_at", "source", "title", "description", "url",
         "ticker_hint", "lang", "sent_score"]


def messages(ticker: str, asset_class: str = "us", *, limit: int = 30) -> pd.DataFrame:
    """Fetch latest StockTwits messages for a US ticker. Limit max ~30 per call."""
    if asset_class not in ("us", "crypto"):
        return pd.DataFrame(columns=_COLS)
    sym = ticker if asset_class == "us" else f"{ticker}.X"  # crypto cashtag convention
    try:
        r = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json",
            params={"limit": min(int(limit), 30)}, timeout=20,
            headers={"User-Agent": "Omni_Stocks/0.1"},
        )
        if r.status_code != 200:
            log.warning(f"stocktwits {r.status_code}: {r.text[:200]}")
            return pd.DataFrame(columns=_COLS)
        data = r.json().get("messages", [])
    except Exception as e:
        log.warning(f"stocktwits fetch failed for {ticker}: {e}")
        return pd.DataFrame(columns=_COLS)

    rows = []
    for m in data:
        sent = (m.get("entities") or {}).get("sentiment") or {}
        b = sent.get("basic")
        score = {"Bullish": 1.0, "Bearish": -1.0}.get(b, 0.0)
        rows.append({
            "published_at": m.get("created_at"),
            "source": "stocktwits",
            "title": (m.get("body") or "")[:500],
            "description": "",
            "url": f"https://stocktwits.com/message/{m.get('id', '')}",
            "ticker_hint": ticker,
            "lang": "en",
            "sent_score": score,
        })
    df = pd.DataFrame(rows, columns=_COLS)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    return df
