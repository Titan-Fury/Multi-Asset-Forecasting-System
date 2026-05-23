"""Earnings-momentum sentiment z-score from existing news stream.

Phase 25.1 — instead of standing up a full transcript ingestion pipeline
(legal/scraping risk on free sources), we filter the existing
`bundle.news_raw` for earnings-related items and compute a 30-day
trailing sentiment z-score per ticker. This captures the same
predictive signal (acceleration/deceleration of earnings sentiment)
without new ingestion infrastructure.

Earnings keyword set is curated for both India and US English business
reporting (yfinance / RSS / Reddit / MoneyControl / Economic Times
typically use these terms).

Output column: `earnings_sentiment_z_30d`

Wired into:
  - `omni.features.build` (offline training-set builder reads from the
    aggregated sentiment frame; feature is added per-row on each ticker)
  - `omni.ondemand.predict` (live row patcher computes from
    bundle.news_raw at score time)

Graceful degradation: returns NaN when no earnings-tagged items in
window, or when sentiment dispersion is too low to form a stable z.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Curated keyword set for earnings-related news items. Matched
# case-insensitively as substrings of headline + body.
_EARN_PATTERNS = (
    r"\bearnings?\b",
    r"\beps\b",
    r"\bquarter\w*\s*(?:result|earnings)",
    r"\bq[1-4]\s*(?:fy)?",
    r"\bguidance\b",
    r"\bmiss(?:es|ed)?\s+(?:estimate|expectation|consensus)",
    r"\bbeat(?:s|en)?\s+(?:estimate|expectation|consensus)",
    r"\brevenue\s+(?:growth|decline|jump|fall)",
    r"\bprofit\s+(?:growth|decline|jump|fall)",
    r"\boutlook\b",
    r"\bforward\s+guidance",
    r"\bconference\s+call\b",
    r"\bearnings\s+call\b",
    r"\b(?:profit|loss)\s+widened",
)
_EARN_RE = re.compile("|".join(_EARN_PATTERNS), re.IGNORECASE)


def _is_earnings_item(title: str, body: str | None = None) -> bool:
    """True if the headline (or optional body) matches an earnings pattern."""
    text = str(title or "")
    if body:
        text = f"{text} {body}"
    if not text.strip():
        return False
    return bool(_EARN_RE.search(text))


def earnings_sentiment_z_30d(news: pd.DataFrame, *, lookback_days: int = 30,
                              min_items: int = 3) -> float | None:
    """Return latest 30d earnings-news sentiment z-score, or None.

    `news` is a long-form frame with columns at minimum:
      title, sent_score, published_at (or date)

    Z-scoring: latest day's mean earnings sentiment vs the trailing
    30d mean+std (rolling, min_periods=min_items).
    """
    if news is None or news.empty:
        return None
    if "sent_score" not in news.columns:
        return None

    df = news.copy()
    if "title" not in df.columns:
        df["title"] = ""
    body_col = "summary" if "summary" in df.columns else None

    # Phase 24.21 — vectorized earnings filter. Was `df.apply(axis=1)` calling
    # `_is_earnings_item` per row, which materializes a pd.Series and Python
    # frame for every row. Compose the haystack as one Series and run a single
    # vectorized regex match — same semantics, ~20-100x faster.
    haystack = df["title"].fillna("").astype(str)
    if body_col:
        haystack = haystack + " " + df[body_col].fillna("").astype(str)
    df["_is_earn"] = haystack.str.contains(_EARN_RE, na=False)
    earn = df[df["_is_earn"]].copy()
    if earn.empty:
        return None

    if "published_at" in earn.columns:
        earn["date"] = pd.to_datetime(earn["published_at"], errors="coerce").dt.normalize()
    elif "date" in earn.columns:
        earn["date"] = pd.to_datetime(earn["date"], errors="coerce").dt.normalize()
    else:
        return None
    earn = earn.dropna(subset=["date", "sent_score"])
    if len(earn) < min_items:
        return None

    today = earn["date"].max()
    cutoff = today - pd.Timedelta(days=lookback_days)
    earn = earn[earn["date"] >= cutoff]
    if len(earn) < min_items:
        return None

    daily = earn.groupby("date")["sent_score"].mean().sort_index()
    if len(daily) < min_items:
        return None

    mu = daily.mean()
    sd = daily.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return None
    z = (daily.iloc[-1] - mu) / sd
    if not np.isfinite(z):
        return None
    return float(z)


__all__ = ["earnings_sentiment_z_30d", "_is_earnings_item"]
