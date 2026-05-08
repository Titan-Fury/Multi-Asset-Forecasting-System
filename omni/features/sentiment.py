"""Sentiment features.

Pipeline:
  1. Pull raw news rows for a ticker / date window.
  2. Score each (title + description) via the LLM router task
     `sentiment_per_article` (FinBERT primary).
  3. Aggregate per (ticker, date): mean signed score, count, std, momentum
     (today vs trailing 7d mean).

Sentiment is computed in UTC date buckets and then **lagged by 1 trading day**
when joined to features (handled by features.build) to avoid look-ahead from
articles published after market close.

Output schema:
  date, ticker, n_articles, sent_mean, sent_std, sent_pos_share,
  sent_neg_share, sent_momentum_7d
"""
from __future__ import annotations

import pandas as pd

from omni.llm.router import route
from omni.llm.backends.base import BackendError
from omni.utils.logger import log


def score_articles(news: pd.DataFrame) -> pd.DataFrame:
    """Add a `sent_score` column in [-1, 1] per row using the router.

    Router task `sentiment_per_article` -> finbert. Cached per (text), so
    repeated runs on the same articles are free.
    """
    if news.empty:
        return news.assign(sent_score=[])
    out = news.copy()
    scores: list[float] = []
    for _, r in out.iterrows():
        text = f"{r.get('title', '')}. {r.get('description', '')}".strip()
        if not text:
            scores.append(0.0)
            continue
        try:
            res = route("sentiment_per_article", text[:512])
            scores.append(float(res.extra.get("signed_score", 0.0)))
        except BackendError as e:
            log.warning(f"sentiment scoring failed: {e}")
            scores.append(0.0)
    out["sent_score"] = scores
    return out


def aggregate(scored: pd.DataFrame, *, ticker_col: str = "ticker_hint") -> pd.DataFrame:
    """Roll daily sentiment per (ticker, UTC date)."""
    if scored.empty or "sent_score" not in scored.columns:
        return pd.DataFrame(columns=[
            "date", "ticker", "n_articles", "sent_mean", "sent_std",
            "sent_pos_share", "sent_neg_share", "sent_momentum_7d",
        ])
    df = scored.copy()
    df["date"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df["ticker"] = df[ticker_col].astype(str)
    df = df[df["ticker"].str.len() > 0]

    grp = df.groupby(["ticker", "date"])
    agg = pd.DataFrame({
        "n_articles": grp.size(),
        "sent_mean": grp["sent_score"].mean(),
        "sent_std": grp["sent_score"].std(ddof=0),
        "sent_pos_share": grp["sent_score"].apply(lambda s: (s > 0.2).mean()),
        "sent_neg_share": grp["sent_score"].apply(lambda s: (s < -0.2).mean()),
    }).reset_index()

    # Momentum: today's mean vs trailing 7-day mean, per ticker
    agg = agg.sort_values(["ticker", "date"]).reset_index(drop=True)
    agg["sent_momentum_7d"] = (
        agg.groupby("ticker")["sent_mean"]
           .transform(lambda s: s - s.shift(1).rolling(7, min_periods=2).mean())
    )
    agg["date"] = pd.to_datetime(agg["date"])
    return agg
