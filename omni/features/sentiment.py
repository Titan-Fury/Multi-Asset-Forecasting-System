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
  sent_neg_share, sent_momentum_7d, panic_kw_count, promoter_kw_count,
  sebi_kw_count
"""
from __future__ import annotations

import re

import pandas as pd

from omni.llm.router import route
from omni.llm.backends.base import BackendError
from omni.utils.logger import log


# Phase 24.18 — keyword groups for "narrative" features. These are matched
# case-insensitively against title+description text. The counts are
# aggregated daily and exposed downstream as panic_keyword_ratio,
# promoter_news_flag, sebi_action_flag (computed in sentiment_z.py).
_PANIC_KEYWORDS = re.compile(
    r"\b(circuit|lower\s+circuit|crash|freeze|halt|tumble|plunge|"
    r"nosedive|slump|panic|sell\s*off|sell-off|rout|bloodbath|"
    r"slide|free\s*fall|free-fall)\b",
    re.IGNORECASE,
)
_PROMOTER_KEYWORDS = re.compile(
    r"\b(promoter|sast|insider\s+(?:trade|trading|buy|sell|deal)|"
    r"pledge|pledg(?:ed|ing)|bulk\s+deal|block\s+deal)\b",
    re.IGNORECASE,
)
_SEBI_KEYWORDS = re.compile(
    r"\b(sebi|investigation|probe|notice|debar|fraud|scam|"
    r"violation|enforcement\s+action)\b",
    re.IGNORECASE,
)


def _kw_match(row: pd.Series, regex: re.Pattern) -> int:
    """Count regex hits on title+description for one news row. Returns 1 if
    any match, else 0 (binary per-article flag — counts at the day level
    become 'how many articles mentioned X'."""
    text = f"{row.get('title', '')} {row.get('description', '')}"
    return 1 if regex.search(text) else 0


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
    # Phase 24.18 — also tag each article with binary keyword-group flags so
    # `aggregate()` can roll them up alongside the sentiment scores.
    out["panic_kw"] = out.apply(lambda r: _kw_match(r, _PANIC_KEYWORDS), axis=1)
    out["promoter_kw"] = out.apply(lambda r: _kw_match(r, _PROMOTER_KEYWORDS), axis=1)
    out["sebi_kw"] = out.apply(lambda r: _kw_match(r, _SEBI_KEYWORDS), axis=1)
    return out


def aggregate(scored: pd.DataFrame, *, ticker_col: str = "ticker_hint") -> pd.DataFrame:
    """Roll daily sentiment per (ticker, UTC date)."""
    if scored.empty or "sent_score" not in scored.columns:
        return pd.DataFrame(columns=[
            "date", "ticker", "n_articles", "sent_mean", "sent_std",
            "sent_pos_share", "sent_neg_share", "sent_momentum_7d",
            "panic_kw_count", "promoter_kw_count", "sebi_kw_count",
        ])
    df = scored.copy()
    df["date"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df["ticker"] = df[ticker_col].astype(str)
    df = df[df["ticker"].str.len() > 0]
    # Backfill keyword flags if score_articles wasn't called (e.g. external
    # feed bypassing the FinBERT router). Defensive — never crash here.
    for col in ("panic_kw", "promoter_kw", "sebi_kw"):
        if col not in df.columns:
            df[col] = 0

    grp = df.groupby(["ticker", "date"])
    agg = pd.DataFrame({
        "n_articles": grp.size(),
        "sent_mean": grp["sent_score"].mean(),
        "sent_std": grp["sent_score"].std(ddof=0),
        "sent_pos_share": grp["sent_score"].apply(lambda s: (s > 0.2).mean()),
        "sent_neg_share": grp["sent_score"].apply(lambda s: (s < -0.2).mean()),
        "panic_kw_count": grp["panic_kw"].sum(),
        "promoter_kw_count": grp["promoter_kw"].sum(),
        "sebi_kw_count": grp["sebi_kw"].sum(),
    }).reset_index()

    # Momentum: today's mean vs trailing 7-day mean, per ticker
    agg = agg.sort_values(["ticker", "date"]).reset_index(drop=True)
    agg["sent_momentum_7d"] = (
        agg.groupby("ticker")["sent_mean"]
           .transform(lambda s: s - s.shift(1).rolling(7, min_periods=2).mean())
    )
    agg["date"] = pd.to_datetime(agg["date"])
    return agg
