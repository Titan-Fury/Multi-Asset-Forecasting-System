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

from omni.llm.backends.base import BackendError
from omni.llm.router import route
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


def _batch_score_finbert(texts: list[str]) -> list[float] | None:
    """Phase 24.21 — try to score every text in a single FinBERT batch
    dispatch. Returns the list of signed scores aligned to `texts`, or None
    if the FinBERT backend isn't available (caller falls back to the
    per-item router path so behaviour is preserved on machines without
    transformers/torch).

    Why: the previous per-row `route("sentiment_per_article", text)` loop
    was the single biggest hot-path bottleneck (200 articles = 200 sequential
    model calls). Native HF pipeline batching is 10-50x faster on GPU and
    still 3-5x faster on CPU because we kill the Python-loop overhead.
    """
    try:
        from omni.llm.backends.finbert import FinBERTBackend
    except Exception:  # noqa: BLE001
        return None
    try:
        # Reuse the FinBERTBackend instance the router already constructed
        # if one exists, so we don't double-load the model.
        from omni.llm.router import Router
        router = Router.get()
        be = router._backends.get("finbert")  # type: ignore[attr-defined]
        if be is None:
            be = FinBERTBackend("finbert", router.cfg.get("backends", {}).get("finbert", {}))
            router._backends["finbert"] = be  # type: ignore[attr-defined]
        results = be.complete_batch(texts)
    except Exception as e:  # noqa: BLE001
        log.debug(f"finbert batch unavailable, falling back to per-item route: {e}")
        return None
    return [float(r.extra.get("signed_score", 0.0)) for r in results]


def score_articles(news: pd.DataFrame) -> pd.DataFrame:
    """Add a `sent_score` column in [-1, 1] per row using the router.

    Router task `sentiment_per_article` -> finbert. Cached per (text), so
    repeated runs on the same articles are free.

    Phase 24.21 — vectorized: one FinBERT batch call instead of N router
    calls inside iterrows(). Falls back to the per-row router if FinBERT
    isn't installed locally so the public behaviour is unchanged.
    """
    if news.empty:
        return news.assign(sent_score=[])
    out = news.copy()

    # Build the text column once, vectorized. `(title + ". " + description)`
    # mirrors the prior per-row construction; `.str.slice(0, 512)` matches
    # the FinBERT 512-token cap. Empty strings are scored as 0.0 below.
    titles = out.get("title", pd.Series([""] * len(out))).fillna("").astype(str)
    descs = out.get("description", pd.Series([""] * len(out))).fillna("").astype(str)
    texts = (titles + ". " + descs).str.strip().str.slice(0, 512)

    non_empty_mask = texts.str.len() > 0
    non_empty_texts = texts[non_empty_mask].tolist()

    scores = [0.0] * len(out)
    batch_scores = _batch_score_finbert(non_empty_texts) if non_empty_texts else []
    if batch_scores is not None and len(batch_scores) == len(non_empty_texts):
        # Fast path: FinBERT batch worked.
        idx_iter = iter(batch_scores)
        for i, ok in enumerate(non_empty_mask):
            if ok:
                scores[i] = next(idx_iter)
    else:
        # Slow fallback: per-row router (preserves original semantics, works
        # without finbert installed).
        for i, (txt, ok) in enumerate(zip(texts, non_empty_mask)):
            if not ok:
                continue
            try:
                res = route("sentiment_per_article", txt)
                scores[i] = float(res.extra.get("signed_score", 0.0))
            except BackendError as e:
                log.warning(f"sentiment scoring failed: {e}")

    out["sent_score"] = scores

    # Phase 24.21 — vectorized keyword tagging. Was three df.apply(axis=1)
    # passes calling _kw_match per row; that's O(rows × 3) Python invocations.
    # `str.contains` with a compiled regex runs in pandas/C and is 20-100x
    # faster on typical article counts.
    text_combined = (titles + " " + descs)
    out["panic_kw"] = text_combined.str.contains(_PANIC_KEYWORDS, na=False).astype(int)
    out["promoter_kw"] = text_combined.str.contains(_PROMOTER_KEYWORDS, na=False).astype(int)
    out["sebi_kw"] = text_combined.str.contains(_SEBI_KEYWORDS, na=False).astype(int)
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

    # Phase 24.21 — was using `grp.apply(lambda s: (s>0.2).mean())` for
    # sent_pos_share / sent_neg_share, which invokes Python per group. Tag
    # bool flags first so we can use a vectorized .mean() aggregation that
    # stays in C.
    df["_pos_flag"] = (df["sent_score"] > 0.2).astype(float)
    df["_neg_flag"] = (df["sent_score"] < -0.2).astype(float)

    grp = df.groupby(["ticker", "date"])
    agg = pd.DataFrame({
        "n_articles": grp.size(),
        "sent_mean": grp["sent_score"].mean(),
        "sent_std": grp["sent_score"].std(ddof=0),
        "sent_pos_share": grp["_pos_flag"].mean(),
        "sent_neg_share": grp["_neg_flag"].mean(),
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

    # Phase 24.22 — Twitter / X side-channel features.
    # When Apify Twitter ingest is enabled, the rows arrive with
    # `source == "twitter"`. We compute a Twitter-only daily sentiment
    # so the model can separate retail-FinTwit signal from formal news
    # (often it leads news by hours and is more reactive in panic regimes).
    # Columns are NaN for tickers with no twitter coverage that day; the
    # downstream feature builder treats NaN as "no signal" rather than 0.
    if "source" in df.columns:
        tw_mask = df["source"].astype(str).str.lower() == "twitter"
        if tw_mask.any():
            tw = df[tw_mask].copy()
            tw_grp = tw.groupby(["ticker", "date"])
            tw_agg = pd.DataFrame({
                "twitter_n_24h": tw_grp.size(),
                "twitter_sent_mean_24h": tw_grp["sent_score"].mean(),
                "twitter_sent_std_24h": tw_grp["sent_score"].std(ddof=0),
            }).reset_index()
            tw_agg["date"] = pd.to_datetime(tw_agg["date"])
            agg = agg.merge(tw_agg, on=["ticker", "date"], how="left")
        else:
            agg["twitter_n_24h"] = 0
            agg["twitter_sent_mean_24h"] = pd.NA
            agg["twitter_sent_std_24h"] = pd.NA
    else:
        agg["twitter_n_24h"] = 0
        agg["twitter_sent_mean_24h"] = pd.NA
        agg["twitter_sent_std_24h"] = pd.NA

    return agg
