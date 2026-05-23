"""Apify Twitter / X scraper client.

Indian FinTwit is a major sentiment leading-indicator that the project
otherwise has zero coverage on (StockTwits is US-only; Reddit is slow).
This module wraps Apify's `apidojo/tweet-scraper` actor (or any
configured equivalent) and emits rows in the SAME schema as
omni.ingest.news so they merge cleanly into the existing
sentiment pipeline.

Output schema (matches omni.ingest.news._NEWS_COLS plus optional
`sent_score` so callers that bypass FinBERT still get a baseline
score from engagement-weighted likes/retweets):

  published_at (UTC), source, title (= tweet text), description (=""),
  url (= tweet permalink), ticker_hint, lang,
  sent_score (optional, in [-1, 1])

Cost shape: at the time of writing, apidojo/tweet-scraper bills
~USD 0.25-0.40 per 1k tweets. For 100 tickers x 30 tweets/day that
is ~USD 22-36 per month. Tighten `max_items` per ticker to control
spend; the default below intentionally errs on the cheap side.

Failure model:
  - Missing APIFY_TOKEN -> empty DataFrame, debug log only (so the
    nightly orchestrator never breaks just because the toggle is off).
  - Actor run failure -> empty DataFrame + warning log.
  - Per-ticker failures -> skip that ticker, never abort the batch.
  - Result is silently merged with news; no separate feature is
    required to get a lift, but `twitter_only_view()` is provided
    for callers that want a Twitter-only aggregation.

Concurrency: a single actor call can take a list of search terms, so
we batch up to BATCH_SIZE tickers per Apify run to amortise the
warm-up cost. This is materially faster AND cheaper than calling
the actor once per ticker.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from omni.utils.logger import log

# ---- public constants the rest of the codebase can read --------------------

NEWS_COLS = [
    "published_at", "source", "title", "description",
    "url", "ticker_hint", "lang", "sent_score",
]
DEFAULT_ACTOR_ID = "apidojo/tweet-scraper"
DEFAULT_MAX_ITEMS_PER_TICKER = 30
DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_BATCH_SIZE = 8           # tickers per actor run
DEFAULT_RUN_TIMEOUT_S = 180

# Twitter-specific noise keywords that almost always indicate a junk
# match for stock tickers (e.g. "TATA" the airline vs TATA the steel
# stock). Caller can pass extra exclusions per-ticker.
_DEFAULT_EXCLUDE = (
    " giveaway", " airdrop", " join my channel", " telegram link",
)

_SENTIMENT_LEX_BULL = (
    "buy", "long", "moon", "breakout", "target hit", "multibagger",
    "bullish", "uptrend", "accumulate", "strong buy", "rocket",
)
_SENTIMENT_LEX_BEAR = (
    "sell", "short", "dump", "exit", "stoploss hit", "stop loss hit",
    "bearish", "downtrend", "avoid", "scam", "fraud", "circuit down",
)


def _have_token() -> bool:
    return bool(os.environ.get("APIFY_TOKEN", "").strip())


def _client():
    """Lazy import so the module stays importable without apify-client."""
    try:
        from apify_client import ApifyClient
    except ImportError:  # pragma: no cover
        log.debug("apify-client not installed; pip install apify-client")
        return None
    token = os.environ.get("APIFY_TOKEN", "").strip()
    if not token:
        return None
    return ApifyClient(token)


def _build_search_terms(tickers: Iterable[str], *,
                         cashtags: bool = True,
                         hashtags: bool = True,
                         lookback_hours: int = DEFAULT_LOOKBACK_HOURS) -> list[str]:
    """Build per-ticker search queries.

    For Indian equities cashtags ($RELIANCE) work but hashtags (#RELIANCE)
    drive most of the volume. We OR them together with `since:` operator
    so the actor itself filters by recency rather than us paying for
    older tweets we'd discard later.
    """
    since = (datetime.now(timezone.utc) - timedelta(hours=int(lookback_hours))
             ).strftime("%Y-%m-%d")
    terms: list[str] = []
    for t in tickers:
        t = (t or "").strip().upper()
        if not t:
            continue
        parts = []
        if cashtags:
            parts.append(f"${t}")
        if hashtags:
            parts.append(f"#{t}")
        # Twitter search supports OR. The since: operator restricts the
        # actor's crawl window so we don't pay per-tweet for stale data.
        q = "(" + " OR ".join(parts) + f") since:{since}" if parts else t
        terms.append(q)
    return terms


def _engagement_score(likes: int, retweets: int, replies: int, text: str) -> float:
    """Cheap baseline sentiment in [-1, 1] using lexicon + engagement weight.

    Used ONLY so callers that skip the FinBERT pipeline still get a usable
    score column. The proper sentiment is computed by features.sentiment
    (FinBERT) once these rows are merged into the news lake.
    """
    text_l = (text or "").lower()
    bull = sum(1 for w in _SENTIMENT_LEX_BULL if w in text_l)
    bear = sum(1 for w in _SENTIMENT_LEX_BEAR if w in text_l)
    if bull == 0 and bear == 0:
        return 0.0
    raw = (bull - bear) / max(bull + bear, 1)
    # Mild engagement weight: a 100-like tweet should weigh slightly more
    # than a 0-like tweet, but never override the lexicon sign.
    weight = 1.0 + min(0.5, (likes + 2 * retweets + replies) / 200.0)
    return max(-1.0, min(1.0, raw * weight))


def _normalise_actor_items(items: list[dict], ticker_hint: str) -> pd.DataFrame:
    """Convert apidojo/tweet-scraper output rows into our news schema."""
    rows = []
    for it in items or []:
        text = (it.get("text") or it.get("fullText") or "").strip()
        if not text:
            continue
        url = it.get("url") or it.get("twitterUrl") or ""
        if not url:
            tid = it.get("id") or it.get("tweetId")
            user = ((it.get("author") or {}).get("userName")
                    or it.get("authorUsername") or "i")
            if tid:
                url = f"https://x.com/{user}/status/{tid}"
        if not url:
            # Without a URL we can't dedup downstream; skip.
            continue
        likes = int(it.get("likeCount") or it.get("favoriteCount") or 0)
        retweets = int(it.get("retweetCount") or 0)
        replies = int(it.get("replyCount") or 0)
        rows.append({
            "published_at": it.get("createdAt") or it.get("created_at"),
            "source": "twitter",
            "title": text[:500],
            "description": "",
            "url": url,
            "ticker_hint": ticker_hint,
            "lang": it.get("lang") or "en",
            "sent_score": _engagement_score(likes, retweets, replies, text),
        })
    df = pd.DataFrame(rows, columns=NEWS_COLS)
    if not df.empty:
        df["published_at"] = pd.to_datetime(
            df["published_at"], errors="coerce", utc=True,
        )
        df = df.dropna(subset=["published_at"]).reset_index(drop=True)
    return df


def _run_actor(search_terms: list[str], *,
                actor_id: str,
                max_items: int,
                lang: str | None,
                run_timeout_s: int) -> list[dict]:
    """Invoke the Apify actor synchronously. Returns the dataset items
    (possibly empty) or [] on any failure."""
    client = _client()
    if client is None or not search_terms:
        return []
    run_input: dict = {
        "searchTerms": search_terms,
        # Both keys are accepted by various Apify Twitter actors so
        # callers can swap actor_id without rebuilding the input.
        "maxItems": int(max_items),
        "maxTweetsPerQuery": int(max_items),
        "sort": "Latest",
        "tweetLanguage": lang,
    }
    if not lang:
        run_input.pop("tweetLanguage", None)
    try:
        run = client.actor(actor_id).call(
            run_input=run_input, timeout_secs=int(run_timeout_s),
        )
    except Exception as e:  # noqa: BLE001
        log.warning(f"apify_twitter: actor call failed: {e}")
        return []
    if not run or "defaultDatasetId" not in run:
        log.warning("apify_twitter: actor returned no dataset id")
        return []
    try:
        page = client.dataset(run["defaultDatasetId"]).list_items(clean=True)
        return list(page.items or [])
    except Exception as e:  # noqa: BLE001
        log.warning(f"apify_twitter: dataset read failed: {e}")
        return []


def _split(seq: list, n: int) -> list[list]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def fetch_for_tickers(tickers: Iterable[str], *,
                      max_items_per_ticker: int = DEFAULT_MAX_ITEMS_PER_TICKER,
                      lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
                      batch_size: int = DEFAULT_BATCH_SIZE,
                      actor_id: str = DEFAULT_ACTOR_ID,
                      lang: str | None = "en",
                      cashtags: bool = True,
                      hashtags: bool = True,
                      sleep_between: float = 1.0,
                      run_timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
                      ) -> pd.DataFrame:
    """Public entry point. Returns a DataFrame in the news schema.

    Empty DataFrame on any failure / when APIFY_TOKEN is unset.
    Never raises (mirrors the rest of omni.ingest.sources).
    """
    if not _have_token():
        log.debug("apify_twitter: APIFY_TOKEN not set; skipping")
        return pd.DataFrame(columns=NEWS_COLS)
    tickers_list = [str(t).upper().strip() for t in tickers if str(t).strip()]
    if not tickers_list:
        return pd.DataFrame(columns=NEWS_COLS)

    parts: list[pd.DataFrame] = []
    batches = _split(tickers_list, max(1, int(batch_size)))
    for bi, batch in enumerate(batches):
        terms = _build_search_terms(
            batch, cashtags=cashtags, hashtags=hashtags,
            lookback_hours=lookback_hours,
        )
        if not terms:
            continue
        # We pay per result, so cap the actor's *total* output by the
        # batch size. Some actors interpret maxItems as "per query"
        # which is fine; the caps below converge.
        items = _run_actor(
            terms,
            actor_id=actor_id,
            max_items=max_items_per_ticker * len(batch),
            lang=lang,
            run_timeout_s=run_timeout_s,
        )
        # The actor may not return per-query attribution. Best-effort
        # ticker-hint resolution: scan each tweet text for the first
        # batch ticker that appears as $TICKER, #TICKER, or whole-word.
        df = _attribute_tickers(items, batch)
        if not df.empty:
            parts.append(df)
            log.info(
                f"apify_twitter: batch {bi+1}/{len(batches)} "
                f"({len(batch)} tickers) -> {len(df)} tweets"
            )
        if bi + 1 < len(batches) and sleep_between > 0:
            time.sleep(sleep_between)

    if not parts:
        return pd.DataFrame(columns=NEWS_COLS)
    out = pd.concat(parts, ignore_index=True)
    # Final dedup: same URL might match multiple tickers in a single batch
    # call (e.g. a tweet mentioning both RELIANCE and TCS). Keep the FIRST
    # ticker assignment per URL; a future per-row fan-out can be added if
    # cross-ticker mention matters for sentiment.
    out = out.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    return out


def _attribute_tickers(items: list[dict], batch: list[str]) -> pd.DataFrame:
    """Attribute each tweet in a multi-ticker batch back to a ticker hint.

    The actor does not tell us which search term produced which tweet, so
    we scan the tweet text for ticker mentions. Falls back to batch[0]
    only if no mention is detectable (rare but possible — actor sometimes
    returns context tweets without the cashtag in the body).
    """
    rows = []
    for it in items or []:
        text = (it.get("text") or it.get("fullText") or "")
        text_u = text.upper()
        chosen = None
        for t in batch:
            if (f"${t}" in text_u
                    or f"#{t}" in text_u
                    or f" {t} " in f" {text_u} "):
                chosen = t
                break
        if chosen is None:
            chosen = batch[0]
        rows.extend(
            _normalise_actor_items([it], ticker_hint=chosen).to_dict("records")
        )
    return pd.DataFrame(rows, columns=NEWS_COLS)


def twitter_only_view(news_df: pd.DataFrame) -> pd.DataFrame:
    """Filter a merged news frame down to just the Twitter rows.

    Useful when callers want a Twitter-specific sentiment column to
    sit alongside the unified `sent_mean` from the full news pipeline.
    """
    if news_df is None or news_df.empty or "source" not in news_df.columns:
        return news_df.iloc[0:0] if news_df is not None else pd.DataFrame()
    return news_df[news_df["source"].astype(str).str.lower() == "twitter"].copy()
