"""News ingestion: NewsAPI (free tier) + RSS feeds + GDelt.

Output schema (parquet, raw tier + lake):
  published_at (UTC), source, title, description, url, ticker_hint, lang
Dedup key: url.

Per-article sentiment is NOT computed here — that's the LLM router's job
in Phase 3. This module only ingests text.

Phase 24.18:
  - Added GDelt as a 3rd source (multilingual, supports historical).
  - New `ingest(run_date=None)` entry point that the daily orchestrator
    runner expects (previously it tried to call `news.ingest` and silently
    logged 'not found' because only `ingest_news_for_tickers` existed).
  - After per-source pulls, the combined frame is also written to the
    lake's `data/lake/news/{run_date}.parquet` so downstream training-set
    builds can pick it up via `lake_io.read_news_window(...)`.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from omni.storage import tiers
from omni.utils.env import get_env, sources
from omni.utils.logger import log

_NEWS_COLS = ["published_at", "source", "title", "description", "url", "ticker_hint", "lang"]


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5), reraise=True)
def _fetch_newsapi(query: str, *, lookback_days: int = 1, page_size: int = 100) -> pd.DataFrame:
    api_key = get_env("NEWSAPI_KEY")
    if not api_key:
        log.warning("NEWSAPI_KEY not set; skipping NewsAPI")
        return pd.DataFrame(columns=_NEWS_COLS)
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    r = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": query, "from": since, "language": "en", "sortBy": "publishedAt",
            "pageSize": page_size, "apiKey": api_key,
        }, timeout=30,
    )
    if r.status_code != 200:
        log.warning(f"NewsAPI {r.status_code}: {r.text[:200]}")
        return pd.DataFrame(columns=_NEWS_COLS)
    arts = r.json().get("articles", [])
    rows = [{
        "published_at": a.get("publishedAt"),
        "source": (a.get("source") or {}).get("name", "newsapi"),
        "title": a.get("title", ""),
        "description": a.get("description", ""),
        "url": a.get("url", ""),
        "ticker_hint": query,
        "lang": "en",
    } for a in arts if a.get("url")]
    return pd.DataFrame(rows, columns=_NEWS_COLS)


def _fetch_rss(feed_url: str) -> pd.DataFrame:
    """Lightweight RSS parser without feedparser dep — uses requests + xml.etree."""
    import xml.etree.ElementTree as ET
    try:
        r = requests.get(feed_url, timeout=20, headers={"User-Agent": "Omni_Stocks/0.1"})
        if r.status_code != 200:
            return pd.DataFrame(columns=_NEWS_COLS)
        root = ET.fromstring(r.content)
        # Handle both RSS 2.0 (channel/item) and Atom (entry)
        items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
        rows = []
        for it in items:
            title = (it.findtext("title") or it.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
            link_el = it.find("link")
            link = link_el.text if link_el is not None and link_el.text else \
                (it.find("{http://www.w3.org/2005/Atom}link").get("href") if it.find("{http://www.w3.org/2005/Atom}link") is not None else "")
            desc = (it.findtext("description") or it.findtext("{http://www.w3.org/2005/Atom}summary") or "").strip()
            pub = it.findtext("pubDate") or it.findtext("{http://www.w3.org/2005/Atom}published") or ""
            rows.append({
                "published_at": pub,
                "source": feed_url,
                "title": title,
                "description": desc,
                "url": link,
                "ticker_hint": "",
                "lang": "en",
            })
        return pd.DataFrame(rows, columns=_NEWS_COLS)
    except Exception as e:
        log.warning(f"RSS fetch failed for {feed_url}: {e}")
        return pd.DataFrame(columns=_NEWS_COLS)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["url", "title"])
    df = df[df["url"].astype(str).str.len() > 0]
    return df


def ingest_news_for_tickers(tickers: Iterable[str], *,
                              lookback_days: int = 1,
                              persist_lake_date: str | None = None,
                              ) -> int:
    """Pull NewsAPI per-ticker queries + global RSS feeds + GDelt per-ticker.

    Persists to the raw tier (per-source dedup) and, when
    ``persist_lake_date`` is set, also writes the combined deduplicated
    frame to ``data/lake/news/{date}.parquet`` via ``lake_io.write_news_day``.
    """
    cfg = sources()
    total = 0
    combined_parts: list[pd.DataFrame] = []
    tickers_list = list(tickers)

    # NewsAPI per-ticker
    if cfg.get("news", {}).get("newsapi", {}).get("enabled"):
        for t in tickers_list:
            df = _normalize(_fetch_newsapi(t, lookback_days=lookback_days))
            if df.empty:
                continue
            tiers.save_raw(df, source="news_newsapi", asset_class="all",
                           ticker=t, dedup_cols=["url"])
            combined_parts.append(df)
            total += len(df)
            log.info(f"newsapi {t}: {len(df)} articles")

    # RSS (global, store under shared 'global' key)
    if cfg.get("news", {}).get("rss", {}).get("enabled"):
        feeds = cfg["news"]["rss"].get("feeds", [])
        for feed in feeds:
            df = _normalize(_fetch_rss(feed))
            if df.empty:
                continue
            slug = hashlib.md5(feed.encode()).hexdigest()[:10]
            tiers.save_raw(df, source="news_rss", asset_class="all",
                           ticker=slug, dedup_cols=["url"])
            combined_parts.append(df)
            total += len(df)
            log.info(f"rss {feed}: {len(df)} items")

    # GDelt per-ticker (Phase 24.18) — multilingual, no API key, rate-limited
    # to ~5s between calls. Falls back gracefully on 429s; the gdelt module
    # already swallows transient errors and returns an empty frame.
    if cfg.get("news", {}).get("gdelt", {}).get("enabled", True):
        try:
            from omni.ingest.sources import gdelt as _gdelt
            for t in tickers_list:
                try:
                    df = _gdelt.news(t, lookback_days=max(lookback_days, 2))
                    if df is None or df.empty:
                        continue
                    df = _normalize(df.assign(ticker_hint=t))
                    if df.empty:
                        continue
                    tiers.save_raw(df, source="news_gdelt", asset_class="all",
                                   ticker=t, dedup_cols=["url"])
                    combined_parts.append(df)
                    total += len(df)
                    log.info(f"gdelt {t}: {len(df)} articles")
                except Exception as exc:  # noqa: BLE001
                    log.debug(f"gdelt {t}: {exc}")
        except ImportError:
            log.debug("gdelt source not importable; skipping")

    # Phase 24.18 — also persist the combined frame to the lake so the
    # training-set builder can read historical news via
    # `lake_io.read_news_window(start, end)`. Without this step the lake
    # `data/lake/news/` directory stays empty and sentiment features in
    # training are all zero.
    if persist_lake_date and combined_parts:
        try:
            from omni.lake import io as lake_io
            combined = pd.concat(combined_parts, ignore_index=True, sort=False)
            combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
            if not combined.empty:
                n = lake_io.write_news_day(combined, persist_lake_date)
                log.info(f"lake news write: {n} rows -> {persist_lake_date}.parquet")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"lake news write failed for {persist_lake_date}: {exc}")

    return total


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _india_universe() -> list[str]:
    """Return the curated India ticker list (lazy import to avoid cycles)."""
    try:
        from omni.lake.backfill import universe_pairs
        return [t for (t, _ac) in universe_pairs(asset_classes=("india",))]
    except Exception as exc:  # noqa: BLE001
        log.warning(f"news.ingest: cannot load universe: {exc}")
        return []


def ingest(run_date: str | None = None) -> int:
    """Daily-orchestrator entry point. Pulls news for the curated India
    universe + global RSS feeds, persists raw + lake snapshot.

    The orchestrator's `run_daily_ingest()` calls `news.ingest(run_date=...)`;
    previously this function didn't exist so the runner silently logged
    'omni.ingest.news.ingest not found'. Phase 24.18 fixes the gap.
    """
    date_iso = run_date or _today_iso()
    tickers = _india_universe()
    if not tickers:
        log.warning("news.ingest: empty ticker universe; skipping")
        return 0
    return ingest_news_for_tickers(
        tickers, lookback_days=2, persist_lake_date=date_iso,
    )
