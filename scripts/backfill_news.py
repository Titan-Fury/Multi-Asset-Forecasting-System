"""Historical news backfill — populate `data/lake/news/` for past dates.

GDelt-based (no API key, free, multilingual). Iterates a window of past
days, queries GDelt per ticker once per day, buckets returned articles by
their `published_at` date and writes one parquet per day.

Resumable: skips dates that already have a parquet on disk; persists a
small JSON checkpoint so a kill+resume picks up where it left off.

Usage:
    python scripts/backfill_news.py                       # 90 days, India only
    python scripts/backfill_news.py --days 30
    python scripts/backfill_news.py --days 180 --asset-classes india,us
    python scripts/backfill_news.py --max-tickers 10      # smoke test

Free-tier reality: GDelt asks for ~5s between requests (else 429s). With
59 India tickers that's 5 minutes per pass. We do ONE pass — meaning we
hit GDelt N_TICKERS times asking for `lookback_days=DAYS` and sort the
returned articles into per-day buckets. So total time is roughly
``N_TICKERS * 5s`` (~5 min) regardless of how many days you request.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omni.lake import io as lake_io          # noqa: E402
from omni.lake import paths as lake_paths    # noqa: E402
from omni.lake.backfill import universe_pairs  # noqa: E402
from omni.utils.logger import log              # noqa: E402

_COLS = ["published_at", "source", "title", "description", "url",
         "ticker_hint", "lang"]
_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_HEADERS = {"User-Agent": "Mozilla/5.0 Omni_Stocks/0.1 (news-backfill)"}


def _gdelt_window(ticker: str, *, start: datetime, end: datetime,
                    max_records: int = 250) -> pd.DataFrame:
    """Fetch GDelt articles for ``ticker`` in [start, end]. Returns the
    raw frame with `published_at` parsed; caller is responsible for any
    deduplication and bucket-by-date.
    """
    params = {
        "query": f"{ticker} sourcelang:eng",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": min(max_records, 250),
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
    }
    try:
        r = requests.get(_GDELT_URL, params=params, timeout=30, headers=_HEADERS)
        if r.status_code == 429:
            return pd.DataFrame(columns=_COLS).attrs.update({"rate_limited": True}) or pd.DataFrame(columns=_COLS)
        if r.status_code != 200:
            return pd.DataFrame(columns=_COLS)
        try:
            arts = r.json().get("articles", [])
        except ValueError:
            return pd.DataFrame(columns=_COLS)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"gdelt {ticker}: {exc}")
        return pd.DataFrame(columns=_COLS)

    rows = [{
        "published_at": a.get("seendate"),
        "source": a.get("domain", "gdelt"),
        "title": a.get("title", ""),
        "description": "",
        "url": a.get("url", ""),
        "ticker_hint": ticker,
        "lang": "en",
    } for a in arts if a.get("url")]
    df = pd.DataFrame(rows, columns=_COLS)
    if not df.empty:
        df["published_at"] = pd.to_datetime(
            df["published_at"], errors="coerce",
            format="%Y%m%dT%H%M%SZ", utc=True,
        )
        df = df.dropna(subset=["published_at"])
    return df


def _checkpoint_path() -> Path:
    return lake_paths.news_dir().parent / "_news_backfill.checkpoint.json"


def _load_checkpoint() -> dict:
    p = _checkpoint_path()
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _save_checkpoint(state: dict) -> None:
    p = _checkpoint_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(state, f, indent=2, default=str)


def _existing_dates() -> set[str]:
    d = lake_paths.news_dir()
    if not d.exists():
        return set()
    return {f.stem for f in d.glob("*.parquet")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Historical news backfill (GDelt)")
    parser.add_argument("--days", type=int, default=90,
                          help="Lookback window in calendar days (default 90)")
    parser.add_argument("--asset-classes", default="india",
                          help="Comma-separated asset_class filter (default india)")
    parser.add_argument("--max-tickers", type=int, default=None,
                          help="Cap ticker count for smoke testing")
    parser.add_argument("--sleep-seconds", type=float, default=5.0,
                          help="Polite delay between GDelt requests (default 5s)")
    parser.add_argument("--max-records-per-ticker", type=int, default=250,
                          help="Cap GDelt results per ticker (default 250)")
    parser.add_argument("--force", action="store_true",
                          help="Re-fetch even if a date's parquet exists")
    args = parser.parse_args()

    classes = [c.strip() for c in args.asset_classes.split(",") if c.strip()]
    pairs = universe_pairs(asset_classes=classes)
    if args.max_tickers:
        pairs = pairs[: args.max_tickers]
    tickers = [t for (t, _ac) in pairs]

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    print(f"[1/3] backfill window: {start.date()} -> {end.date()}  "
            f"({args.days}d, {len(tickers)} tickers)")
    print(f"      sleep_between={args.sleep_seconds}s  "
            f"max_records={args.max_records_per_ticker}")

    existing = _existing_dates() if not args.force else set()
    print(f"      already-on-disk dates: {len(existing)}")

    state = _load_checkpoint()
    state.setdefault("last_run", None)
    state.setdefault("tickers_done", [])
    print(f"      checkpoint: {len(state.get('tickers_done', []))} tickers already pulled")

    print("[2/3] fetching GDelt per-ticker window...")
    all_articles: list[pd.DataFrame] = []
    rate_limit_streak = 0
    for i, t in enumerate(tickers, 1):
        if t in state.get("tickers_done", []) and not args.force:
            print(f"      [{i:>3}/{len(tickers)}] {t:<12} (checkpoint skip)")
            continue
        df = _gdelt_window(
            t, start=start, end=end,
            max_records=args.max_records_per_ticker,
        )
        n = len(df)
        if n == 0:
            rate_limit_streak += 1
            print(f"      [{i:>3}/{len(tickers)}] {t:<12} 0 articles  (streak={rate_limit_streak})")
        else:
            rate_limit_streak = 0
            all_articles.append(df)
            print(f"      [{i:>3}/{len(tickers)}] {t:<12} {n:>4} articles  "
                    f"({df['published_at'].min().date()} -> {df['published_at'].max().date()})")
        state.setdefault("tickers_done", []).append(t)
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        _save_checkpoint(state)
        if rate_limit_streak >= 5:
            print(f"  WARNING: 5 consecutive empty results — possibly rate-limited. "
                    f"Sleeping 60s and continuing.")
            time.sleep(60)
            rate_limit_streak = 0
            continue
        time.sleep(args.sleep_seconds)

    if not all_articles:
        print("[3/3] no articles fetched. Lake state unchanged.")
        return 1

    combined = pd.concat(all_articles, ignore_index=True, sort=False)
    combined = combined.dropna(subset=["url", "published_at"])
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    combined["date_bucket"] = combined["published_at"].dt.tz_convert("UTC").dt.date.astype(str)
    print(f"[3/3] writing per-day parquet files...")
    print(f"      total articles: {len(combined):,}  unique URLs after dedup")
    print(f"      unique dates:   {combined['date_bucket'].nunique()}")

    written = 0
    for date_iso, group in combined.groupby("date_bucket"):
        if date_iso in existing and not args.force:
            continue
        out = group.drop(columns=["date_bucket"]).reset_index(drop=True)
        n = lake_io.write_news_day(out, str(date_iso))
        written += n
    print(f"      wrote {written:,} rows across {combined['date_bucket'].nunique()} day-files")

    state["tickers_done"] = []  # reset for next run
    _save_checkpoint(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
