"""Build the pooled training set from the data lake.

Reads per-ticker prices from data/lake/, the latest fundamentals snapshot,
and writes a unified training parquet with:
- all features the on-demand path produces
- forward targets for horizons 1d / 5d / 21d
- ticker, asset_class, date metadata so the trainer can split by date

Output: ``data/lake/training/training_set.parquet``

Usage:
  python scripts/build_training_set.py
  python scripts/build_training_set.py --asset-classes us,india --limit 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from omni.lake import io as lake_io  # noqa: E402
from omni.lake import paths as lake_paths  # noqa: E402
from omni.lake.quarterly import (
    pivot_for_join,
    read_all_quarterly,
)  # noqa: E402
from omni.ondemand.features_inline import build_features_inline  # noqa: E402
from omni.utils.logger import log  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pooled training set from lake")
    p.add_argument("--asset-classes", default="us,india,crypto",
                    help="comma-separated asset classes to include")
    p.add_argument("--limit", type=int, default=None,
                    help="cap tickers per asset class (smoke testing)")
    p.add_argument("--horizons", default="1,5,21",
                    help="comma-separated forward horizons in trading days")
    p.add_argument("--output", default=None,
                    help="override output path (default: data/lake/training/training_set.parquet)")
    p.add_argument("--embed-since", default="2024-01-01",
                    help="only attach news headlines for dates >= this ISO date "
                         "(controls Nemotron embedding API call volume; "
                         "older dates get zero-embeddings, same as before).")
    p.add_argument("--no-news", action="store_true",
                    help="skip news plumbing entirely (emb_* will be all zeros)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    classes = [c.strip() for c in args.asset_classes.split(",") if c.strip()]
    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    out_path = (Path(args.output) if args.output
                else lake_paths.lake_root() / "training" / "training_set.parquet")

    available = lake_paths.list_known_price_tickers()
    pairs = [(t, ac) for (ac, t) in available if ac in classes]
    if args.limit:
        # cap per asset class
        per: dict[str, int] = {ac: 0 for ac in classes}
        capped = []
        for t, ac in pairs:
            if per[ac] < args.limit:
                capped.append((t, ac))
                per[ac] += 1
        pairs = capped

    log.info(f"build_training_set: {len(pairs)} tickers, horizons={horizons}, "
              f"out={out_path}")

    # Latest fundamentals snapshot — applied to all rows of each ticker.
    fund_all = lake_io.read_fundamentals_asof(pd.Timestamp.now(tz="UTC").date().isoformat())
    if fund_all is None or fund_all.empty:
        log.warning("no fundamentals snapshot in lake; training will use price-only features")
        fund_all = pd.DataFrame()

    # Quarterly fundamentals (long → wide for merge_asof)
    quarterly_wide = pd.DataFrame()
    quarterly_long = read_all_quarterly()
    if not quarterly_long.empty:
        quarterly_wide = pivot_for_join(quarterly_long)
        log.info(f"quarterly fundamentals: {quarterly_wide.shape} "
                  f"({quarterly_wide['ticker'].nunique()} tickers, "
                  f"{quarterly_wide['asof_date'].nunique()} unique quarter-ends)")
    else:
        log.warning("no quarterly fundamentals in lake; running without (run "
                     "scripts/backfill_quarterly.py to enable)")

    # ------------------------------------------------------------------
    # Phase 23 — load all news ONCE so embeddings actually populate during
    # build_features_inline. Previously sentiment=None made emb_* all-zero.
    # We cap the embed window via --embed-since to keep Nemotron calls finite.
    # ------------------------------------------------------------------
    news_all = pd.DataFrame()
    if not args.no_news:
        try:
            today_iso = pd.Timestamp.now(tz="UTC").date().isoformat()
            news_all = lake_io.read_news_window(args.embed_since, today_iso)
            log.info(f"news loaded for embedding: {len(news_all):,} headlines "
                     f"({args.embed_since} -> {today_iso})")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"news load failed: {exc}; emb_* will be zeros")
            news_all = pd.DataFrame()

    # Phase 24.18 — also build the AGGREGATED sentiment frame here so the
    # per-ticker training frames get sent_mean/n_articles/panic_kw_count/etc.
    # populated. Without this step `build_features_inline` sees an empty
    # sentiment frame and the merge is skipped, leaving every sentiment
    # column NaN/0 in training (Sentiment Health Score 0/10).
    sent_agg_all = pd.DataFrame()
    if not news_all.empty:
        try:
            from omni.features.sentiment import aggregate, score_articles
            log.info(f"scoring {len(news_all):,} articles via FinBERT...")
            scored = score_articles(news_all)
            sent_agg_all = aggregate(scored, ticker_col="ticker_hint")
            log.info(
                f"sentiment aggregated: {len(sent_agg_all):,} (ticker, date) rows; "
                f"{(sent_agg_all['n_articles'] > 0).sum() if 'n_articles' in sent_agg_all.columns else 0} "
                f"with non-zero coverage"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(f"sentiment scoring failed: {exc}; columns will be zero")
            sent_agg_all = pd.DataFrame()

    parts: list[pd.DataFrame] = []
    t0 = time.time()
    for i, (ticker, ac) in enumerate(pairs, 1):
        prices = lake_io.read_prices(ticker, ac)
        if prices is None or prices.empty or len(prices) < 252:
            continue
        ticker_fund = (fund_all[fund_all.get("ticker", "") == ticker]
                        if not fund_all.empty else pd.DataFrame())

        # Phase 23 — per-ticker news proxy. build_features_inline reads
        # sentiment.attrs["news_raw"]; an empty df with attrs is enough.
        ticker_sent = pd.DataFrame()
        if not news_all.empty:
            sym = ticker.upper().replace(".NS", "").replace(".BO", "")
            if "ticker_hint" in news_all.columns:
                m = news_all["ticker_hint"].astype(str).str.upper().str.contains(sym, na=False)
                ticker_news = news_all[m] if m.any() else pd.DataFrame()
            else:
                # No ticker hint column — pass full news; embed_daily_news
                # will day-filter and dedupe via cache.
                ticker_news = news_all
            # Phase 24.18 — start with the AGGREGATED sentiment frame so the
            # downstream merge in features_inline actually populates
            # sent_mean / n_articles / panic_kw_count / promoter_kw_count
            # / sebi_kw_count for this ticker.
            if not sent_agg_all.empty and "ticker" in sent_agg_all.columns:
                ticker_sent = sent_agg_all[
                    sent_agg_all["ticker"].astype(str).str.upper().str.contains(sym, na=False)
                ].copy()
            if not ticker_news.empty:
                ticker_sent.attrs["news_raw"] = ticker_news

        feats = build_features_inline(
            ac, ticker, prices,
            fundamentals=ticker_fund if not ticker_fund.empty else None,
            sentiment=ticker_sent,
            macro=None,        # macro joined at training time separately
            horizons=horizons,
        )
        if feats is None or feats.empty:
            continue

        # Merge quarterly fundamentals via merge_asof so each price row gets
        # the most recent quarter's metrics that were known at that date.
        #
        # Phase 28.5 — `build_features_inline` now also emits q_* + asof_date
        # via its own LIVE yfinance fetch (Tier A2, meant for the inference
        # path where there's no lake). At training time the lake-backed
        # `quarterly_wide` is the canonical, deterministic source, so when it
        # is present we DROP the inline-produced q_*/asof_date first to avoid
        # a merge_asof column collision (q_total_assets_x / _y). When the lake
        # has no quarterly data we keep the inline columns as the only source.
        if not quarterly_wide.empty:
            inline_q = [c for c in feats.columns
                        if c.startswith("q_") or c == "asof_date"]
            if inline_q:
                feats = feats.drop(columns=inline_q)
            qsub = quarterly_wide[quarterly_wide["ticker"] == ticker.upper()].copy()
            if not qsub.empty:
                qsub = qsub.sort_values("asof_date")
                qsub["asof_date"] = (pd.to_datetime(qsub["asof_date"], errors="coerce")
                                       .dt.tz_localize(None).astype("datetime64[ns]"))
                feats["date"] = (pd.to_datetime(feats["date"], errors="coerce")
                                   .dt.tz_localize(None).astype("datetime64[ns]"))
                feats = feats.sort_values("date")
                feats = pd.merge_asof(
                    feats, qsub.drop(columns=["ticker", "asset_class"],
                                       errors="ignore"),
                    left_on="date", right_on="asof_date",
                    direction="backward",
                )

        parts.append(feats)
        if i % 25 == 0:
            log.info(f"  [{i}/{len(pairs)}] {ac}/{ticker}: rows so far = "
                      f"{sum(len(p) for p in parts):,}")

    if not parts:
        log.error("no training rows produced")
        return 1
    df = pd.concat(parts, ignore_index=True, sort=False)

    # ---- Step-2 fix: compute cross-sectional ranks AFTER pooling ----
    # When done per-ticker in features_inline, ranks are always 1.0 (single-row
    # group). Computing here on the pooled frame gives proper percentile rank
    # within (date, asset_class) — the actual signal we want.
    try:
        from omni.features.cross_sectional import add_cross_sectional_ranks
        before = set(df.columns)
        df = add_cross_sectional_ranks(df)
        added = sorted(set(df.columns) - before)
        if added:
            log.info(f"cross-sectional ranks added (post-pool): {len(added)} cols "
                      f"e.g. {added[:5]}")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"post-pool cross-sectional ranks failed: {exc}")

    # Drop duplicate (date, ticker) rows. A few tickers carry overlapping
    # vendor history segments (e.g. a re-listed symbol or a re-fetched
    # window), so the per-ticker frames can emit the same (date, ticker)
    # more than once. Left unchecked these duplicates silently overweight
    # those rows in training AND fan out any downstream (date, ticker)
    # join. Keep the last occurrence (most-recently-built segment wins).
    if {"date", "ticker"}.issubset(df.columns):
        n_before = len(df)
        df = df.drop_duplicates(["date", "ticker"], keep="last").reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            log.info(f"dropped {n_dropped:,} duplicate (date, ticker) rows "
                      f"({n_before:,} -> {len(df):,})")

    # Persist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    log.info(f"training set written: {len(df):,} rows × {len(df.columns)} cols  "
              f"({elapsed:.1f}s) -> {out_path}")
    print(f"\nrows:       {len(df):,}")
    print(f"cols:       {len(df.columns)}")
    print(f"tickers:    {df['ticker'].nunique() if 'ticker' in df.columns else 0}")
    print(f"date range: {df['date'].min().date() if 'date' in df.columns else '?'} -> "
          f"{df['date'].max().date() if 'date' in df.columns else '?'}")
    print(f"output:     {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
