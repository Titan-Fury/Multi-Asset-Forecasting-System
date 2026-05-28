"""On-demand multi-source data lookup with provenance.

Single entry point for any ticker request. Pulls prices, fundamentals,
news, macro and regime from whatever live sources are available, with
graceful per-source degradation. Each piece is cached at its TTL tier.

Returns a LookupBundle that downstream `predict.py` and `deepanalysis.py`
consume directly. Never writes to the storage tier.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from omni.ingest.fundamentals import fetch_fundamentals as _yf_fundamentals
from omni.ingest.prices import fetch_prices as _yf_prices
from omni.ingest.sources import (
    currents as _currents,
    edgar as _edgar,
    fmp as _fmp,
    gdelt as _gdelt,
    marketaux as _marketaux,
    newsapi_live as _newsapi,
    polygon as _polygon,
    reddit_public as _reddit,
    stocktwits as _stocktwits,
)
from omni.llm.backends.base import BackendError
from omni.llm.router import route as _llm_route
from omni.ondemand.cache import TTL, cached
from omni.ondemand.macro_live import compute_live as _macro_live
from omni.ondemand.quality import Provenance
from omni.ondemand.regime_live import RegimeOutput, fit_or_load as _regime_load
from omni.utils.logger import log

# Phase 24.21 — negative cache for tickers we already proved don't resolve
# anywhere. Keyed by (asset_class, ticker.upper()), value = monotonic-clock
# expiry. We'd otherwise re-pay the full prices+fundamentals+news fan-out on
# every dashboard click for a typo'd ticker, AND log "no prices for ..." each
# time. 24h TTL is short enough that genuinely new listings still get picked
# up the next day, long enough to suppress the warning storm we see in logs.
import time as _time

_NEG_TTL_SECONDS = 24 * 60 * 60
_NEG_MAX_SIZE = 10_000  # cap to prevent unbounded growth from typo storms
_negative_cache: dict[tuple[str, str], float] = {}


def _neg_cache_hit(asset_class: str, ticker: str) -> bool:
    key = (asset_class, ticker.upper())
    expiry = _negative_cache.get(key)
    if expiry is None:
        return False
    if _time.monotonic() > expiry:
        _negative_cache.pop(key, None)
        return False
    return True


def _neg_cache_put(asset_class: str, ticker: str) -> None:
    # Periodic eviction of expired entries before insert so a burst of
    # invalid tickers from a fuzzy-search bot can't grow this dict
    # without bound for the lifetime of the Streamlit process.
    if len(_negative_cache) >= _NEG_MAX_SIZE:
        now = _time.monotonic()
        expired = [k for k, exp in _negative_cache.items() if exp <= now]
        for k in expired:
            _negative_cache.pop(k, None)
        # If nothing was expired, drop the oldest 25% by expiry.
        if len(_negative_cache) >= _NEG_MAX_SIZE:
            for k, _ in sorted(_negative_cache.items(), key=lambda kv: kv[1])[: _NEG_MAX_SIZE // 4]:
                _negative_cache.pop(k, None)
    key = (asset_class, ticker.upper())
    _negative_cache[key] = _time.monotonic() + _NEG_TTL_SECONDS


@dataclass
class LookupBundle:
    ticker: str
    asset_class: str
    display_name: str
    asof: pd.Timestamp
    prices: pd.DataFrame              # OHLCV
    fundamentals: pd.DataFrame        # 0 or 1 row (run_date snapshot)
    news_raw: pd.DataFrame            # union of news + stocktwits
    news_sentiment: pd.DataFrame      # daily aggregate
    macro: pd.DataFrame
    regime: RegimeOutput | None
    provenance: Provenance


# ---------- prices: yfinance + polygon cross-check ----------

@cached(ttl=TTL.prices, key="prices_live:{asset_class}:{ticker}")
def _prices_combined(asset_class: str, ticker: str) -> tuple[pd.DataFrame, list[str], float | None]:
    """Fetch prices from yfinance (primary) and polygon (cross-check).

    Returns (df, sources_used, agreement_pct).
    """
    end = date.today()
    start = end - timedelta(days=int(5 * 365.25))
    sources: list[str] = []
    yf_df = pd.DataFrame()
    try:
        yf_df = _yf_prices(asset_class, ticker, start=start.isoformat(), end=end.isoformat())
        if not yf_df.empty:
            sources.append("yfinance")
    except Exception as e:
        log.warning(f"yfinance prices failed {ticker}: {e}")

    poly_df = _polygon.eod_prices(asset_class, ticker, lookback_days=730)
    if not poly_df.empty:
        sources.append("polygon")

    agreement: float | None = None
    if not yf_df.empty and not poly_df.empty:
        merged = yf_df[["date", "adj_close"]].merge(
            poly_df[["date", "adj_close"]].rename(columns={"adj_close": "poly_close"}),
            on="date", how="inner",
        ).tail(30)
        if not merged.empty:
            diffs = (merged["adj_close"] - merged["poly_close"]).abs() / merged["adj_close"]
            agreement = float((diffs < 0.005).mean())

    base = yf_df if not yf_df.empty else poly_df
    return base, sources, agreement


# ---------- fundamentals: yfinance + fmp merge ----------

@cached(ttl=TTL.fundamentals, key="fund_live:{asset_class}:{ticker}")
def _fundamentals_combined(asset_class: str, ticker: str) -> tuple[pd.DataFrame, list[str]]:
    if asset_class in ("crypto", "fx"):
        return pd.DataFrame(), []
    sources: list[str] = []
    yf = _yf_fundamentals(asset_class, ticker)
    if not yf.empty:
        sources.append("yfinance")
    fmp = _fmp.fundamentals(ticker, asset_class)
    if not fmp.empty:
        sources.append("fmp")
    if yf.empty and fmp.empty:
        # Phase 25 / Item 3: no live data — fall back to the most recent
        # PIT snapshot (replay-safe; uses what we previously persisted).
        try:
            from omni.storage.fundamentals_snapshots import load_pit
            pit = load_pit(asset_class, ticker, asof=None)
            if not pit.empty:
                pit = pit.drop(columns=[c for c in ("snapshot_date",)
                                         if c in pit.columns])
                return pit, ["pit_snapshot"]
        except Exception:  # noqa: BLE001
            pass
        return pd.DataFrame(), []
    if yf.empty:
        merged_df = fmp
    elif fmp.empty:
        merged_df = yf
    else:
        # Merge: FMP wins on conflicts where it has a non-null value.
        merged = yf.iloc[0].to_dict()
        for k, v in fmp.iloc[0].items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                merged[k] = v
        merged["source"] = "yfinance+fmp"
        merged_df = pd.DataFrame([merged])

    # Phase 25 / Item 3: persist this fetch as a dated PIT snapshot so
    # replays can reproduce point-in-time fundamentals instead of
    # silently using whatever yfinance returns today. Best-effort —
    # write failures don't block the live path.
    try:
        from omni.storage.fundamentals_snapshots import snapshot_save
        snapshot_save(asset_class, ticker, merged_df)
    except Exception as exc:  # noqa: BLE001
        log.debug(f"fundamentals_pit snapshot skipped for {ticker}: {exc}")

    return merged_df, sources


# ---------- news: union + dedup + capped sentiment ----------

@cached(ttl=TTL.news, key="news_live_v3:{asset_class}:{ticker}:{display_name}")
def _news_combined(asset_class: str, ticker: str, display_name: str
                   ) -> tuple[pd.DataFrame, list[str], int]:
    # Phase 28.4 — parallel news-source fan-out. Previously the 10+
    # source calls (NewsAPI, Currents, Marketaux, GDELT, StockTwits,
    # EDGAR, Reddit, Indian RSS, BSE notices, Telegram, SEBI-PIT) ran
    # serially over 5-15 seconds total. Each call is a network round
    # trip that doesn't depend on the others, so we now fan them out
    # via ThreadPoolExecutor. Expected wall-clock saving: 3-5x on
    # cold-cache predictions where every source has to be hit.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sources: list[str] = []
    parts: list[pd.DataFrame] = []

    def _try(name: str, fn):
        """Wrap a source fetch so an exception in one source doesn't
        derail the others. Returns (name, df_or_None)."""
        try:
            df = fn()
            return (name, df)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"{name} fetch failed {ticker}: {exc}")
            return (name, None)

    # Build the per-source task list. Each entry is (name, zero-arg
    # callable that returns either a DataFrame or None). India-specific
    # sources are gated on asset_class so we don't waste threads.
    tasks: list[tuple[str, callable]] = [
        ("newsapi",   lambda: _newsapi.news(ticker, display_name)),
        ("currents",  lambda: _currents.news(ticker, display_name=display_name)),
        ("marketaux", lambda: _marketaux.news(ticker, asset_class=asset_class)),
        ("gdelt",     lambda: _gdelt.news(ticker, display_name=display_name)),
        ("stocktwits",lambda: _stocktwits.messages(ticker, asset_class=asset_class)),
        ("reddit",    lambda: _reddit.posts(ticker, display_name=display_name,
                                              asset_class=asset_class)),
    ]
    if asset_class == "us":
        tasks.append(("edgar", lambda: _edgar.recent_filings(ticker, days=180)))
    if asset_class == "india":
        from omni.ingest.sources import (
            indian_rss as _ind_rss,
            bse_notices as _bse,
            telegram_channels as _tg,
            sebi_insider as _pit,
        )
        tasks.append(("indian_rss",
                       lambda: _ind_rss.fetch_indian_news(
                           ticker, display_name=display_name)))
        tasks.append(("bse_notices",
                       lambda: _bse.matching_notices(
                           ticker, display_name=display_name)))
        tasks.append(("telegram",
                       lambda: _tg.fetch_telegram_news(
                           ticker, display_name=display_name)))
        # SEBI PIT — needs ticker filter post-fetch; do that in the
        # post-processing block below.
        tasks.append(("sebi_pit_raw",
                       lambda: _pit.insider_disclosures(days=30)))

    # Execute all source calls in parallel with a small thread pool.
    # 8 workers comfortably handles the ~10 sources without thrashing
    # the network or breaching free-tier rate limits (each individual
    # source is quota-policed by its own client).
    results: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_try, name, fn): name for name, fn in tasks}
        for fut in as_completed(futs):
            name, df = fut.result()
            results[name] = df

    # Reassemble in the legacy stable order so downstream blending /
    # source-attribution stays deterministic across runs.
    def _push(name: str, df: pd.DataFrame | None,
              *, assign_nan: bool = False,
              cols_keep: list[str] | None = None) -> None:
        if df is None or df.empty:
            return
        out = df.assign(sent_score=np.nan) if assign_nan else df.copy()
        if cols_keep:
            out = out[[c for c in cols_keep if c in out.columns]].copy()
        sources.append(name)
        parts.append(out)

    _push("newsapi", results.get("newsapi"), assign_nan=True)
    _push("currents", results.get("currents"), assign_nan=True)
    _push("marketaux", results.get("marketaux"))
    _push("gdelt", results.get("gdelt"), assign_nan=True)
    _push("stocktwits", results.get("stocktwits"))

    if asset_class == "us":
        cols_keep = ["title", "url", "source", "published_at", "sent_score"]
        _push("edgar", results.get("edgar"), cols_keep=cols_keep)

    cols_keep = ["title", "url", "source", "published_at", "sent_score"]
    _push("reddit", results.get("reddit"), cols_keep=cols_keep)

    if asset_class == "india":
        _push("indian_rss", results.get("indian_rss"), cols_keep=cols_keep)
        _push("bse_notices", results.get("bse_notices"), cols_keep=cols_keep)
        _push("telegram", results.get("telegram"), cols_keep=cols_keep)

        # SEBI PIT post-processing (filter to ticker, re-format rows).
        pit_df = results.get("sebi_pit_raw")
        try:
            if pit_df is not None and not pit_df.empty and "ticker" in pit_df.columns:
                sub = pit_df[pit_df["ticker"].astype(str).str.upper() == ticker.upper()]
                if not sub.empty:
                    rows_pit = []
                    for _, r in sub.iterrows():
                        person = str(r.get("person", "Insider"))[:40]
                        ttype = str(r.get("transaction_type", "?"))[:20]
                        qty = r.get("qty", "")
                        val = r.get("value_lakhs", "")
                        title = f"PIT: {person} {ttype} qty={qty} value={val}L"
                        rows_pit.append({
                            "title": title,
                            "url": "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading",
                            "source": "sebi_pit",
                            "published_at": str(pd.Timestamp(r.get("date")).isoformat()),
                            "sent_score": (0.4 if "buy" in ttype.lower() or "acqui" in ttype.lower()
                                            else -0.4 if "sell" in ttype.lower() or "dispos" in ttype.lower()
                                            else 0.0),
                        })
                    if rows_pit:
                        sources.append("sebi_pit"); parts.append(pd.DataFrame(rows_pit))
        except Exception as e:  # noqa: BLE001
            log.warning(f"sebi_pit post-processing failed {ticker}: {e}")

    if not parts:
        return pd.DataFrame(), [], 0

    df = pd.concat(parts, ignore_index=True, sort=False)
    df = df.dropna(subset=["title"]).copy()
    df = df[df["title"].astype(str).str.len() > 5]
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at"]).sort_values("published_at", ascending=False)
    return df, sources, len(df)


def _score_sentiments(news: pd.DataFrame, *, top_k: int = 25) -> pd.DataFrame:
    """Score the newest top_k articles via cheap LLM bulk classifier.

    Marketaux and StockTwits already provide sent_score; we only call the
    LLM for rows where sent_score is NaN.
    """
    if news.empty:
        return news
    out = news.copy()
    if "sent_score" not in out.columns:
        out["sent_score"] = np.nan
    head = out.head(top_k).copy()
    needs = head[head["sent_score"].isna()].copy()
    if needs.empty:
        return out

    scores: list[float] = []
    for _, r in needs.iterrows():
        text = f"{r.get('title','')}. {r.get('description','') or ''}".strip()[:400]
        if not text:
            scores.append(0.0); continue
        try:
            res = _llm_route("bulk_classification",
                              f"Classify this finance headline as bullish (+1), neutral (0), or bearish (-1) for the stock. Reply with only the number.\n\nHeadline: {text}")
            txt = (res.text if hasattr(res, "text") else str(res)).strip().lower()
            # Best-effort parse
            if "+1" in txt or "bullish" in txt or "positive" in txt:
                scores.append(1.0)
            elif "-1" in txt or "bearish" in txt or "negative" in txt:
                scores.append(-1.0)
            else:
                scores.append(0.0)
        except (BackendError, Exception) as e:
            log.debug(f"sentiment LLM skip: {e}")
            scores.append(0.0)
    out.loc[needs.index, "sent_score"] = scores
    out["sent_score"] = out["sent_score"].fillna(0.0)
    return out


def _aggregate_sentiment(news: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if news.empty or "sent_score" not in news.columns:
        return pd.DataFrame(columns=["date", "ticker", "n_articles", "sent_mean",
                                       "sent_std", "sent_pos_share", "sent_neg_share",
                                       "sent_momentum_7d"])
    df = news.copy()
    df["date"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date"])
    grp = df.groupby("date")["sent_score"]
    agg = pd.DataFrame({
        "n_articles": grp.size(),
        "sent_mean": grp.mean(),
        "sent_std": grp.std(ddof=0),
        "sent_pos_share": grp.apply(lambda s: (s > 0.2).mean()),
        "sent_neg_share": grp.apply(lambda s: (s < -0.2).mean()),
    }).reset_index()
    agg["ticker"] = ticker
    agg = agg.sort_values("date").reset_index(drop=True)
    agg["sent_momentum_7d"] = (
        agg["sent_mean"] - agg["sent_mean"].shift(1).rolling(7, min_periods=2).mean()
    )
    return agg


# ---------- public ----------

def lookup_ticker(ticker: str, asset_class: str,
                  *, display_name: str | None = None,
                  force_fresh: bool = False) -> LookupBundle | None:
    """End-to-end fetch for one ticker. Returns None if no prices available.

    Phase 24.22 — when ``force_fresh=True``, every cached entry that
    mentions this ticker (prices, fundamentals, news, prediction card,
    deep card) is purged BEFORE the network fan-out. Use this for
    "predict tomorrow" runs after market close so the cache TTLs (4h
    prices, 30min news) don't serve stale snapshots from earlier today.
    """
    display_name = display_name or ticker
    prov = Provenance()
    asof = pd.Timestamp.now().normalize()

    # Phase 24.22 — force-fresh path: drop ALL cache entries that mention
    # this ticker so the live fetch path is guaranteed to re-pull
    # prices/news/fundamentals from the source rather than serving the
    # last cached frame.
    if force_fresh:
        try:
            from omni.ondemand.cache import purge_ticker as _purge
            n = _purge(ticker)
            log.info(f"force_fresh[{ticker}]: purged {n} cache entries")
            # Also drop the negative-cache entry so a name that previously
            # failed gets a real second look.
            _negative_cache.pop((asset_class, ticker.upper()), None)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"force_fresh purge failed for {ticker}: {exc}")

    # Phase 24.21 — short-circuit for tickers we've already proven invalid in
    # the last 24h. Avoids both the wasted network fan-out and the repeating
    # "lookup: no prices" warning that was showing up ~36x in logs for the
    # same handful of typos / delisted symbols.
    if _neg_cache_hit(asset_class, ticker):
        log.debug(f"lookup: negative-cache hit for {asset_class}/{ticker}")
        return None

    # PRICES (mandatory)
    prices, p_sources, agreement = _prices_combined(asset_class, ticker)
    prov.prices_sources = p_sources
    prov.prices_agreement_pct = agreement
    if prices is None or prices.empty:
        prov.add_warning("no prices from any source")
        log.warning(f"lookup: no prices for {asset_class}/{ticker}")
        _neg_cache_put(asset_class, ticker)
        return None

    # FUNDAMENTALS (optional)
    fund, f_sources = _fundamentals_combined(asset_class, ticker)
    prov.fundamentals_sources = f_sources
    if not f_sources and asset_class not in ("crypto", "fx"):
        prov.add_warning("no fundamentals (yfinance/fmp empty)")

    # NEWS (optional)
    news_raw, n_sources, n_count = _news_combined(asset_class, ticker, display_name)
    prov.news_sources = n_sources
    prov.news_count = n_count
    news_scored = _score_sentiments(news_raw, top_k=25)
    news_agg = _aggregate_sentiment(news_scored, ticker)

    # MACRO (shared cache, 24h)
    try:
        macro = _macro_live(lookback_years=8)
        prov.macro_fresh = not macro.empty
        if macro.empty:
            prov.add_warning("macro empty (no FRED/yf access)")
    except Exception as e:
        log.warning(f"macro_live failed: {e}")
        macro = pd.DataFrame()
        prov.add_warning(f"macro error: {e}")

    # REGIME (shared cache, 24h)
    try:
        rres = _regime_load()
        if rres is None:
            regime_out = None
            prov.add_warning("regime not fitted (insufficient data)")
        else:
            _, regime_out = rres
            prov.regime_fresh = True
    except Exception as e:
        log.warning(f"regime_live failed: {e}")
        regime_out = None
        prov.add_warning(f"regime error: {e}")

    # ---- Phase 20: microstructure inputs (India only) ----------------------
    # Plumb bhavcopy / FII-DII / intraday / OI snapshot into the bundle so
    # `omni.features.microstructure.compute_microstructure` actually has data.
    # We stash them on `news_agg.attrs` because LookupBundle has fixed fields
    # and `features_inline.py` already reads from `sentiment.attrs`.
    if asset_class == "india":
        try:
            from omni.ingest.sources.nse_bhavcopy import fetch_range as _bhav_range
            end = pd.Timestamp.now().normalize()
            start = end - pd.tseries.offsets.BDay(60)
            bhav = _bhav_range(start, end, tickers_nse=[ticker.replace(".NS", "").replace(".BO", "").upper()])
            if not bhav.empty:
                news_agg.attrs["bhavcopy"] = bhav
                prov.add_warning(f"bhavcopy rows: {len(bhav)}") if len(bhav) < 5 else None
        except Exception as e:  # noqa: BLE001
            log.debug(f"lookup: bhavcopy fetch failed: {e}")

        try:
            from omni.ingest.sources.nse_flows import fii_dii_activity
            flows = fii_dii_activity()
            if not flows.empty:
                news_agg.attrs["fii_dii"] = flows
        except Exception as e:  # noqa: BLE001
            log.debug(f"lookup: fii_dii fetch failed: {e}")

        try:
            from omni.ondemand import oi_archive  # see new module below
            oi_hist = oi_archive.snapshot_and_history(ticker)
            if not oi_hist.empty:
                news_agg.attrs["oi_history"] = oi_hist
        except Exception as e:  # noqa: BLE001
            log.debug(f"lookup: oi_history fetch failed: {e}")

        # Intraday 5-min bars (best-effort; on-demand cache)
        try:
            from omni.ingest.intraday import fetch_intraday_bars
            ib = fetch_intraday_bars(ticker, asset_class, interval="5m")
            if ib is not None and not ib.empty:
                news_agg.attrs["intraday"] = ib
        except Exception as e:  # noqa: BLE001
            log.debug(f"lookup: intraday fetch failed: {e}")

    return LookupBundle(
        ticker=ticker, asset_class=asset_class, display_name=display_name,
        asof=asof, prices=prices, fundamentals=fund,
        news_raw=news_scored, news_sentiment=news_agg,
        macro=macro, regime=regime_out, provenance=prov,
    )
