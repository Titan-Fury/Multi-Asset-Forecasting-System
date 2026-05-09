"""On-demand multi-source data lookup with provenance.

Single entry point for any ticker request. Pulls prices, fundamentals,
news, macro and regime from whatever live sources are available, with
graceful per-source degradation. Each piece is cached at its TTL tier.

Returns a LookupBundle that downstream `predict.py` and `deepanalysis.py`
consume directly. Never writes to the storage tier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from omni.ingest.fundamentals import fetch_fundamentals as _yf_fundamentals
from omni.ingest.prices import _yf_symbol, fetch_prices as _yf_prices
from omni.ingest.sources import (
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
        return pd.DataFrame(), []
    if yf.empty:
        return fmp, sources
    if fmp.empty:
        return yf, sources
    # Merge: FMP wins on conflicts where it has a non-null value.
    merged = yf.iloc[0].to_dict()
    for k, v in fmp.iloc[0].items():
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            merged[k] = v
    merged["source"] = "yfinance+fmp"
    return pd.DataFrame([merged]), sources


# ---------- news: union + dedup + capped sentiment ----------

@cached(ttl=TTL.news, key="news_live_v3:{asset_class}:{ticker}:{display_name}")
def _news_combined(asset_class: str, ticker: str, display_name: str
                   ) -> tuple[pd.DataFrame, list[str], int]:
    sources: list[str] = []
    parts: list[pd.DataFrame] = []

    nws = _newsapi.news(ticker, display_name)
    if not nws.empty:
        sources.append("newsapi"); parts.append(nws.assign(sent_score=np.nan))
    mar = _marketaux.news(ticker, asset_class=asset_class)
    if not mar.empty:
        sources.append("marketaux"); parts.append(mar)
    gd = _gdelt.news(ticker, display_name=display_name)
    if not gd.empty:
        sources.append("gdelt"); parts.append(gd.assign(sent_score=np.nan))
    st = _stocktwits.messages(ticker, asset_class=asset_class)
    if not st.empty:
        sources.append("stocktwits"); parts.append(st)

    # SEC EDGAR — Form 4 (insider trades) + 8-K (material events). US only.
    if asset_class == "us":
        try:
            ed = _edgar.recent_filings(ticker, days=180)
            if not ed.empty:
                # Keep only the columns shared with the news union schema
                cols_keep = ["title", "url", "source", "published_at", "sent_score"]
                # description / form fields could be useful later; keep title-only for now
                ed_slim = ed[[c for c in cols_keep if c in ed.columns]].copy()
                sources.append("edgar"); parts.append(ed_slim)
        except Exception as e:
            log.warning(f"edgar fetch failed {ticker}: {e}")

    # Reddit (public no-auth JSON) — retail crowd sentiment.
    try:
        rd = _reddit.posts(ticker, display_name=display_name, asset_class=asset_class)
        if rd is not None and not rd.empty:
            cols_keep = ["title", "url", "source", "published_at", "sent_score"]
            rd_slim = rd[[c for c in cols_keep if c in rd.columns]].copy()
            sources.append("reddit"); parts.append(rd_slim)
    except Exception as e:
        log.warning(f"reddit fetch failed {ticker}: {e}")

    # Indian-press RSS (MoneyControl, ET, LiveMint, Hindu BusinessLine).
    # India-only — much higher quality than international sources for NSE/BSE names.
    if asset_class == "india":
        try:
            from omni.ingest.sources import indian_rss as _ind_rss
            ir = _ind_rss.fetch_indian_news(ticker, display_name=display_name)
            if ir is not None and not ir.empty:
                cols_keep = ["title", "url", "source", "published_at", "sent_score"]
                ir_slim = ir[[c for c in cols_keep if c in ir.columns]].copy()
                sources.append("indian_rss"); parts.append(ir_slim)
        except Exception as e:
            log.warning(f"indian_rss fetch failed {ticker}: {e}")

        # BSE Corporate Announcements (results, insider disclosures, board meetings)
        try:
            from omni.ingest.sources import bse_notices as _bse
            bn = _bse.matching_notices(ticker, display_name=display_name)
            if bn is not None and not bn.empty:
                cols_keep = ["title", "url", "source", "published_at", "sent_score"]
                bn_slim = bn[[c for c in cols_keep if c in bn.columns]].copy()
                sources.append("bse_notices"); parts.append(bn_slim)
        except Exception as e:
            log.warning(f"bse_notices fetch failed {ticker}: {e}")

        # Telegram public channels (Indian retail finance community)
        try:
            from omni.ingest.sources import telegram_channels as _tg
            tg = _tg.fetch_telegram_news(ticker, display_name=display_name)
            if tg is not None and not tg.empty:
                cols_keep = ["title", "url", "source", "published_at", "sent_score"]
                tg_slim = tg[[c for c in cols_keep if c in tg.columns]].copy()
                sources.append("telegram"); parts.append(tg_slim)
        except Exception as e:
            log.warning(f"telegram fetch failed {ticker}: {e}")

        # SEBI PIT-2015 insider disclosures (promoters, KMPs)
        try:
            from omni.ingest.sources import sebi_insider as _pit
            pit_df = _pit.insider_disclosures(days=30)
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
        except Exception as e:
            log.warning(f"sebi_pit fetch failed {ticker}: {e}")

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
                  *, display_name: str | None = None) -> LookupBundle | None:
    """End-to-end fetch for one ticker. Returns None if no prices available."""
    display_name = display_name or ticker
    prov = Provenance()
    asof = pd.Timestamp.now().normalize()

    # PRICES (mandatory)
    prices, p_sources, agreement = _prices_combined(asset_class, ticker)
    prov.prices_sources = p_sources
    prov.prices_agreement_pct = agreement
    if prices is None or prices.empty:
        prov.add_warning("no prices from any source")
        log.warning(f"lookup: no prices for {asset_class}/{ticker}")
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
