"""DF-input feature builder for the on-demand path.

Mirrors omni.features.build.build_for_ticker but takes the prices, fundamentals,
sentiment, and macro frames as inputs instead of reading from the storage tier.
Never persists anything. Pure function -> ready to score.

The output schema and leakage rules are IDENTICAL to build_for_ticker, so a
universal model trained from this builder is interchangeable with one trained
from the batch path. This is the contract that makes the hybrid plan work.
"""
from __future__ import annotations

import pandas as pd

from omni.features import targets as _tgt
from omni.features import technical as _tech


def build_features_inline(
    asset_class: str,
    ticker: str,
    prices: pd.DataFrame,
    *,
    fundamentals: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    """Build a feature frame for one ticker from in-memory inputs.

    Parameters
    ----------
    prices : DataFrame[date, open, high, low, close, adj_close, volume]
    fundamentals : optional DataFrame[run_date, ticker, ...] - latest snapshot wins
    sentiment : optional DataFrame[date, ticker, sent_mean, n_articles, ...]
    macro : optional DataFrame[date, vix_level, ...]
    horizons : forward-return horizons for target columns

    Returns
    -------
    DataFrame with technical/fundamental/sentiment/macro features merged plus
    forward-return target columns. Empty if `prices` is empty.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()

    tech = _tech.compute(prices)
    out = tech.copy()
    out["ticker"] = ticker
    out["asset_class"] = asset_class
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")

    # Fundamentals (latest <= date per row, only equities). Yfinance reports the
    # run date as today, but the daily price bar may end yesterday — so any
    # fundamentals stamped strictly in the future would merge_asof to NaN. Roll
    # the latest snapshot's run_date back to the last price date so it lands on
    # the most recent rows. Quarterly statements rarely change intra-day, so
    # this is leak-free for live prediction.
    if asset_class not in ("crypto", "fx") and fundamentals is not None and not fundamentals.empty:
        f = fundamentals[fundamentals["ticker"] == ticker].copy()
        if not f.empty:
            f["run_date"] = pd.to_datetime(f["run_date"]).astype("datetime64[ns]")
            last_price_date = pd.to_datetime(out["date"].max())
            f.loc[f["run_date"] > last_price_date, "run_date"] = last_price_date
            f = f.sort_values("run_date")
            out = pd.merge_asof(
                out.sort_values("date"),
                f.drop(columns=[c for c in ("ticker", "asset_class", "source") if c in f.columns]),
                left_on="date", right_on="run_date", direction="backward",
            )

    # Sentiment lagged +1 business day (leak-free), then forward-filled
    # so the most recent price row inherits the latest available sentiment.
    if sentiment is not None and not sentiment.empty:
        s = sentiment[sentiment["ticker"] == ticker].copy() if "ticker" in sentiment.columns else sentiment.copy()
        if not s.empty:
            s["date"] = (pd.to_datetime(s["date"]).astype("datetime64[ns]")
                          + pd.tseries.offsets.BDay(1))
            drop_cols = [c for c in ("ticker",) if c in s.columns]
            out = out.merge(s.drop(columns=drop_cols), on="date", how="left")
            # Phase 24.18 — added panic_kw_count, promoter_kw_count,
            # sebi_kw_count to the forward-fill list so the rolling-7d
            # narrative features in sentiment_z see consistent inputs even
            # on weekends/holidays where sentiment.aggregate produces no row.
            sent_cols = [c for c in ("n_articles", "sent_mean", "sent_std",
                                       "sent_pos_share", "sent_neg_share",
                                       "sent_momentum_7d",
                                       "panic_kw_count", "promoter_kw_count",
                                       "sebi_kw_count") if c in out.columns]
            if sent_cols:
                out = out.sort_values("date")
                out[sent_cols] = out[sent_cols].ffill()

    # Macro (same-day join)
    if macro is not None and not macro.empty:
        m = macro.copy()
        m["date"] = pd.to_datetime(m["date"]).astype("datetime64[ns]")
        out = out.merge(m, on="date", how="left")

    # Targets
    tg = _tgt.compute(prices, horizons=horizons)
    if not tg.empty:
        tg["date"] = pd.to_datetime(tg["date"]).astype("datetime64[ns]")
        out = out.merge(tg, on="date", how="left")

    # ---- step-2 high-impact feature additions ----
    # Order matters: vol-adjusted momentum needs ret_Hd, sentiment z needs sent_mean,
    # cross-sectional ranks need raw fundamentals + tech features, regime
    # interactions need 'regime' col (added downstream) so we skip if absent.
    try:
        from omni.features.vol_adjusted_momentum import add_vol_adjusted_momentum
        out = add_vol_adjusted_momentum(out)
    except Exception as exc:  # noqa: BLE001
        from omni.utils.logger import log
        log.warning(f"vol-adj momentum failed: {exc}")
    try:
        from omni.features.sentiment_z import add_sentiment_z
        out = add_sentiment_z(out)
    except Exception as exc:  # noqa: BLE001
        from omni.utils.logger import log
        log.warning(f"sentiment_z failed: {exc}")
    # Audit-doc B2 fix: do NOT compute cross-sectional ranks on a single ticker
    # in the live path — every percentile collapses to 0.5 and the model sees
    # constant features that disagree with training (which ranks across pooled
    # universe). Instead emit zero-filled placeholder columns so the schema
    # matches but feature distribution is harmless.
    try:
        from omni.features.cross_sectional import add_cross_sectional_ranks
        # Probe the column set this transform produces by running on a
        # cheap two-row dummy frame, then zero-fill those columns on `out`.
        if "asset_class" not in out.columns:
            out["asset_class"] = asset_class
        probe_cols = list(out.columns)
        try:
            sample = out.head(2).copy()
            transformed = add_cross_sectional_ranks(sample)
            new_cols = [c for c in transformed.columns if c not in probe_cols]
        except Exception:
            new_cols = []
        for c in new_cols:
            if c not in out.columns:
                out[c] = 0.5  # neutral percentile
    except Exception as exc:  # noqa: BLE001
        from omni.utils.logger import log
        log.warning(f"cross-sectional rank placeholder failed: {exc}")
    try:
        from omni.features.earnings import add_earnings_features
        out = add_earnings_features(out, ticker=ticker, asset_class=asset_class)
    except Exception as exc:  # noqa: BLE001
        from omni.utils.logger import log
        log.warning(f"earnings features failed: {exc}")

    # ---- S1-lite microstructure (Phase 19) ---------------------------------
    # Free-feed proxy for VPIN/OFI/Kyle's lambda. India equities only;
    # non-India returns NaN columns to keep the schema stable.
    try:
        from omni.features.microstructure import compute_microstructure
        bhav = sentiment.attrs.get("bhavcopy") if sentiment is not None else None
        out = compute_microstructure(
            out, asset_class, ticker, prices,
            bhavcopy_df=bhav,
            flows_df=getattr(sentiment, "attrs", {}).get("fii_dii") if sentiment is not None else None,
            intraday_bars=getattr(sentiment, "attrs", {}).get("intraday") if sentiment is not None else None,
            oi_history=getattr(sentiment, "attrs", {}).get("oi_history") if sentiment is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        from omni.utils.logger import log
        log.warning(f"microstructure features failed: {exc}")

    # ---- S3 news embeddings (Phase 19) -------------------------------------
    # Adds emb_00..emb_31 (PCA-reduced Nvidia Nemotron embeddings).
    # Lagged +1 BDay; opt-in via OMNI_DISABLE_EMB=1 to skip for benchmarking.
    import os as _os
    if _os.environ.get("OMNI_DISABLE_EMB", "0") != "1":
        try:
            from omni.features.news_embeddings import add_news_embeddings
            news_raw = getattr(sentiment, "attrs", {}).get("news_raw") if sentiment is not None else None
            out = add_news_embeddings(out, ticker, news_raw=news_raw)
        except Exception as exc:  # noqa: BLE001
            from omni.utils.logger import log
            log.warning(f"news embeddings failed: {exc}")
    # NB: regime_interactions runs in build_features_full when 'regime' present;
    # for the inline (live) path the rubric scorer reads regime separately so
    # interactions aren't strictly needed for the live row.

    return out
