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


# ----------------------------------------------------------------------
# Phase 28 — recent-pool slice for live cross-sectional ranking.
# Reads the most-recent ~14 trading days from training_set.parquet for
# the given asset_class so the inference path can compute REAL ranks
# instead of the legacy 0.5 placeholder. Cached 5 minutes to amortise
# the parquet read across rapid-fire prediction calls.
# ----------------------------------------------------------------------

# Phase 28.4 — staleness gate. When the lake's latest date is more than
# this many CALENDAR days behind today, refuse to use the slice (would
# rank live row against stale peers) and fall through to the 0.5
# placeholder. Roughly 7 trading days; generous enough for long
# weekends and India holiday clusters.
_MAX_SLICE_AGE_DAYS = 10
def _load_recent_pool_slice(asset_class: str, *,
                              lookback_days: int = 14) -> "pd.DataFrame | None":
    """Return the latest `lookback_days` rows from the lake for cross-
    sectional rank parity. None when the lake is unavailable."""
    try:
        from omni.ondemand.cache import cached, TTL  # local import: tolerate
        # absent diskcache backend (e.g. fresh checkout).
    except Exception:  # noqa: BLE001
        cached, TTL = None, None

    def _read(asset_class: str) -> "pd.DataFrame | None":
        try:
            from pathlib import Path
            p = Path("data") / "lake" / "training" / "training_set.parquet"
            if not p.exists():
                return None
            # PyArrow filters keep the read O(asset_class slice) instead
            # of O(full panel).
            try:
                df = pd.read_parquet(
                    p, filters=[("asset_class", "==", asset_class)],
                    columns=None,  # need every rank-input col + date
                )
            except Exception:
                # Fallback: read full and filter in-memory.
                df = pd.read_parquet(p)
                if "asset_class" in df.columns:
                    df = df[df["asset_class"] == asset_class]
            if df is None or df.empty or "date" not in df.columns:
                return None
            df["date"] = pd.to_datetime(df["date"])
            # Phase 28.4 — freshness gate. The original implementation
            # forced the live row's date to `df["date"].max()` regardless
            # of how stale that max was, then ranked against peers from
            # that stale date. When the daily orchestrator hadn't run for
            # a week+, that meant ranking today's CIANAGRO against peers
            # from a week ago — silently incorrect ranks that fed into
            # ten or so `*_pct` features. We now refuse slices whose max
            # date is more than `_MAX_SLICE_AGE_DAYS` calendar days behind
            # today and let the caller fall through to the 0.5 placeholder
            # path (which at least signals "no data" rather than "fake
            # data"). The threshold is generous (10 calendar days, ~7
            # trading days) so weekends / Diwali holidays don't trip it.
            try:
                from omni.utils.logger import log
                max_dt = pd.Timestamp(df["date"].max())
                # Phase 28.4 fix-up — `pd.Timestamp.utcnow()` is deprecated in
                # pandas 2.x (will return tz-aware in a future major). Use the
                # supported `.now("UTC")` form, then strip the tz so the
                # arithmetic against the (naive) `max_dt` from the parquet
                # works without raising. Wrapping the previous deprecated
                # call inside a broad except meant the freshness gate was
                # silently disabled if pandas raised — that's exactly the
                # silent failure mode this gate was meant to prevent.
                today = (pd.Timestamp.now("UTC")
                            .tz_localize(None)
                            .normalize())
                age_days = max(0, int((today - max_dt.normalize()).days))
                if age_days > _MAX_SLICE_AGE_DAYS:
                    log.warning(
                        f"_load_recent_pool_slice[{asset_class}]: latest "
                        f"date {max_dt.date()} is {age_days}d old "
                        f"(> {_MAX_SLICE_AGE_DAYS}d gate); refusing to "
                        f"rank against stale peers, falling back to "
                        f"0.5 placeholders"
                    )
                    return None
            except (ValueError, TypeError, AttributeError):
                # Narrow exception list — these are the only things date
                # arithmetic can legitimately raise (all-NaT max, pandas
                # API drift, missing tzinfo). A broad `except Exception`
                # would silently mask the very class of bug the gate is
                # meant to detect.
                pass
            cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
            return df[df["date"] >= cutoff].reset_index(drop=True)
        except Exception:  # noqa: BLE001
            return None

    if cached is None or TTL is None:
        return _read(asset_class)

    @cached(ttl=300, key="features_inline_pool_slice:{asset_class}")
    def _read_cached(asset_class: str):
        return _read(asset_class)

    try:
        return _read_cached(asset_class)
    except Exception:  # noqa: BLE001
        return None


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

    # Phase 28.4 — per-row circuit-history features (India only). The
    # training path (`omni.features.build.build_for_ticker`) calls
    # `add_circuit_features` against the raw OHLC frame to produce 9
    # feature columns: is_circuit_bar, circuit_hits_{5,20,60}d,
    # circuit_lower_hits_20d, circuit_upper_hits_20d,
    # consecutive_lock_days, days_since_last_lock,
    # last_lock_direction_signed. Inference previously omitted this whole
    # block; models with non-zero split importance on those features were
    # being served the training-time median (a constant) at every live
    # row. mask_circuit_labels is intentionally NOT called here — it
    # operates on `fwd_*` label columns which the prediction path doesn't
    # consume (is_target_col strips them from feature_cols at training
    # time).
    try:
        from omni.features.circuit import add_circuit_features
        ohlc_cols = [c for c in ("date", "open", "high", "low", "close",
                                  "adj_close") if c in prices.columns]
        if ohlc_cols and "date" in ohlc_cols:
            ch = add_circuit_features(prices[ohlc_cols].copy(),
                                        asset_class=asset_class)
            ch_cols = [c for c in ch.columns
                        if c not in ("open", "high", "low", "close", "adj_close")]
            ch = ch[ch_cols]
            ch["date"] = pd.to_datetime(ch["date"]).astype("datetime64[ns]")
            out = out.merge(ch, on="date", how="left")
    except Exception as exc:  # noqa: BLE001
        from omni.utils.logger import log
        log.warning(f"circuit features inline skipped: {exc}")

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

    # Phase 28.4 — quality composites (Sloan accruals, Piotroski F-Score,
    # AFP QMJ). Training path runs `add_quality` immediately after the
    # fundamentals merge so the composite formulas can read net_income,
    # operating_cash_flow, etc. Inference previously omitted these three
    # columns; the rubric scorer's QMJ rank and the model's piotroski
    # split were both being served the training-time median. Crypto/FX
    # produces all-NaN columns by design (matches training) so the
    # universal schema stays stable.
    try:
        from omni.features.quality import add_quality
        out = add_quality(out)
    except Exception as exc:  # noqa: BLE001
        from omni.utils.logger import log
        log.warning(f"quality features inline skipped: {exc}")

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

    # Phase 28.4 — diff-encoding + lagged-realised-label features. The
    # training path (`omni.features.build.build_for_ticker`) calls
    # `add_diff_and_lagged` after the targets merge so `lag_real_fwd_dir_<H>d`
    # can read fwd_dir_*. Inference previously skipped this step entirely,
    # which meant the model's splits on those 7 columns were silently
    # served the training-time median at every live row. With the schema
    # contract (Phase 28.4) the absence would now surface as an error log
    # line, but the proper fix is to actually produce the columns.
    try:
        from omni.features.diff_lagged import add_diff_and_lagged
        out = add_diff_and_lagged(out, horizons=horizons)
    except Exception as exc:  # noqa: BLE001
        from omni.utils.logger import log
        log.warning(f"diff_lagged inline failed: {exc}")

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
    # Phase 28 — cross-sectional rank parity.
    # PRIOR BUG: this block emitted 0.5 placeholders for every `*_pct` column
    # because a single live row trivially has rank 0.5 within its own
    # group. The training pipeline computes ranks across the pooled panel
    # (build_training_set.py), so the model ALSO trained on real ranks —
    # at inference the 0.5 constant stripped all signal from features the
    # model relied on (`ret_20d_pct`, `panic_keyword_ratio_pct`, etc.).
    # FIX: read the most-recent slice of training_set.parquet for the same
    # asset_class, append the live row, run the rank transform, copy the
    # resulting `*_pct` columns back. If the slice is unavailable (cold
    # start / no lake), fall back to the old 0.5 placeholder.
    try:
        from omni.features.cross_sectional import add_cross_sectional_ranks
        if "asset_class" not in out.columns:
            out["asset_class"] = asset_class
        probe_cols = list(out.columns)
        sample = out.head(2).copy()
        try:
            transformed = add_cross_sectional_ranks(sample)
            new_cols = [c for c in transformed.columns if c not in probe_cols]
        except Exception:
            new_cols = []

        # Try the live-rank path first.
        ranked_ok = False
        try:
            slice_df = _load_recent_pool_slice(asset_class)
            if slice_df is not None and not slice_df.empty:
                live_row = out.tail(1).copy()
                # Align the live row's date to the slice's max date so the
                # group-by-(date, asset_class) puts it in the SAME bucket as
                # the persisted population. This is what makes the rank
                # meaningful — comparing today's CIANAGRO against today's
                # other India equities, not against itself.
                latest_date = slice_df["date"].max()
                live_row = live_row.copy()
                live_row["date"] = latest_date
                live_row["asset_class"] = asset_class
                # Ensure column overlap; the rank transform only operates on
                # known input cols (_RANK_COLS in cross_sectional.py).
                pooled = pd.concat([slice_df, live_row], ignore_index=True, sort=False)
                pooled_ranked = add_cross_sectional_ranks(pooled)
                live_ranked = pooled_ranked.iloc[[-1]]
                for c in new_cols:
                    if c in live_ranked.columns:
                        out[c] = float(live_ranked[c].iloc[0]) if pd.notna(
                            live_ranked[c].iloc[0]) else 0.5
                ranked_ok = True
        except Exception as exc:  # noqa: BLE001
            from omni.utils.logger import log
            log.debug(f"live cross-sectional rank failed, falling back: {exc}")

        if not ranked_ok:
            # Cold-start fallback: 0.5 placeholder (identical to legacy
            # behaviour so we never regress relative to before this fix).
            for c in new_cols:
                if c not in out.columns:
                    out[c] = 0.5
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
