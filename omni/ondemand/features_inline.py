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
                              lookback_days: int = 14) -> pd.DataFrame | None:
    """Return the latest `lookback_days` rows from the lake for cross-
    sectional rank parity. None when the lake is unavailable."""
    try:
        from omni.ondemand.cache import TTL, cached  # local import: tolerate
        # absent diskcache backend (e.g. fresh checkout).
    except Exception:  # noqa: BLE001
        cached, TTL = None, None

    def _read(asset_class: str) -> pd.DataFrame | None:
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


# ----------------------------------------------------------------------
# Phase 28.5 (Tier A2) — live quarterly fundamentals at inference.
# The training pipeline (build_training_set.py) merges ~24 `q_*` columns
# (q_total_assets, q_net_income, q_free_cash_flow, ...) plus `asof_date`
# via quarterly.read_all_quarterly() -> pivot_for_join() -> merge_asof.
# The inference path never produced these, so they were silently
# median-filled (the CIANAGRO/OMNITECH probe flagged 24 q_* columns
# missing). This helper fetches the same quarterly data live (yfinance),
# pivots it to the identical wide schema, and is cached so the per-call
# network cost is amortised. Quarterly statements change ~4x/year, so a
# 24h TTL is generous.
# ----------------------------------------------------------------------
def _load_quarterly_wide(asset_class: str, ticker: str) -> pd.DataFrame | None:
    """Return wide-format quarterly fundamentals (q_* cols + asof_date)
    for one ticker, or None when unavailable. Cached to amortise the
    yfinance round-trip across rapid-fire prediction calls."""
    if asset_class in ("crypto", "fx"):
        return None

    try:
        from omni.ondemand.cache import cached
    except Exception:  # noqa: BLE001
        cached = None

    def _fetch(asset_class: str, ticker: str) -> pd.DataFrame | None:
        try:
            from omni.lake.quarterly import (
                fetch_quarterly_fundamentals,
                pivot_for_join,
            )
            long_df = fetch_quarterly_fundamentals(asset_class, ticker)
            if long_df is None or long_df.empty:
                return None
            wide = pivot_for_join(long_df)
            if wide is None or wide.empty:
                return None
            return wide
        except Exception:  # noqa: BLE001
            return None

    if cached is None:
        return _fetch(asset_class, ticker)

    @cached(ttl=24 * 3600, key="features_inline_quarterly:{asset_class}:{ticker}")
    def _fetch_cached(asset_class: str, ticker: str):
        return _fetch(asset_class, ticker)

    try:
        return _fetch_cached(asset_class, ticker)
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

    # Phase 28.6 — causal OHLCV bad-tick / outlier sanitiser. Runs BEFORE
    # both technical.compute and targets.compute so a corrupt bar can't
    # poison the rolling features OR the forward labels (both read this
    # same frame). Leak-free (trailing stats + abs caps above circuit
    # limits). Gated by an env var purely so the A/B control run can
    # disable it without a code edit; on by default.
    import os as _os
    if _os.getenv("OMNI_DISABLE_OHLCV_SANITIZE", "0") != "1":
        try:
            from omni.ingest.sanitize import sanitize_ohlcv
            prices, _san_stats = sanitize_ohlcv(prices, asset_class)
            if (_san_stats["n_dropped_integrity"] + _san_stats["n_dropped_extreme"]
                    + _san_stats["n_dup"]) > 0:
                from omni.utils.logger import log
                log.info(f"ohlcv sanitize [{asset_class}/{ticker}]: {_san_stats}")
        except Exception as exc:  # noqa: BLE001
            from omni.utils.logger import log
            log.warning(f"ohlcv sanitize skipped for {ticker}: {exc}")
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

    # Phase 28.5 (Tier A2) — quarterly fundamentals merge.
    # Mirrors build_training_set.py: fetch live quarterly statements,
    # pivot to the q_* wide schema, and merge_asof(backward) so each row
    # gets the most recent quarter known as-of that date. Produces the
    # ~24 q_* columns + asof_date that the inference path was missing
    # (silently median-filled before this fix). When yfinance has no
    # statements for a micro-cap, the q_* columns stay NaN and fall
    # through to median-fill — correct for fundamentals (a missing
    # balance sheet is not "zero assets") and counted toward the Tier A3
    # data-coverage haircut.
    if asset_class not in ("crypto", "fx"):
        try:
            qwide = _load_quarterly_wide(asset_class, ticker)
            if qwide is not None and not qwide.empty:
                qsub = qwide.copy()
                if "ticker" in qsub.columns:
                    qsub = qsub[qsub["ticker"].astype(str).str.upper() == ticker.upper()]
                if not qsub.empty:
                    qsub["asof_date"] = (pd.to_datetime(qsub["asof_date"], errors="coerce")
                                           .dt.tz_localize(None).astype("datetime64[ns]"))
                    qsub = qsub.sort_values("asof_date")
                    out["date"] = (pd.to_datetime(out["date"], errors="coerce")
                                     .dt.tz_localize(None).astype("datetime64[ns]"))
                    out = out.sort_values("date")
                    out = pd.merge_asof(
                        out,
                        qsub.drop(columns=[c for c in ("ticker", "asset_class")
                                            if c in qsub.columns]),
                        left_on="date", right_on="asof_date", direction="backward",
                    )
        except Exception as exc:  # noqa: BLE001
            from omni.utils.logger import log
            log.warning(f"quarterly fundamentals inline skipped: {exc}")

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

    # Phase 28.5 (Tier A1) — news-narrative true-zeros.
    # When a ticker has NO scored news (empty feed, or a blocked source
    # like Reddit 403), the sentiment merge above is skipped and
    # `add_sentiment_z` no-ops because it requires `sent_mean`. The result
    # is that ~8 news-derived columns are absent from the frame and get
    # silently median-filled in `_align_to_universal` — i.e. a no-news
    # micro-cap is scored as if it had AVERAGE news activity, which is
    # wrong. The semantically-correct value for a no-news stock is ZERO
    # articles / zero panic ratio / no flags, not the training median.
    #
    # We only set a default when the column is ABSENT or all-NaN, so real
    # sentiment values (when news exists) are never clobbered. The `*_pct`
    # cross-sectional rank columns are intentionally NOT touched here —
    # a rank is relative, so the neutral 0.5 placeholder applied by the
    # block below is the right default for those.
    _NARRATIVE_ZERO_DEFAULTS = {
        # counts → 0 (truly zero articles / mentions)
        "n_articles": 0.0,
        "panic_kw_count": 0.0,
        "promoter_kw_count": 0.0,
        "sebi_kw_count": 0.0,
        "twitter_n_24h": 0.0,
        # scores / ratios / momentum → 0.0 (neutral, no signal)
        "sent_mean": 0.0,
        "sent_std": 0.0,
        "sent_pos_share": 0.0,
        "sent_neg_share": 0.0,
        "sent_momentum_7d": 0.0,
        "sent_z_30d": 0.0,
        "sent_accel_3d": 0.0,
        "panic_keyword_ratio": 0.0,
        "news_volume_z_60d": 0.0,
        # flags → 0.0 (no promoter / SEBI news in window)
        "promoter_news_flag": 0.0,
        "sebi_action_flag": 0.0,
    }
    try:
        for _col, _default in _NARRATIVE_ZERO_DEFAULTS.items():
            if _col not in out.columns:
                out[_col] = _default
            else:
                # Fill only the live (last) row's NaN; preserve any real
                # historical values already merged in.
                out[_col] = out[_col].fillna(_default)
    except Exception as exc:  # noqa: BLE001
        from omni.utils.logger import log
        log.warning(f"narrative zero-fill failed: {exc}")

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
