"""Feature build orchestrator.

Per ticker:
  1. Load hot OHLCV.
  2. Compute technical features.
  3. Compute fundamental features (latest snapshot as-of each row date).
  4. Compute sentiment aggregates (lagged 1 trading day).
  5. Join macro features.
  6. Compute MICROSTRUCTURE features (Phase 24.16):
       - delivery_pct_z_60d, amihud_illiq_20d, signed_vol_proxy,
         oi_change_pct_5d, fii_dii_net_z_20d, microstr_score
     India equities only; non-India tickers get NaN columns so the
     universal feature schema stays stable across asset classes.
  7. Compute forward-return targets (kept separate from feature columns).
  8. Persist to features tier.

Anti-leakage rules enforced here:
  - Fundamentals are joined as `<= row.date` (never future snapshots).
  - Sentiment is shifted +1 business day before joining.
  - Macro is taken as same-day close (already lagged because reported
    after-hours).
  - Targets are produced separately and NEVER concatenated into the feature
    columns -- stored as a parallel set of columns.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from omni.features import macro as _macro
from omni.features import microstructure as _micro
from omni.features import targets as _tgt
from omni.features import technical as _tech
from omni.storage import tiers
from omni.utils.env import universe
from omni.utils.logger import log
from omni.utils.paths import local_data_root

# Columns that are LABELS, never features
_TARGET_PREFIX = ("fwd_ret_", "fwd_dir_", "fwd_q_")


def is_target_col(c: str) -> bool:
    return c.startswith(_TARGET_PREFIX)


def _load_oi_history(ticker: str) -> pd.DataFrame | None:
    """Load per-ticker OI snapshot history from the lake. Phase 24.16 —
    populated daily by `omni.orchestrator.daily` via the (now hardened)
    NSE OI scraper. Returns None if the snapshot file isn't present."""
    p = local_data_root() / "lake" / "oi_snapshots" / f"{ticker.upper()}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:  # noqa: BLE001
        log.debug(f"oi_history load failed for {ticker}: {e}")
        return None


def build_for_ticker(asset_class: str, ticker: str,
                     *,
                     fundamentals: pd.DataFrame | None = None,
                     sentiment: pd.DataFrame | None = None,
                     macro: pd.DataFrame | None = None,
                     bhavcopy_df: pd.DataFrame | None = None,
                     flows_df: pd.DataFrame | None = None,
                     horizons: tuple[int, ...] = (1, 5, 20),
                     persist: bool = True) -> pd.DataFrame:
    try:
        prices = tiers.load_hot(asset_class, ticker)
    except FileNotFoundError:
        log.warning(f"no hot data for {asset_class}/{ticker}")
        return pd.DataFrame()

    tech = _tech.compute(prices)
    out = tech.copy()
    out["ticker"] = ticker
    out["asset_class"] = asset_class
    out["date"] = pd.to_datetime(out["date"])

    # Fundamentals (latest <= date per row, only equities)
    if asset_class not in ("crypto", "fx") and fundamentals is not None and not fundamentals.empty:
        f = fundamentals[fundamentals["ticker"] == ticker].copy()
        f["run_date"] = pd.to_datetime(f["run_date"])
        f = f.sort_values("run_date")
        # Use merge_asof (backward) to attach latest snapshot as-of each row
        out = pd.merge_asof(
            out.sort_values("date"),
            f.drop(columns=[c for c in ("ticker", "asset_class", "source") if c in f.columns]),
            left_on="date", right_on="run_date", direction="backward",
        )

    # Sentiment (lagged +1 business day)
    if sentiment is not None and not sentiment.empty:
        s = sentiment[sentiment["ticker"] == ticker].copy()
        if not s.empty:
            s["date"] = pd.to_datetime(s["date"]) + pd.tseries.offsets.BDay(1)
            out = out.merge(s.drop(columns=["ticker"]), on="date", how="left")

    # Macro
    if macro is not None and not macro.empty:
        m = macro.copy()
        m["date"] = pd.to_datetime(m["date"])
        out = out.merge(m, on="date", how="left")

    # Phase 24.16 — microstructure features (India only; other asset classes
    # get NaN columns so the universal schema stays stable). Reads OI
    # history from the lake; passes through bhavcopy + flows if the caller
    # preloaded them at universe-build scope (cheap to reuse).
    try:
        oi_hist = _load_oi_history(ticker) if asset_class == "india" else None
        out = _micro.compute_microstructure(
            out, asset_class=asset_class, ticker=ticker, prices=prices,
            bhavcopy_df=bhavcopy_df, flows_df=flows_df,
            intraday_bars=None,           # daily build doesn't have intraday
            oi_history=oi_hist,
        )
    except Exception as e:  # noqa: BLE001
        # Microstructure is opt-in / best-effort — never fail the whole build
        # over it. The columns will be NaN for this ticker.
        log.debug(f"microstructure compute skipped for {ticker}: {e}")

    # Targets (separate columns, leakage-safe)
    tg = _tgt.compute(prices, horizons=horizons)
    tg["date"] = pd.to_datetime(tg["date"])
    out = out.merge(tg, on="date", how="left")

    if persist:
        tiers.save_features(out, asset_class=asset_class, ticker=ticker)
        log.info(f"features {asset_class}/{ticker}: rows={len(out)} cols={out.shape[1]}")
    return out


def build_universe(*, horizons: tuple[int, ...] = (1, 5, 20)) -> dict[str, int]:
    """Run the full build across the configured universe.

    Caller is responsible for providing prepared `fundamentals` / `sentiment` /
    `macro` frames if those should be joined; for v1 we recompute macro per
    call and skip sentiment until news has been scored externally.

    Phase 24.16 — preload bhavcopy + FII/DII flows ONCE per universe-build
    (country-wide signals broadcast across all India tickers). Avoids one
    NSE fetch per ticker × 195 tickers.
    """
    cfg = universe()
    macro = _macro.compute()

    # Preload once. Soft-fail: if either fetch dies, we just pass None to
    # compute_microstructure which returns NaN for that signal but keeps the
    # rest of the pillar working.
    bhav = None
    flows = None
    try:
        from omni.ingest.sources.nse_bhavcopy import latest_bhavcopy  # type: ignore
        bhav = latest_bhavcopy()
    except Exception as e:  # noqa: BLE001
        log.debug(f"bhavcopy preload skipped: {e}")
    try:
        from omni.ingest.sources.nse_flows import fii_dii_activity
        flows = fii_dii_activity()
    except Exception as e:  # noqa: BLE001
        log.debug(f"flows preload skipped: {e}")

    counts: dict[str, int] = {}
    for asset_class, spec in cfg["asset_classes"].items():
        for ticker in spec["tickers"]:
            try:
                df = build_for_ticker(
                    asset_class, ticker,
                    macro=macro, bhavcopy_df=bhav, flows_df=flows,
                    horizons=horizons,
                )
                counts[f"{asset_class}/{ticker}"] = len(df)
            except Exception as e:
                log.error(f"feature build failed {asset_class}/{ticker}: {e}")
                counts[f"{asset_class}/{ticker}"] = -1
    return counts
