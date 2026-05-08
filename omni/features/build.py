"""Feature build orchestrator.

Per ticker:
  1. Load hot OHLCV.
  2. Compute technical features.
  3. Compute fundamental features (latest snapshot as-of each row date).
  4. Compute sentiment aggregates (lagged 1 trading day).
  5. Join macro features.
  6. Compute forward-return targets (kept separate from feature columns).
  7. Persist to features tier.

Anti-leakage rules enforced here:
  - Fundamentals are joined as `<= row.date` (never future snapshots).
  - Sentiment is shifted +1 business day before joining.
  - Macro is taken as same-day close (already lagged because reported
    after-hours).
  - Targets are produced separately and NEVER concatenated into the feature
    columns -- stored as a parallel set of columns.
"""
from __future__ import annotations

import pandas as pd

from omni.features import fundamental as _fund
from omni.features import macro as _macro
from omni.features import sentiment as _sent
from omni.features import targets as _tgt
from omni.features import technical as _tech
from omni.storage import tiers
from omni.utils.env import universe
from omni.utils.logger import log

# Columns that are LABELS, never features
_TARGET_PREFIX = ("fwd_ret_", "fwd_dir_", "fwd_q_")


def is_target_col(c: str) -> bool:
    return c.startswith(_TARGET_PREFIX)


def build_for_ticker(asset_class: str, ticker: str,
                     *,
                     fundamentals: pd.DataFrame | None = None,
                     sentiment: pd.DataFrame | None = None,
                     macro: pd.DataFrame | None = None,
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
    """
    cfg = universe()
    macro = _macro.compute()

    counts: dict[str, int] = {}
    for asset_class, spec in cfg["asset_classes"].items():
        for ticker in spec["tickers"]:
            try:
                df = build_for_ticker(asset_class, ticker, macro=macro, horizons=horizons)
                counts[f"{asset_class}/{ticker}"] = len(df)
            except Exception as e:
                log.error(f"feature build failed {asset_class}/{ticker}: {e}")
                counts[f"{asset_class}/{ticker}"] = -1
    return counts
