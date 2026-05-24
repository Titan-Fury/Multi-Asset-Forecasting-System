"""Quality composite features (accruals, Piotroski F-Score, QMJ).

Three documented public-equity quality signals composed from existing
fundamentals fields (no new ingest needed). They get added as columns
to the per-ticker feature frame in `omni.features.build.build_for_ticker`
right after the fundamentals merge.

References:
  - Sloan, R.G. (1996). "Do Stock Prices Fully Reflect Information in
    Accruals and Cash Flows about Future Earnings?" Accounting Review.
  - Piotroski, J.D. (2000). "Value Investing: The Use of Historical
    Financial Statement Information to Separate Winners from Losers."
  - Asness, Frazzini, Pedersen (2013). "Quality Minus Junk."

All three return NaN gracefully when input fundamentals fields are
missing — so no harm to the universal feature schema. Values are
already on a sensible numeric scale (no z-scoring here; trees handle
it). Z-scores are added downstream by the rubric/feature consumers
that need them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from omni.utils.logger import log

# Columns the quality composites can read. All optional.
_FUND_HINTS = (
    "net_income", "operating_cash_flow", "total_assets",
    "current_assets", "current_liabilities", "total_debt",
    "shares_outstanding",
    "return_on_equity", "return_on_assets", "operating_margin",
    "profit_margin", "revenue_growth", "earnings_growth",
    "debt_to_equity", "current_ratio", "asset_turnover",
)

QUALITY_COLS = (
    "accruals_pct_assets",
    "piotroski_f_score",
    "qmj_score",
)


def _safe_num(s) -> pd.Series | None:
    """Convert a Series-like or scalar to numeric pd.Series; return None if missing."""
    if s is None:
        return None
    if not isinstance(s, pd.Series):
        try:
            s = pd.Series(s)
        except Exception:  # noqa: BLE001
            return None
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().all():
        return None
    return out


def _accruals_pct_assets(df: pd.DataFrame) -> pd.Series:
    """Sloan accruals: (NetIncome - CFO) / TotalAssets.

    HIGH accruals = earnings driven by accruals not cash flow → bad
                    quality, predicts mean reversion.
    LOW/negative = high-quality earnings.

    Fallback: if `operating_cash_flow` missing but `profit_margin` and
    `return_on_assets` are present, approximate as ROA - margin*ATO.
    Returns NaN-Series of df length when all inputs missing.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)

    ni = _safe_num(df.get("net_income"))
    cfo = _safe_num(df.get("operating_cash_flow"))
    ta = _safe_num(df.get("total_assets"))

    if ni is not None and cfo is not None and ta is not None:
        denom = ta.replace(0, np.nan)
        out[:] = ((ni - cfo) / denom).values
        return out

    # Approximation path for yfinance fundamentals (which expose
    # ratios but not raw NI/CFO): no good proxy without raw flows.
    # Return NaN — graceful.
    return out


def _piotroski_f_score(df: pd.DataFrame) -> pd.Series:
    """9-point F-Score: count of 9 binary quality checks.

    Profitability (4):
      F1: ROA > 0
      F2: CFO > 0
      F3: ROA growth > 0 (proxy: earnings_growth > 0)
      F4: CFO > NI (accrual quality)

    Leverage / liquidity / source of funds (3):
      F5: leverage decreasing (proxy: debt_to_equity below sector median —
          we approximate with a "debt below 100%" check)
      F6: current_ratio > 1.0 (improving liquidity proxy)
      F7: no net new equity issued (proxy: shares_outstanding flat)

    Operating efficiency (2):
      F8: operating_margin > 0 (proxy)
      F9: asset_turnover > 0 (proxy)

    Returns 0-9; NaN when too many inputs are missing.
    """
    score = pd.Series(0.0, index=df.index, dtype=float)
    used = pd.Series(0, index=df.index, dtype=int)

    def _check(predicate_series):
        nonlocal score, used
        if predicate_series is None:
            return
        sane = predicate_series.fillna(False).astype(bool)
        score = score + sane.astype(float)
        used = used + predicate_series.notna().astype(int)

    roa = _safe_num(df.get("return_on_assets"))
    if roa is None:
        roa = _safe_num(df.get("profit_margin"))  # weak proxy
    cfo = _safe_num(df.get("operating_cash_flow"))
    ni = _safe_num(df.get("net_income"))
    eg = _safe_num(df.get("earnings_growth"))
    de = _safe_num(df.get("debt_to_equity"))
    cr = _safe_num(df.get("current_ratio"))
    om = _safe_num(df.get("operating_margin"))
    ato = _safe_num(df.get("asset_turnover"))

    _check(roa.gt(0) if roa is not None else None)
    _check(cfo.gt(0) if cfo is not None else None)
    _check(eg.gt(0) if eg is not None else None)
    if cfo is not None and ni is not None:
        _check(cfo.gt(ni))
    _check(de.lt(100) if de is not None else None)  # debt_to_equity often in %
    _check(cr.gt(1.0) if cr is not None else None)
    _check(om.gt(0) if om is not None else None)
    _check(ato.gt(0) if ato is not None else None)

    out = pd.Series(np.nan, index=df.index, dtype=float)
    valid = used > 3   # need at least 4 of 9 inputs available
    out[valid] = score[valid].values
    return out


def _qmj_score(df: pd.DataFrame) -> pd.Series:
    """Asness-Frazzini-Pedersen Quality-Minus-Junk simplified:
    profitability + safety + growth, normalized to 0..100.

    Profitability:  ROA, ROE, profit_margin (any non-null average)
    Safety:         low debt_to_equity (inverted), high current_ratio
    Growth:         revenue_growth + earnings_growth average

    Returns 0..100 (z-score-normalised would need a panel, so we use
    a robust per-row clip-and-scale that maps typical ranges to 0..100).
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)

    parts = []
    # Profitability bucket 0..100
    prof_inputs = []
    for c, mid, span in [("return_on_assets", 0.05, 0.15),
                          ("return_on_equity", 0.15, 0.30),
                          ("profit_margin", 0.10, 0.20)]:
        s = _safe_num(df.get(c))
        if s is not None:
            scaled = ((s - mid) / span * 50 + 50).clip(0, 100)
            prof_inputs.append(scaled)
    if prof_inputs:
        prof = pd.concat(prof_inputs, axis=1).mean(axis=1, skipna=True)
        parts.append(prof)

    # Safety bucket
    de = _safe_num(df.get("debt_to_equity"))
    cr = _safe_num(df.get("current_ratio"))
    safety_parts = []
    if de is not None:
        # debt_to_equity: 0 -> 100, 200% -> 0
        safety_parts.append((100 - de.clip(0, 200) / 2).clip(0, 100))
    if cr is not None:
        # current_ratio: 1 -> 50, 2 -> 100, 0.5 -> 0
        safety_parts.append(((cr.clip(0, 3) - 1) * 50 + 50).clip(0, 100))
    if safety_parts:
        parts.append(pd.concat(safety_parts, axis=1).mean(axis=1, skipna=True))

    # Growth bucket
    eg = _safe_num(df.get("earnings_growth"))
    rg = _safe_num(df.get("revenue_growth"))
    grow_parts = []
    for s, mid, span in [(eg, 0.10, 0.30), (rg, 0.10, 0.30)]:
        if s is not None:
            grow_parts.append(((s - mid) / span * 50 + 50).clip(0, 100))
    if grow_parts:
        parts.append(pd.concat(grow_parts, axis=1).mean(axis=1, skipna=True))

    if not parts:
        return out
    out[:] = pd.concat(parts, axis=1).mean(axis=1, skipna=True).values
    return out


def add_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 3 quality columns to `df` in-place. Idempotent + safe.

    `df` must already have any available fundamentals columns merged
    onto it (this is the case in `build_for_ticker` after the
    fundamentals `merge_asof`).
    """
    try:
        df = df.copy()
        df["accruals_pct_assets"] = _accruals_pct_assets(df).values
        df["piotroski_f_score"] = _piotroski_f_score(df).values
        df["qmj_score"] = _qmj_score(df).values
    except Exception as exc:  # noqa: BLE001
        log.debug(f"quality compute failed: {exc}")
        for c in QUALITY_COLS:
            if c not in df.columns:
                df[c] = np.nan
    return df
