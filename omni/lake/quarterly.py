"""Quarterly fundamentals backfill — historical financials per ticker.

yfinance exposes ``quarterly_financials``, ``quarterly_balance_sheet``,
``quarterly_cashflow`` for the last 5-7 quarters. We pull these once per
ticker and persist as a long-format parquet under
``data/lake/fundamentals/quarterly/{TICKER}.parquet``.

Schema:
    asof_date     (datetime, fiscal quarter end)
    ticker        (str)
    asset_class   (str)
    metric        (str, e.g. 'Total Revenue', 'Net Income')
    value         (float)

Downstream: ``build_training_set.py`` merges this via merge_asof so each
price row gets the most recent fundamentals available AS OF that date.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from omni.lake import paths as lake_paths
from omni.utils.logger import log


_KEEP_METRICS_INCOME = {
    "Total Revenue", "Operating Revenue", "Net Income",
    "Operating Income", "EBITDA", "EBIT", "Gross Profit",
    "Pretax Income", "Tax Provision", "Diluted EPS", "Basic EPS",
}
_KEEP_METRICS_BS = {
    "Total Assets", "Total Equity Gross Minority Interest",
    "Total Debt", "Net Debt", "Working Capital", "Cash And Cash Equivalents",
    "Long Term Debt", "Current Liabilities", "Current Assets",
}
_KEEP_METRICS_CF = {
    "Operating Cash Flow", "Free Cash Flow",
    "Capital Expenditure", "Cash Flow From Continuing Operating Activities",
}


def _yf_symbol(asset_class: str, ticker: str) -> str:
    # Pass through tickers that already carry an explicit Yahoo suffix or
    # are an index symbol (^NSEI, ^GSPC, ...). This prevents double-suffix
    # bugs like CIANAGRO.BO -> CIANAGRO.BO.NS.
    if ticker.startswith("^") or "=" in ticker or "-" in ticker or "." in ticker:
        return ticker
    # Known BSE-only India names → .BO directly; same shared registry used
    # by omni.ingest.prices so quarterly fundamentals don't 404 on .NS.
    if asset_class == "india":
        from omni.ingest.prices import is_bse_only
        if is_bse_only(ticker):
            return f"{ticker}.BO"
    suffix = {"india": ".NS", "us": "", "crypto": "-USD", "fx": "=X"}.get(asset_class, "")
    return f"{ticker}{suffix}"


def fetch_quarterly_fundamentals(asset_class: str, ticker: str) -> pd.DataFrame:
    """Return long-format DataFrame of quarterly metrics for ``ticker``.

    Empty if asset_class is crypto/fx or yfinance has no data for this name.
    """
    if asset_class in ("crypto", "fx"):
        return pd.DataFrame()
    try:
        import yfinance as yf
        t = yf.Ticker(_yf_symbol(asset_class, ticker))
    except Exception as e:
        log.warning(f"yfinance ticker init failed {ticker}: {e}")
        return pd.DataFrame()

    rows: list[dict] = []

    def _melt(df: pd.DataFrame | None, allowed: set[str]) -> None:
        if df is None or df.empty:
            return
        for metric in df.index:
            if metric not in allowed:
                continue
            for asof_col, value in df.loc[metric].items():
                if pd.isna(value):
                    continue
                rows.append({
                    "asof_date": pd.Timestamp(asof_col).normalize(),
                    "ticker": ticker.upper(),
                    "asset_class": asset_class,
                    "metric": str(metric),
                    "value": float(value),
                })

    try:
        _melt(t.quarterly_financials, _KEEP_METRICS_INCOME)
    except Exception as e:
        log.warning(f"quarterly_financials {ticker}: {e}")
    try:
        _melt(t.quarterly_balance_sheet, _KEEP_METRICS_BS)
    except Exception as e:
        log.warning(f"quarterly_balance_sheet {ticker}: {e}")
    try:
        _melt(t.quarterly_cashflow, _KEEP_METRICS_CF)
    except Exception as e:
        log.warning(f"quarterly_cashflow {ticker}: {e}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["asof_date", "metric"]).reset_index(drop=True)


def quarterly_path(ticker: str) -> "Path":  # type: ignore[name-defined]
    return lake_paths.lake_root() / "fundamentals" / "quarterly" / f"{ticker.upper()}.parquet"


def write_quarterly(df: pd.DataFrame, ticker: str) -> int:
    if df is None or df.empty:
        return 0
    p = quarterly_path(ticker)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return len(df)


def read_quarterly(ticker: str) -> pd.DataFrame:
    p = quarterly_path(ticker)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def backfill_quarterly(pairs: Iterable[tuple[str, str]]) -> dict:
    """For each (ticker, asset_class) pull + persist quarterly history.

    Returns a stats dict with rows_total etc.
    """
    pairs = list(pairs)
    n_total = 0
    n_tickers = 0
    for ticker, ac in pairs:
        if ac in ("crypto", "fx"):
            continue
        df = fetch_quarterly_fundamentals(ac, ticker)
        rows = write_quarterly(df, ticker)
        if rows > 0:
            n_total += rows
            n_tickers += 1
            log.info(f"quarterly: {ac}/{ticker} -> {rows} rows")
    return {"tickers_with_data": n_tickers, "rows_total": n_total,
             "n_pairs": len([p for p in pairs if p[1] not in ("crypto", "fx")])}


def read_all_quarterly() -> pd.DataFrame:
    """Concat all per-ticker quarterly parquets into one long frame."""
    d = lake_paths.lake_root() / "fundamentals" / "quarterly"
    if not d.exists():
        return pd.DataFrame()
    parts = []
    for f in d.glob("*.parquet"):
        try:
            parts.append(pd.read_parquet(f))
        except Exception as e:
            log.warning(f"quarterly read failed {f}: {e}")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def pivot_for_join(quarterly_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long format → wide so each row is one (ticker, asof_date) with
    all metrics as columns. Suitable for merge_asof joining to price rows.
    """
    if quarterly_long is None or quarterly_long.empty:
        return pd.DataFrame()
    wide = quarterly_long.pivot_table(
        index=["asof_date", "ticker", "asset_class"],
        columns="metric", values="value", aggfunc="last",
    ).reset_index()
    wide.columns.name = None
    # Sanitise metric names → lower_snake_case
    rename = {}
    for c in wide.columns:
        if c in ("asof_date", "ticker", "asset_class"):
            continue
        new = (str(c).lower()
                  .replace(" ", "_").replace("/", "_")
                  .replace("-", "_").replace("__", "_").strip("_"))
        rename[c] = f"q_{new}"
    return wide.rename(columns=rename)
