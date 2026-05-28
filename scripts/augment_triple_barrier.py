"""Phase 24.18 — Augment the training set with triple-barrier labels.

Reads data/lake/training/training_set.parquet, runs triple_barrier() per
ticker on the embedded OHLC, and adds tb_label_<H>d + tb_first_<H>d columns.
Writes the augmented set back in place (with a .bak copy).

This is far cheaper than rebuilding the lake — we only need the OHLC that's
already in the panel (or recompute from `adj_close` if low/high missing).
"""
from __future__ import annotations

import sys
import warnings
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from omni.utils.logger import log
from omni.features.targets import triple_barrier


def main() -> int:
    ts = Path("data/lake/training/training_set.parquet")
    if not ts.exists():
        log.error(f"no training set at {ts}")
        return 1

    log.info(f"loading {ts}...")
    df = pd.read_parquet(ts)
    log.info(f"loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    have_low = "low" in df.columns
    have_high = "high" in df.columns
    log.info(f"OHLC available: high={have_high} low={have_low} "
             f"(triple_barrier falls back to adj_close if missing)")

    if "ticker" not in df.columns or "date" not in df.columns:
        log.error("training set missing ticker/date columns")
        return 1
    if "log_ret_1d" not in df.columns:
        log.error("training set missing log_ret_1d — can't reconstruct close path")
        return 1

    df["date"] = pd.to_datetime(df["date"])
    horizons = (1, 5, 21)

    # The training set is feature-only — no OHLC. Reconstruct a synthetic
    # close per ticker from cumulative log_ret_1d. Triple-barrier then runs
    # on the close-only path (high=low=close) which is a slightly looser
    # version of de Prado's intraday-aware variant but still much better
    # than naive fwd_dir as a label.
    tb_chunks = []
    n_tickers = df["ticker"].nunique()
    for i, (tkr, sub) in enumerate(df.groupby("ticker", sort=False)):
        sub = sub.sort_values("date").reset_index(drop=True)
        try:
            r = pd.to_numeric(sub["log_ret_1d"], errors="coerce").fillna(0.0)
            synth_close = pd.Series(np.exp(r.cumsum()).values, index=sub.index)
            ohlc = pd.DataFrame({
                "date": sub["date"].values,
                "adj_close": synth_close.values,
                "high": synth_close.values,
                "low": synth_close.values,
            })
            tb = triple_barrier(ohlc, horizons=horizons,
                                pt_mult=1.5, sl_mult=1.0, vol_window=40)
            tb["ticker"] = tkr
            tb_chunks.append(tb)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"triple_barrier failed for {tkr}: {exc}")
        if (i + 1) % 25 == 0:
            log.info(f"  [{i+1}/{n_tickers}] processed")

    if not tb_chunks:
        log.error("no triple-barrier output produced")
        return 1

    tb_df = pd.concat(tb_chunks, ignore_index=True)
    tb_cols = [c for c in tb_df.columns
               if c.startswith(("tb_label_", "tb_first_"))]
    log.info(f"triple_barrier output: {tb_df.shape}, cols={tb_cols}")

    # Merge back on (ticker, date)
    df = df.merge(tb_df[["ticker", "date", *tb_cols]],
                  on=["ticker", "date"], how="left")

    # Quick distribution check on the headline tb_label_5d
    if "tb_label_5d" in df.columns:
        vc = df["tb_label_5d"].value_counts(dropna=False).to_dict()
        n_total = len(df)
        log.info(f"tb_label_5d distribution (of {n_total:,} rows): {vc}")

    # Backup, then overwrite
    bak = ts.with_suffix(ts.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(ts, bak)
        log.info(f"backed up original -> {bak.name}")

    df.to_parquet(ts, index=False)
    log.info(f"wrote augmented training set: {df.shape[0]:,} rows "
             f"x {df.shape[1]} cols -> {ts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
