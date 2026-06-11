"""Time-varying NSE delivery% reversal split (replaces the static snapshot).

Phase 28.25 used a single bhavcopy day as a static per-name delivery proxy.
Here we use the real multi-day delivery history (2020-2026) and classify each
(name, date) by its TRAILING-20d mean delivery vs the cross-sectional median
that day -> membership varies over time, no static-snapshot look-ahead.

delivery_d is known at end of day d, so conditioning the rev5->fwd(d,d+5) trade
on it is information-clean.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni.backtest.portfolio import simulate_cross_sectional_ls  # noqa: E402

H = 5
px = pd.read_parquet("data/lake/smallcap_prices.parquet")
px["date"] = pd.to_datetime(px["date"])
px = px.sort_values(["ticker", "date"])
g = px.groupby("ticker")["adj_close"]
px["rev5"] = -(g.transform(lambda s: s / s.shift(H) - 1.0))
px["fwd_ret_5d"] = g.transform(lambda s: s.shift(-H) / s - 1.0)

dl = pd.read_parquet("data/lake/nse_delivery_history.parquet")
dl["date"] = pd.to_datetime(dl["date"])
dl = dl.sort_values(["ticker", "date"])
# trailing-20d mean delivery per name (uses only past+current delivery)
dl["deliv_t20"] = (dl.groupby("ticker")["deliv_per"]
                   .transform(lambda s: s.rolling(20, min_periods=10).mean()))

m = px.merge(dl[["date", "ticker", "deliv_t20"]], on=["date", "ticker"], how="inner")
m = m.dropna(subset=["rev5", "fwd_ret_5d", "deliv_t20"])
print(f"merged panel: {m['ticker'].nunique()} names, {len(m):,} rows, "
      f"{m['date'].min().date()}..{m['date'].max().date()}")

# daily cross-sectional delivery rank -> time-varying low/high membership
m["dr"] = m.groupby("date")["deliv_t20"].rank(pct=True)


def ic(frame):
    xs = []
    for _, gg in frame.groupby("date"):
        if len(gg) >= 6:
            v, _ = spearmanr(gg["rev5"], gg["fwd_ret_5d"])
            if np.isfinite(v):
                xs.append(v)
    return float(np.mean(xs)) if xs else float("nan")


def run(frame, label):
    r0 = simulate_cross_sectional_ls(frame, score_col="rev5", top_q=0.3, cost_bps=0)
    r10 = simulate_cross_sectional_ls(frame, score_col="rev5", top_q=0.3, cost_bps=10)
    print(f"{label:42s} names~{frame['ticker'].nunique():3d} IC={ic(frame):+.4f} "
          f"gross_Sh={r0['sharpe']:+.2f} net@10={r10['ann_return']:+.4f} np={r10['n_trades']}")


print("\n== time-varying delivery split (trailing-20d, daily membership) ==")
run(m[m["dr"] < 0.5], "LOW delivery (speculative)")
run(m[m["dr"] >= 0.5], "HIGH delivery (investor-held)")
print("\n== control: same 2020-2026 window, no split ==")
run(m, "ALL (2020-2026 overlap)")
