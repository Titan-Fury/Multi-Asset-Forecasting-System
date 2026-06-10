"""Cross-sectional 5-day reversal eval on the NSE mid/small-cap panel.

Reads data/lake/smallcap_prices.parquet (built by fetch_smallcap_panel.py),
computes a 5d reversal score (rev5 = -ret_5d), and reports:
  * daily rank-IC vs fwd 5d return (Spearman)
  * net-of-cost long/short Sharpe across a cost grid
  * the same broken out by sub-period

Honest-eval notes: non-overlapping h-day periods (handled by the simulator),
market-neutral book, survivorship-caveated (yfinance = current survivors only).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni.backtest.portfolio import simulate_cross_sectional_ls  # noqa: E402

H = 5
panel = pd.read_parquet("data/lake/smallcap_prices.parquet")
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["ticker", "date"])

g = panel.groupby("ticker")["adj_close"]
panel["ret_5d"] = g.transform(lambda s: s / s.shift(H) - 1.0)
panel["fwd_ret_5d"] = g.transform(lambda s: s.shift(-H) / s - 1.0)
panel["rev5"] = -panel["ret_5d"]
d = panel.dropna(subset=["rev5", "fwd_ret_5d"]).copy()
print(f"panel: {panel['ticker'].nunique()} names, {len(panel):,} rows, "
      f"{panel['date'].min().date()}..{panel['date'].max().date()}")


def daily_ic(frame):
    ics = []
    for _, gg in frame.groupby("date"):
        if len(gg) >= 8:
            ic, _ = spearmanr(gg["rev5"], gg["fwd_ret_5d"])
            if np.isfinite(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


print("\n== full-sample cost sensitivity ==")
for cb in (0, 5, 10, 15, 20):
    r = simulate_cross_sectional_ls(d, score_col="rev5", top_q=0.2, cost_bps=cb)
    print(f"cost={cb:>2}bps  sharpe={r['sharpe']:+.2f}  net_ann={r['ann_return']:+.4f}")

print(f"\nfull-sample IC = {daily_ic(d):.4f}")

print("\n== sub-period ==")
for lo, hi in [("2014-01-01", "2018-12-31"), ("2019-01-01", "2022-12-31"),
               ("2023-01-01", "2026-12-31")]:
    sub = d[(d["date"] >= lo) & (d["date"] <= hi)]
    if sub.empty:
        continue
    r10 = simulate_cross_sectional_ls(sub, score_col="rev5", top_q=0.2, cost_bps=10)
    r15 = simulate_cross_sectional_ls(sub, score_col="rev5", top_q=0.2, cost_bps=15)
    r0 = simulate_cross_sectional_ls(sub, score_col="rev5", top_q=0.2, cost_bps=0)
    print(f"{lo[:4]}-{hi[:4]}: IC={daily_ic(sub):.4f} gross_Sh={r0['sharpe']:.2f} "
          f"net@10={r10['ann_return']:+.4f} net@15={r15['ann_return']:+.4f} "
          f"np={r10['n_trades']}")
