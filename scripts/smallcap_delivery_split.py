"""A2 probe: does 5d reversal concentrate in LOW delivery-% (speculative) names?

Delivery% (DELIV_PER from NSE bhavcopy) = fraction of traded volume taken to
demat (vs intraday churn). Low delivery -> more weak-hand / noise-trader churn,
where overreaction-driven reversal should be stronger.

CAVEAT: only one bhavcopy day (2026-06-19) is available, so delivery is used as
a STATIC per-name character proxy applied across 2014-2026. Delivery regime is
fairly persistent for a name, but this is a single snapshot -> suggestive, not
definitive. Proper test needs delivery history (multi-day bhavcopy ingestion).
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
panel["rev5"] = -(g.transform(lambda s: s / s.shift(H) - 1.0))
panel["fwd_ret_5d"] = g.transform(lambda s: s.shift(-H) / s - 1.0)

bhav = pd.read_csv("NSE_bhavcopy/sec_bhavdata_full_19062026.csv", skipinitialspace=True)
bhav.columns = [c.strip() for c in bhav.columns]
bhav = bhav[bhav["SERIES"] == "EQ"]
deliv = dict(zip(bhav["SYMBOL"], pd.to_numeric(bhav["DELIV_PER"], errors="coerce")))
panel["deliv"] = panel["ticker"].map(deliv)

matched = panel.dropna(subset=["deliv"])["ticker"].nunique()
med = panel["deliv"].median()
print(f"matched delivery for {matched} names | median DELIV_PER = {med:.1f}%")

d = panel.dropna(subset=["rev5", "fwd_ret_5d", "deliv"]).copy()


def ic(frame):
    xs = []
    for _, gg in frame.groupby("date"):
        if len(gg) >= 6:
            v, _ = spearmanr(gg["rev5"], gg["fwd_ret_5d"])
            if np.isfinite(v):
                xs.append(v)
    return float(np.mean(xs)) if xs else float("nan")


for label, sub in [("LOW deliv (<median, speculative)", d[d["deliv"] < med]),
                   ("HIGH deliv (>=median, investor)", d[d["deliv"] >= med])]:
    r0 = simulate_cross_sectional_ls(sub, score_col="rev5", top_q=0.3, cost_bps=0)
    r10 = simulate_cross_sectional_ls(sub, score_col="rev5", top_q=0.3, cost_bps=10)
    print(f"\n{label}: names={sub['ticker'].nunique()}")
    print(f"  IC={ic(sub):.4f}  gross_Sh={r0['sharpe']:.2f}  "
          f"net@10={r10['ann_return']:+.4f}  np={r10['n_trades']}")
