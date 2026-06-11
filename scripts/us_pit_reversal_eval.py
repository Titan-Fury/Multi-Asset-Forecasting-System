"""US survivorship-clean cross-sectional 5d reversal eval.

Panel = data/lake/us_pit_prices.parquet (yfinance active + Tiingo delisted),
with a `delisted` flag and `dollar_vol` for a PIT illiquidity cut.

Key tests:
  1. SURVIVORSHIP: reversal on the FULL panel (incl. delisted losers) vs
     survivors-only. If reversal shrinks when delisted names are included, the
     yfinance-style "survivors only" result was survivorship-inflated.
  2. ILLIQUIDITY: within each day restrict to the bottom dollar-volume tertile
     (small/illiquid, PIT-correct, no look-ahead) vs the top tertile.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni.backtest.portfolio import simulate_cross_sectional_ls  # noqa: E402

H = 5
p = pd.read_parquet("data/lake/us_pit_prices.parquet")
# Fold in the widened delisted set (resumable checkpoint) if present.
_extra = Path("data/lake/us_delisted_extra.parquet")
if _extra.exists():
    ex = pd.read_parquet(_extra)
    ex = ex[~ex["ticker"].isin(p["ticker"].unique())]
    p = pd.concat([p, ex], ignore_index=True)
p["date"] = pd.to_datetime(p["date"])
p = p.sort_values(["ticker", "date"])
# Returns on the full contiguous series first (price filter would break shift).
g = p.groupby("ticker")["adj_close"]
p["rev5"] = -(g.transform(lambda s: s / s.shift(H) - 1.0))
p["fwd_ret_5d"] = g.transform(lambda s: s.shift(-H) / s - 1.0)
# Price floor: drop sub-$5 penny rows where delisted/illiquid names produce
# explosive % moves that dominate the cross-section. Standard small-cap hygiene.
p = p[p["adj_close"] >= 5.0].copy()
# Winsorize forward returns per date (1/99 pct) so a single delisted blowup
# can't swamp the period mean.
def _winz(s):
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)
p["fwd_ret_5d"] = p.groupby("date")["fwd_ret_5d"].transform(_winz)

n_del = p[p["delisted"]]["ticker"].nunique()
n_act = p[~p["delisted"]]["ticker"].nunique()
print(f"panel: {p['ticker'].nunique()} names ({n_del} delisted, {n_act} active), "
      f"{len(p):,} rows, {p['date'].min().date()}..{p['date'].max().date()}")


def ic(frame):
    xs = []
    for _, gg in frame.groupby("date"):
        if len(gg) >= 8:
            v, _ = spearmanr(gg["rev5"], gg["fwd_ret_5d"])
            if np.isfinite(v):
                xs.append(v)
    return float(np.mean(xs)) if xs else float("nan")


def run(frame, label):
    d = frame.dropna(subset=["rev5", "fwd_ret_5d"])
    r0 = simulate_cross_sectional_ls(d, score_col="rev5", top_q=0.2, cost_bps=0)
    r10 = simulate_cross_sectional_ls(d, score_col="rev5", top_q=0.2, cost_bps=10)
    print(f"{label:34s} IC={ic(d):+.4f} gross_Sh={r0['sharpe']:+.2f} "
          f"net@10={r10['ann_return']:+.4f} np={r10['n_trades']}")


print("\n== 1. survivorship effect ==")
run(p, "FULL (incl. delisted losers)")
run(p[~p["delisted"]], "SURVIVORS only (delisted dropped)")

print("\n== 2. illiquidity cut (PIT daily dollar-volume tertile) ==")
p["dv_rank"] = p.groupby("date")["dollar_vol"].rank(pct=True)
run(p[p["dv_rank"] <= 0.33], "BOTTOM tertile (illiquid/small)")
run(p[p["dv_rank"] >= 0.67], "TOP tertile (liquid/large)")

print("\n== sub-period (full panel) ==")
for lo, hi in [("2014-01-01", "2018-12-31"), ("2019-01-01", "2022-12-31"),
               ("2023-01-01", "2026-12-31")]:
    run(p[(p["date"] >= lo) & (p["date"] <= hi)], f"{lo[:4]}-{hi[:4]}")
