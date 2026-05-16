"""End-to-end CIANAGRO probe + reality check.

1. Run predict_ticker for CIANAGRO (uses live yfinance data — last
   close should be May 15 2026 since today is May 16).
2. Pull the actual OHLCV for the last ~30 sessions.
3. Compare: predicted direction/probability vs realised h+1, h+5, h+21
   returns where available.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

from omni.ondemand.cache import purge_ticker
from omni.ondemand.predict import predict_ticker
from omni.ondemand.universe import resolve

TK = "CIANAGRO"
purge_ticker(TK)
res = resolve(TK)
print(f"resolve: {res}")
card = predict_ticker(res.ticker, res.asset_class, display_name=res.display_name)
d = card.to_dict()

print()
print("=" * 78)
print(f"  CIANAGRO prediction card")
print("=" * 78)
for k in ("ticker", "asof", "last_close", "last_close_date",
          "proba_up", "blended_p", "conviction", "direction",
          "expected_move_pct", "expected_low_pct", "expected_high_pct",
          "model_kind"):
    print(f"  {k:>22s}: {d.get(k)}")

print()
print("  pillars:")
for k, v in (d.get("pillars") or {}).items():
    print(f"    {k:>22s}: {v}")

print()
print("  per-horizon proba:")
for k, v in (d.get("per_horizon_proba") or {}).items():
    print(f"    h={k}: {v}")

print()
print("=" * 78)
print(f"  Actual price history (last 25 sessions)")
print("=" * 78)
import yfinance as yf
yt = yf.Ticker(f"{TK}.NS")
hist = yt.history(period="3mo", auto_adjust=False)
if hist.empty:
    yt = yf.Ticker(f"{TK}.BO")
    hist = yt.history(period="3mo", auto_adjust=False)
if hist.empty:
    print("  yfinance returned empty for both .NS and .BO")
    sys.exit(0)
hist = hist.reset_index()
hist["date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
hist = hist.sort_values("date").reset_index(drop=True)
hist["ret_1d"] = hist["Close"].pct_change()
print(hist[["date", "Open", "High", "Low", "Close", "Volume", "ret_1d"]].tail(25).to_string(index=False))

print()
last = hist.iloc[-1]
print(f"  Last bar: {last['date'].date()}  close={last['Close']:.2f}  ret_1d={last['ret_1d']*100:+.2f}%")

# If predict was made 'as-of' day d, h+5 realised return uses close at d+5
asof_str = str(d.get("last_close_date", ""))
if asof_str:
    asof = pd.Timestamp(asof_str)
    after = hist[hist["date"] > asof]
    if not after.empty:
        c0 = float(hist[hist["date"] <= asof]["Close"].iloc[-1])
        for h in (1, 5, 21):
            target = after.iloc[h - 1] if len(after) >= h else None
            if target is not None:
                ret = (float(target["Close"]) / c0 - 1) * 100
                print(f"  realised ret h={h:>2d} ({target['date'].date()}): {ret:+.2f}%")
            else:
                print(f"  realised ret h={h:>2d}: not yet realised")
