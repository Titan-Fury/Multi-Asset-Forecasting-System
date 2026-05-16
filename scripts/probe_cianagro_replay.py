"""Historical replay: what would the model have said at CIANAGRO's May 8 close?

We can't roll the wall clock, but we can build the full feature frame
from the live bundle and then score the row whose date <= 2026-05-08.
That's a leakage-free replay because:
  - features for date d use only data <= d (technicals, fundamentals
    join as-of, sentiment shifted +1 BD, etc.)
  - we never touch fwd_* labels (those are dropped before scoring)
  - the model itself was trained on 2026-05-14, so it has NO knowledge
    of the May 11-15 crash. Scoring the May 8 row is exactly the same
    operation the system would have run if invoked on May 8 close.

Live alt-data (insider z, FII/DII z, block deals) come from TODAY's
sources — but those source pulls run via patch_altdata only when the
field is NaN on the live row. For a May 8 replay we explicitly skip
the altdata patch so we're scoring the May 8 row exactly as it would
have been seen at the time.
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

from omni.ensemble.pipeline import assemble_signals
from omni.features.build import is_target_col
from omni.models.xgboost_5d import _DROP_ALWAYS
from omni.ondemand.cache import purge_ticker
from omni.ondemand.lookup import lookup_ticker
from omni.ondemand.predict import (
    _align_to_universal, _cached_models, _last_row_features, _score_one,
    _model_version_tag,
)
from omni.ondemand.universe import resolve
from omni.rubric.scorer import score_row

ASOF_LIST = ["2026-05-08", "2026-05-15"]
TK = "CIANAGRO"

purge_ticker(TK)
res = resolve(TK)
bundle = lookup_ticker(res.ticker, res.asset_class, display_name=res.display_name)
if bundle is None:
    print("lookup failed")
    sys.exit(1)

feats = _last_row_features(bundle)
feats["date"] = pd.to_datetime(feats["date"]).dt.tz_localize(None)
print(f"feature frame shape: {feats.shape}, date range: "
      f"{feats['date'].min().date()} -> {feats['date'].max().date()}")

universal, stack_rec, _cf, _cm, meta_head, meta_heads, regime_stacks = \
    _cached_models(_model_version_tag())

regime_proba = (bundle.regime.proba_today
                  if bundle.regime is not None else None)
regime_label = (bundle.regime.label_today
                  if bundle.regime is not None else None)

print(f"regime today: label={regime_label}  proba={regime_proba}")
print()
print(f"{'asof':<12s} {'close':>10s} {'p_up':>7s} {'blended':>8s} "
      f"{'conv':>6s} {'dir':>4s} {'tech':>5s} {'sent':>5s} {'rs':>4s} "
      f"{'em':>4s} {'kind':<32s}")
print("-" * 110)

for asof_str in ASOF_LIST:
    asof = pd.Timestamp(asof_str)
    sub = feats[feats["date"] <= asof]
    if sub.empty:
        print(f"{asof_str}: no rows on or before this date")
        continue
    last = sub.tail(1)
    feat_only = last[[c for c in last.columns
                       if c not in _DROP_ALWAYS and not is_target_col(c)]]
    feat_only = feat_only.apply(pd.to_numeric, errors="coerce")

    proba, kind = _score_one(
        feat_only,
        universal=universal, stack_rec=stack_rec,
        asset_class=bundle.asset_class, meta_head=meta_head,
        regime_stacks=regime_stacks,
        regime_proba=regime_proba, regime_label=regime_label,
    )

    rgl = regime_label or "chop"
    asof_ts = pd.to_datetime(last.iloc[0]["date"])
    sig_df = assemble_signals(
        last,
        proba_up=pd.Series([float(proba)], index=last.index),
        regime_override=pd.Series({asof_ts: rgl}),
    )
    if sig_df.empty:
        print(f"{asof_str}: pipeline produced no rows")
        continue
    row = sig_df.iloc[0]

    feat_row = last.iloc[0].copy()
    rub = score_row(feat_row, regime=rgl, asset_class=bundle.asset_class)

    close = float(last.iloc[0].get("adj_close", 0.0))
    print(f"{asof_str:<12s} {close:>10.2f} {proba:>7.3f} "
          f"{float(row.get('blended_p', 0.5)):>8.3f} "
          f"{float(row.get('conviction', 0.0)):>6.2f} "
          f"{int(row.get('direction', 0)):>4d} "
          f"{rub.technical_score:>5.0f} "
          f"{rub.sentiment_score:>5.0f} "
          f"{rub.relative_strength_score:>4.0f} "
          f"{rub.earnings_momentum_score:>4.0f} "
          f"{kind:<32s}")

print()
print("Realised price after May 8 (from yfinance):")
import yfinance as yf
yt = yf.Ticker(f"{TK}.NS") if not yf.Ticker(f"{TK}.NS").history(period="5d", auto_adjust=False).empty else yf.Ticker(f"{TK}.BO")
hist = yt.history(period="3mo", auto_adjust=False).reset_index()
hist["date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
hist = hist.sort_values("date")
c_may8 = float(hist[hist["date"] == "2026-05-08"]["Close"].iloc[0])
for h, lbl in ((1, "h+1"), (5, "h+5"), (21, "h+21")):
    after = hist[hist["date"] > "2026-05-08"]
    if len(after) >= h:
        target = after.iloc[h - 1]
        ret = (float(target["Close"]) / c_may8 - 1) * 100
        print(f"  {lbl} ({target['date'].date()}): "
              f"close={float(target['Close']):.2f}  ret={ret:+.2f}%")
    else:
        print(f"  {lbl}: not enough realised data ({len(after)} sessions after May 8)")
