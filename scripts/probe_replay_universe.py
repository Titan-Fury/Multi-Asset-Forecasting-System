"""Multi-ticker May 8 replay to see verdict + model behaviour breadth.

For each ticker, builds the live feature frame, scores the row at
asof = 2026-05-08 close, prints the verdict banner.
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
import yfinance as yf

from omni.dashboard.verdict import verdict
from omni.ensemble.pipeline import assemble_signals
from omni.features.build import is_target_col
from omni.models.xgboost_5d import _DROP_ALWAYS
from omni.ondemand.cache import purge_ticker
from omni.ondemand.lookup import lookup_ticker
from omni.ondemand.predict import (
    _cached_models, _last_row_features, _model_version_tag, _score_one,
)
from omni.ondemand.universe import resolve
from omni.rubric.scorer import score_row

ASOF = pd.Timestamp("2026-05-08")
TICKERS = ["CIANAGRO", "RELIANCE", "TCS", "HDFCBANK", "TATAMOTORS",
            "IRCTC", "ZOMATO", "PAYTM", "BANKBARODA"]

universal, stack_rec, _cf, _cm, meta_head, meta_heads, regime_stacks = \
    _cached_models(_model_version_tag())

print(f"{'ticker':<12s} {'p_up':>5s} {'conv':>5s} {'dir':>4s} "
      f"{'h+5_real%':>10s}  verdict")
print("-" * 110)


def _realised_h5(ticker: str) -> float | None:
    for sfx in (".NS", ".BO"):
        try:
            h = yf.Ticker(ticker + sfx).history(period="3mo", auto_adjust=False)
            if h.empty:
                continue
            h = h.reset_index()
            h["date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None)
            h = h.sort_values("date")
            base = h[h["date"] == ASOF]
            after = h[h["date"] > ASOF]
            if base.empty or len(after) < 5:
                continue
            return (float(after.iloc[4]["Close"]) / float(base.iloc[0]["Close"]) - 1) * 100
        except Exception:
            continue
    return None


for tk in TICKERS:
    purge_ticker(tk)
    res = resolve(tk)
    if res is None:
        print(f"{tk:<12s} resolve failed")
        continue
    try:
        bundle = lookup_ticker(res.ticker, res.asset_class, display_name=res.display_name)
    except Exception as exc:  # noqa: BLE001
        print(f"{tk:<12s} lookup raised {type(exc).__name__}")
        continue
    if bundle is None:
        print(f"{tk:<12s} no bundle")
        continue
    feats = _last_row_features(bundle)
    feats["date"] = pd.to_datetime(feats["date"]).dt.tz_localize(None)
    sub = feats[feats["date"] <= ASOF]
    if sub.empty:
        print(f"{tk:<12s} no rows on or before {ASOF.date()}")
        continue
    last = sub.tail(1)
    feat_only = last[[c for c in last.columns
                       if c not in _DROP_ALWAYS and not is_target_col(c)]]
    feat_only = feat_only.apply(pd.to_numeric, errors="coerce")
    regime_proba = bundle.regime.proba_today if bundle.regime is not None else None
    regime_label = bundle.regime.label_today if bundle.regime is not None else None
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
        print(f"{tk:<12s} no signal row")
        continue
    row = sig_df.iloc[0]
    rub = score_row(last.iloc[0].copy(), regime=rgl, asset_class=bundle.asset_class)
    pillars = {
        "technical": rub.technical_score,
        "fundamental": rub.fundamental_score,
        "sentiment": rub.sentiment_score,
        "macro": rub.macro_score,
        "microstructure": getattr(rub, "microstructure_score", 50.0),
        "oi": getattr(rub, "oi_score", 50.0),
        "relative_strength": getattr(rub, "relative_strength_score", 50.0),
        "insider": getattr(rub, "insider_score", 50.0),
        "earnings_momentum": getattr(rub, "earnings_momentum_score", 50.0),
    }
    card = {
        "proba_up": proba,
        "blended_p": float(row.get("blended_p", 0.5)),
        "conviction": float(row.get("conviction", 0.0)),
        "direction": int(row.get("direction", 0)),
        "passes_gate": bool(row.get("passes_gate", False)),
        "regime": rgl,
        "model_kind": kind,
        "min_conviction": 15.0,
        "pillars": pillars,
    }
    v = verdict(card)
    h5 = _realised_h5(tk)
    h5_str = f"{h5:+.2f}" if h5 is not None else "n/a"
    print(f"{tk:<12s} {proba:>5.2f} {card['conviction']:>5.1f} "
          f"{card['direction']:>4d} {h5_str:>10s}  "
          f"[{v.tone:<5s}] {v.headline}")
    print(f"             {v.summary}")
    if v.disagreement:
        print(f"             {v.disagreement}")
    print()
