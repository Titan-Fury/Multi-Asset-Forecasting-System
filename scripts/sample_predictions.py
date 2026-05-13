"""Quick sample of predictions through the retrained model.

Bypasses the on-disk snapshot files and calls predict_ticker on a
small basket directly. Reports proba_up distribution to verify the
retrain improved spread.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pandas as pd

from omni.ondemand.predict import predict_ticker

# Mix of large-cap India + US to stress the universal model.
# Keep small — each cold prediction takes 10-30s with news/fundamentals fetch.
TICKERS = [
    ("RELIANCE", "india"),
    ("ITC", "india"),
    ("AAPL", "us"),
    ("CIANAGRO", "india"),
]

print(f"Sampling {len(TICKERS)} predictions through retrained model...")
results = []
for ticker, ac in TICKERS:
    try:
        c = predict_ticker(ticker, ac)
    except Exception as e:
        print(f"  {ticker:<14s} ERROR: {e}")
        continue
    if c is None:
        print(f"  {ticker:<14s} no data")
        continue
    d = c.to_dict()
    results.append({
        "ticker": ticker,
        "asset_class": ac,
        "proba_up": d.get("proba_up"),
        "blended_p": d.get("blended_p"),
        "conviction": d.get("conviction"),
        "direction": d.get("direction"),
        "regime": d.get("regime"),
        "model_kind": d.get("model_kind"),
        "passes_gate": d.get("passes_gate"),
    })

df = pd.DataFrame(results)
print()
print("=" * 70)
print(f"  Sample: {len(df)} predictions")
print("=" * 70)
print(df.to_string(index=False))
print()
print("=" * 70)
print("  Distribution stats")
print("=" * 70)
for col in ("proba_up", "blended_p", "conviction"):
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            print(f"  {col:<14s}  n={len(s):2d}  "
                  f"min={s.min():>7.3f}  med={s.median():>7.3f}  "
                  f"max={s.max():>7.3f}  std={s.std():>6.4f}")
print()
print("=" * 70)
print("  Verdict")
print("=" * 70)
pu = pd.to_numeric(df["proba_up"], errors="coerce").dropna()
if not pu.empty:
    std = float(pu.std())
    if std < 0.02:
        print(f"  STILL COLLAPSED. proba_up std = {std:.4f}")
        print(f"  The retrain bumped AUC 0.497 -> 0.5117 but the model is")
        print(f"  still very close to coin-flip in absolute probability space.")
        print(f"  Features need stronger signal at 5d horizon, OR labeling")
        print(f"  needs refinement (purged CV may be over-conservative).")
    elif std < 0.05:
        print(f"  IMPROVED. proba_up std = {std:.4f} (was 0.011).")
        print(f"  Model has slight directional edge now.")
    else:
        print(f"  HEALTHY. proba_up std = {std:.4f}.")
n_passing = int(df["passes_gate"].sum()) if "passes_gate" in df.columns else 0
print(f"  Gate emit rate: {n_passing}/{len(df)} = {100*n_passing/max(1,len(df)):.0f}%")
