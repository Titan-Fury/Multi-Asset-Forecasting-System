"""Verify circuit-bar detection against actual CIANAGRO history.

CIANAGRO had ~22 consecutive +5% upper-circuit days followed by 5
consecutive -5% lower-circuit days. The detector should flag all of
them (they have daily |ret| ~5% and zero intraday range).
"""
import pandas as pd
import yfinance as yf

from omni.features.circuit import detect_circuit_bars, mask_circuit_labels

# Pull live CIANAGRO history
yt = yf.Ticker("CIANAGRO.NS")
h = yt.history(period="3mo", auto_adjust=False).reset_index()
if h.empty:
    yt = yf.Ticker("CIANAGRO.BO")
    h = yt.history(period="3mo", auto_adjust=False).reset_index()
if h.empty:
    print("yfinance returned empty for CIANAGRO; aborting")
    raise SystemExit(1)

# Normalise columns to match build.py's lowercase OHLC schema
h.columns = [c.lower() for c in h.columns]
h = h.rename(columns={"date": "date"})
h["adj_close"] = h.get("adj close", h["close"])
h = h.sort_values("date").reset_index(drop=True)
print(f"CIANAGRO bars: {len(h)}, range {h['date'].iloc[0].date()} -> {h['date'].iloc[-1].date()}")

is_circ = detect_circuit_bars(h, asset_class="india")
print(f"\ndetected {int(is_circ.sum())} circuit-locked bars out of {len(h)}")

# Show the detected ones
flagged = h.loc[is_circ, ["date", "open", "high", "low", "close"]].copy()
flagged["ret"] = (flagged["close"] / h["close"].shift(1)).reindex(flagged.index) - 1
flagged["range_pct"] = (flagged["high"] - flagged["low"]) / flagged["open"]
print("\nflagged bars (last 30):")
print(flagged.tail(30).to_string(index=False,
                                    formatters={"ret": "{:+.4f}".format,
                                                 "range_pct": "{:.4f}".format}))

# Synthesize a fwd_dir_5d label and verify mask_circuit_labels nulls them
import numpy as np
h["fwd_dir_5d"] = np.sign(h["close"].pct_change(5).shift(-5)).astype(float)
n_before = h["fwd_dir_5d"].notna().sum()
masked = mask_circuit_labels(h, horizons=(1, 5, 21), asset_class="india")
n_after = masked["fwd_dir_5d"].notna().sum()
print(f"\nfwd_dir_5d non-null: {n_before} -> {n_after} (dropped {n_before - n_after} circuit rows)")

# Sanity: detect_circuit_bars on a US ticker should be all-False
print("\nUS ticker (AAPL) sanity:")
hu = yf.Ticker("AAPL").history(period="1mo", auto_adjust=False).reset_index()
hu.columns = [c.lower() for c in hu.columns]
hu["adj_close"] = hu.get("adj close", hu["close"])
print(f"  detected circuit bars (asset_class='us'): "
      f"{int(detect_circuit_bars(hu, asset_class='us').sum())} (expected 0)")
print(f"  detected circuit bars (asset_class='india' on AAPL): "
      f"{int(detect_circuit_bars(hu, asset_class='india').sum())} (expected ~0)")
