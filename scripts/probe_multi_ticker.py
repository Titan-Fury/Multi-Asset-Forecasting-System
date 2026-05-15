"""Multi-ticker probe to verify min_conv=30 gate now emits signals.
Checks proba_up, conviction, and pass/block across a range of tickers."""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from omni.utils.logger import log
from omni.ondemand.cache import purge_ticker
from omni.ondemand.predict import predict_ticker
from omni.ondemand.universe import resolve

TICKERS = [
    # Large cap (Nifty)
    "RELIANCE", "HDFCBANK", "TCS", "INFY",
    # Midcap PSU
    "BANKBARODA", "TATAPOWER", "TATAMOTORS",
    # Smallcap retail
    "IRCTC", "RVNL", "SUZLON",
    # New age
    "ZOMATO", "PAYTM",
    # Watchlist
    "CIANAGRO",
]

print(f"{'ticker':<14s} {'proba_up':>8s} {'rubric':>7s} {'blended':>8s} {'conv':>5s} {'dir':>4s} {'gate':>6s}  reasons")
print("-" * 100)

n_pass, n_block, n_fail = 0, 0, 0
for t in TICKERS:
    purge_ticker(t)
    try:
        res = resolve(t)
        if res is None:
            print(f"{t:<14s} (resolve failed)")
            n_fail += 1
            continue
        card = predict_ticker(res.ticker, res.asset_class, display_name=res.display_name)
        if card is None:
            print(f"{t:<14s} (predict_ticker returned None)")
            n_fail += 1
            continue
        d = card.to_dict()
        p = float(d.get("proba_up", 0.5) or 0.5)
        rb = float(d.get("rubric_total", 50) or 50)
        bp = float(d.get("blended_p", 0.5) or 0.5)
        c = float(d.get("conviction", 0) or 0)
        di = int(d.get("direction", 0) or 0)
        g = bool(d.get("passes_gate", False))
        r = (d.get("reasons") or "ok")[:40]
        d_lbl = {1: "LONG", -1: "SHRT", 0: "FLAT"}[di]
        gate_lbl = "PASS" if g else "block"
        if g:
            n_pass += 1
        else:
            n_block += 1
        print(f"{t:<14s} {p:>8.4f} {rb:>7.1f} {bp:>8.4f} {c:>5.1f} {d_lbl:>4s} {gate_lbl:>6s}  {r}")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"probe {t} crashed: {exc}")
        n_fail += 1
        print(f"{t:<14s} ERROR: {exc}")

print("-" * 100)
print(f"summary: {n_pass} passed, {n_block} blocked, {n_fail} failed")
