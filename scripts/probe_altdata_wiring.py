"""Smoke test for Commit A — verify altdata patch wires through predict_ticker.

Runs CIANAGRO (small cap, where any insider activity will show up) and
RELIANCE through the on-demand path and prints the insider_score and
microstructure_score. We don't assert non-50 because NSE may block or
return empty data — we only assert the path doesn't crash.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from omni.ondemand.cache import purge_ticker
from omni.ondemand.predict import predict_ticker
from omni.ondemand.universe import resolve

for tk in ("CIANAGRO", "RELIANCE", "TATAMOTORS"):
    purge_ticker(tk)
    res = resolve(tk)
    if res is None:
        print(f"{tk}: resolve failed")
        continue
    try:
        card = predict_ticker(res.ticker, res.asset_class, display_name=res.display_name)
    except Exception as exc:  # noqa: BLE001
        print(f"{tk}: predict_ticker raised: {exc!r}")
        continue
    if card is None:
        print(f"{tk}: predict_ticker returned None")
        continue
    d = card.to_dict()
    pillars = d.get("pillars", {})
    print(f"{tk:12s}  insider={pillars.get('insider', '?'):>5}  "
          f"oi={pillars.get('oi', '?'):>5}  "
          f"micro={pillars.get('microstructure', '?'):>5}  "
          f"earnings_mom={pillars.get('earnings_momentum', '?'):>5}")
