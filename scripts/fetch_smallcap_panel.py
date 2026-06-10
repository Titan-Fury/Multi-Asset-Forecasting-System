import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni.ingest.prices import fetch_prices  # noqa: E402

cfg = yaml.safe_load(open("configs/universe_smallcap.yaml"))
tickers = cfg["tickers"]
ac = cfg.get("asset_class", "india")
START, END, MIN_ROWS = "2014-01-01", "2026-06-01", 300

frames, kept, dropped, failed = [], [], [], []
for i, tk in enumerate(tickers, 1):
    try:
        df = fetch_prices(ac, tk, start=START, end=END)
        if df.empty or len(df) < MIN_ROWS:
            dropped.append((tk, 0 if df.empty else len(df)))
            continue
        sub = df[["date", "adj_close"]].copy()
        sub["ticker"] = tk
        frames.append(sub)
        kept.append(tk)
    except Exception as e:  # noqa: BLE001
        failed.append((tk, str(e)[:60]))
    if i % 20 == 0:
        logger.info(f"[{i}/{len(tickers)}] kept={len(kept)} dropped={len(dropped)} failed={len(failed)}")

panel = pd.concat(frames, ignore_index=True).dropna(subset=["adj_close"])
out = Path("data/lake/smallcap_prices.parquet")
panel.to_parquet(out, index=False)
print(f"KEPT={len(kept)} DROPPED={len(dropped)} FAILED={len(failed)} ROWS={len(panel)}")
print(f"date range: {panel['date'].min()} .. {panel['date'].max()}")
print("dropped (short):", dropped[:15])
print("failed:", failed[:15])
