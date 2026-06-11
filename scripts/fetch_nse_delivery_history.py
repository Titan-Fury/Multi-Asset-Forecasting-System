"""Ingest multi-day NSE delivery% history from the public NSE archive.

Downloads sec_bhavdata_full_<DDMMYYYY>.csv for each weekday in the window
(archive depth ~2020+), parses EQ rows, and stores a compact panel
[date, ticker, deliv_per, turnover_lacs] -> data/lake/nse_delivery_history.parquet.

Public archive, browser UA only (no key). Weekends/holidays 404 and are skipped.
"""
import io
import time
import urllib.request
import urllib.error
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
op = urllib.request.build_opener()
op.addheaders = [("User-Agent", UA), ("Accept", "*/*")]
BASE = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"

START, END = date(2020, 1, 1), date(2026, 6, 1)

rows = []
d, n_ok, n_miss = START, 0, 0
while d <= END:
    if d.weekday() < 5:  # Mon-Fri
        url = BASE.format(d=d.strftime("%d%m%Y"))
        try:
            raw = op.open(url, timeout=30).read()
            df = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            eq = df[df["SERIES"] == "EQ"]
            sub = pd.DataFrame({
                "date": pd.to_datetime(d),
                "ticker": eq["SYMBOL"].astype(str).str.strip(),
                "deliv_per": pd.to_numeric(eq["DELIV_PER"], errors="coerce"),
                "turnover_lacs": pd.to_numeric(eq["TURNOVER_LACS"], errors="coerce"),
            })
            rows.append(sub)
            n_ok += 1
        except urllib.error.HTTPError:
            n_miss += 1
        except Exception as e:  # noqa: BLE001
            print(f"  parse err {d}: {type(e).__name__} {str(e)[:50]}", flush=True)
        time.sleep(0.15)
    d += timedelta(days=1)
    if (n_ok + n_miss) % 100 == 0 and (n_ok + n_miss) > 0:
        print(f"...{d} ok={n_ok} miss={n_miss} rows={sum(len(r) for r in rows):,}", flush=True)

panel = pd.concat(rows, ignore_index=True).dropna(subset=["deliv_per"])
Path("data/lake").mkdir(parents=True, exist_ok=True)
panel.to_parquet("data/lake/nse_delivery_history.parquet", index=False)
print(f"DONE days_ok={n_ok} days_miss={n_miss} rows={len(panel):,} "
      f"names={panel['ticker'].nunique()} {panel['date'].min().date()}..{panel['date'].max().date()}")
