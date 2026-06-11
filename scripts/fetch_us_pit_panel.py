import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

env = {}
for l in open(".env"):
    if re.match(r"^[A-Za-z]", l):
        k, _, v = l.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
TOK = env["TIINGO_API_KEY"]

import yaml

cfg = yaml.safe_load(open("configs/universe_us_pit.yaml"))
tickers = cfg["tickers"]
delisted = set(cfg.get("delisted", []))  # names to route to Tiingo
START, END, MIN_ROWS = "2014-01-01", "2026-06-01", 300


def fetch_yf(tk):
    """Fast keyless source for still-listed names."""
    try:
        df = yf.download(tk, start=START, end=END, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        ac = df["Adj Close"]; vol = df["Volume"]
        ac = ac.iloc[:, 0] if hasattr(ac, "columns") else ac
        vol = vol.iloc[:, 0] if hasattr(vol, "columns") else vol
        return pd.DataFrame({
            "date": pd.to_datetime(df["Date"]).dt.tz_localize(None),
            "adj_close": pd.to_numeric(ac, errors="coerce"),
            "dollar_vol": pd.to_numeric(ac, errors="coerce") * pd.to_numeric(vol, errors="coerce"),
            "ticker": tk,
        })
    except Exception:
        return None


def fetch_tiingo(tk):
    url = (f"https://api.tiingo.com/tiingo/daily/{tk}/prices"
           f"?startDate={START}&endDate={END}&token={TOK}")
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:  # nosec B310 - https-only, hardcoded Tiingo host
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(20); continue
            if e.code == 404:
                return None
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None


frames, kept, miss, short = [], [], [], []
for i, tk in enumerate(tickers, 1):
    is_del = tk in delisted
    if is_del:
        px = fetch_tiingo(tk)
        sub = None
        if px and isinstance(px, list) and len(px) >= MIN_ROWS:
            df = pd.DataFrame(px)
            sub = pd.DataFrame({
                "date": pd.to_datetime(df["date"]).dt.tz_localize(None),
                "adj_close": pd.to_numeric(df["adjClose"], errors="coerce"),
                "dollar_vol": pd.to_numeric(df["adjClose"], errors="coerce")
                              * pd.to_numeric(df["adjVolume"], errors="coerce"),
                "ticker": tk,
            })
        time.sleep(0.8)  # only throttle the Tiingo path
    else:
        sub = fetch_yf(tk)  # active -> fast keyless

    if sub is None or sub.empty:
        miss.append(tk)
    elif len(sub) < MIN_ROWS:
        short.append(tk)
    else:
        sub["delisted"] = is_del
        frames.append(sub); kept.append(tk)
    if i % 25 == 0:
        print(f"[{i}/{len(tickers)}] kept={len(kept)} miss={len(miss)} short={len(short)}", flush=True)

panel = pd.concat(frames, ignore_index=True).dropna(subset=["adj_close"])
Path("data/lake").mkdir(parents=True, exist_ok=True)
panel.to_parquet("data/lake/us_pit_prices.parquet", index=False)
print(f"KEPT={len(kept)} MISS={len(miss)} SHORT={len(short)} ROWS={len(panel)}")
print(f"range: {panel['date'].min().date()}..{panel['date'].max().date()}")
print("miss sample:", miss[:20])
