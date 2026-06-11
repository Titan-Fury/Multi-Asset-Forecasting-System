"""Widen the US delisted set for a tighter survivorship estimate.

Only 55/150 delisted names survived Tiingo in Phase 28.24 (~37%), because many
Polygon delisted symbols are obscure/old and absent from Tiingo. Here we bias the
sample toward RECENT delistings (delisted_utc >= 2019), which Tiingo covers far
better, fetch their history via Tiingo, and APPEND to the existing panel.
"""
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

env = {}
for l in open(".env"):
    if re.match(r"^[A-Za-z]", l):
        k, _, v = l.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
PKEY = env["POLYGON_API_KEY"]; TOK = env["TIINGO_API_KEY"]
START, END, MIN_ROWS = "2014-01-01", "2026-06-01", 300


def page(url):
    if not url.startswith("https://"):
        raise ValueError("refusing non-https url")
    for _ in range(6):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:  # nosec B310 - https-only, hardcoded Polygon host
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(13); continue
            raise
    return {}


def collect_recent_delisted(max_pages=8, since="2019-01-01"):
    out, url = [], (f"https://api.polygon.io/v3/reference/tickers?market=stocks&type=CS"
                    f"&active=false&limit=1000&apiKey={PKEY}")
    for _ in range(max_pages):
        b = page(url)
        for s in b.get("results", []):
            t, du = s.get("ticker", ""), (s.get("delisted_utc") or "")
            if re.match(r"^[A-Z]{1,5}$", t) and du >= since:
                out.append(t)
        nxt = b.get("next_url")
        if not nxt:
            break
        url = nxt + f"&apiKey={PKEY}"
        time.sleep(13)
    return sorted(set(out))


def fetch_tiingo(tk):
    url = (f"https://api.tiingo.com/tiingo/daily/{tk}/prices"
           f"?startDate={START}&endDate={END}&token={TOK}")
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    for _ in range(20):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:  # nosec B310 - https-only, hardcoded Tiingo host
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(60); continue  # wait out the hourly cap
            if e.code == 404:
                return None
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None


existing = pd.read_parquet("data/lake/us_pit_prices.parquet")
have = set(existing["ticker"].unique())

# Resumable checkpoint: accumulate new delisted names here, append every batch,
# skip anything already fetched so the job can be stopped/restarted freely
# (Tiingo free tier throttles hard once the hourly cap is hit).
CKPT = Path("data/lake/us_delisted_extra.parquet")
done = set()
if CKPT.exists():
    prev = pd.read_parquet(CKPT)
    done = set(prev["ticker"].unique())
    print(f"checkpoint: {len(done)} delisted already fetched", flush=True)

cand = [t for t in collect_recent_delisted() if t not in have and t not in done]
random.seed(11)
random.shuffle(cand)
cand = cand[:300]
print(f"recent-delisted candidates to fetch this run: {len(cand)}", flush=True)


def flush(frames):
    if not frames:
        return
    batch = pd.concat(frames, ignore_index=True).dropna(subset=["adj_close"])
    if CKPT.exists():
        batch = pd.concat([pd.read_parquet(CKPT), batch], ignore_index=True)
    batch.to_parquet(CKPT, index=False)


frames, kept, miss, short = [], [], [], []
for i, tk in enumerate(cand, 1):
    px = fetch_tiingo(tk)
    if not px or not isinstance(px, list):
        miss.append(tk)
    elif len(px) < MIN_ROWS:
        short.append(tk)
    else:
        df = pd.DataFrame(px)
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(df["date"]).dt.tz_localize(None),
            "adj_close": pd.to_numeric(df["adjClose"], errors="coerce"),
            "dollar_vol": pd.to_numeric(df["adjClose"], errors="coerce")
                          * pd.to_numeric(df["adjVolume"], errors="coerce"),
            "ticker": tk, "delisted": True,
        }))
        kept.append(tk)
    if i % 10 == 0:
        flush(frames); frames = []  # checkpoint
        print(f"[{i}/{len(cand)}] kept={len(kept)} miss={len(miss)} short={len(short)} (saved)", flush=True)
    time.sleep(0.8)

flush(frames)
extra = pd.read_parquet(CKPT)
print(f"DONE this run kept={len(kept)} miss={len(miss)} short={len(short)}")
print(f"checkpoint total: {extra['ticker'].nunique()} delisted names, {len(extra):,} rows")
