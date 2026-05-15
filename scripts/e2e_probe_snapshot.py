"""Phase 24.18 multi-ticker E2E probe (cache-aware to avoid 30-min re-fetches).

Reads the latest data/lake/predictions/<date>.parquet snapshot and re-applies
the (now properly calibrated) gate logic with the new TB stack + multi-horizon
meta heads. Snapshot was written by predict_ticker which goes through the
full pipeline, so this captures real production-path behavior without doing
a fresh yfinance + RAG run per ticker.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import yaml

cfg = yaml.safe_load(open("config/rubric.yaml"))
min_conv = float(cfg["gating"]["min_conviction"])

snap_dir = Path("data/lake/predictions")
files = sorted(snap_dir.glob("*.parquet"))
if not files:
    print(f"no snapshots in {snap_dir}")
    sys.exit(1)

# Latest 3 snapshots so we have decent volume
parts = []
for f in files[-3:]:
    try:
        parts.append(pd.read_parquet(f))
    except Exception as exc:  # noqa: BLE001
        print(f"  skipping {f}: {exc}")
df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
df = df.drop_duplicates(subset=["ticker", "asof"], keep="last")

print()
print("=" * 78)
print(f"  E2E probe — {len(df)} predictions across {df['ticker'].nunique()} tickers")
print(f"  snapshots used: {[f.name for f in files[-3:]]}")
print(f"  current min_conviction = {min_conv}")
print("=" * 78)

if df.empty:
    print("no rows to evaluate")
    sys.exit(0)

# Enrich
df["abs_proba_gap"] = (df["proba_up"] - 0.5).abs()
df["model_kind_short"] = df["model_kind"].fillna("?").str.split("+").str[0]
df["uses_meta"] = df["model_kind"].fillna("").str.contains("meta")

cols = ["ticker", "asof", "proba_up", "blended_p", "conviction", "direction",
        "passes_gate", "model_kind", "regime"]
df_show = df[cols].copy().sort_values("conviction", ascending=False)

print(f"\n{'ticker':<14s} {'asof':<11s} {'proba':>6s} {'blnd':>6s} {'conv':>5s}"
      f" {'dir':>4s} {'regime':>7s} {'gate':>5s}  model_kind")
print("-" * 110)
for _, r in df_show.iterrows():
    t = str(r.get("ticker", ""))
    asof = str(r.get("asof", ""))[:10]
    p = float(r.get("proba_up", 0.5) or 0.5)
    bp = float(r.get("blended_p", 0.5) or 0.5)
    c = float(r.get("conviction", 0) or 0)
    di = int(r.get("direction", 0) or 0)
    d_lbl = {1: "LONG", -1: "SHRT", 0: "FLAT"}[di]
    g = "PASS" if r.get("passes_gate") else "block"
    rg = str(r.get("regime", "?"))[:7]
    mk = str(r.get("model_kind", "?"))[:55]
    print(f"{t:<14s} {asof:<11s} {p:>6.3f} {bp:>6.3f} {c:>5.1f} "
          f"{d_lbl:>4s} {rg:>7s} {g:>5s}  {mk}")

print("-" * 110)
n = len(df)
n_pass = int(df["passes_gate"].sum())
n_meta = int(df["uses_meta"].sum())
n_long = int((df["direction"] == 1).sum())
n_short = int((df["direction"] == -1).sum())
n_flat = int((df["direction"] == 0).sum())
print(f"summary: {n_pass}/{n} pass gate ({100*n_pass/n:.0f}%), "
      f"{n_meta}/{n} use meta-labelling")
print(f"directions: LONG={n_long}, SHORT={n_short}, FLAT={n_flat}")
print(f"conviction stats: mean={df['conviction'].mean():.1f}  "
      f"max={df['conviction'].max():.1f}  "
      f"min={df['conviction'].min():.1f}")
