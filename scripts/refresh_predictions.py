"""Refresh today's prediction snapshot with the new 6-pillar rubric.

Wraps a focused subset of the daily orchestrator: scores the existing
universe through `predict_ticker` and writes the result to
`data/lake/predictions/<today>.parquet` so the dashboard picks up the
new pillar values without waiting for the full ingest/feature/train
pipeline.

This is the FAST path. The full daily orchestrator
(`omni.orchestrator.daily.run_daily`) does ingest + feature build +
walk-forward train + backtest + persist; that's a 30+ minute run.
This script skips everything except the predict-and-persist legs.

Run:
    python scripts/refresh_predictions.py
    python scripts/refresh_predictions.py --asset-classes india,us
    python scripts/refresh_predictions.py --max-tickers 20
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-classes", default="india,us",
                    help="comma-separated asset classes to score")
    ap.add_argument("--max-tickers", type=int, default=0,
                    help="0 = all; otherwise limit per asset class")
    ap.add_argument("--skip-write", action="store_true",
                    help="run scoring but don't write the snapshot")
    args = ap.parse_args()

    from omni.utils.env import universe as _universe
    from omni.ondemand.predict import predict_ticker

    cfg = _universe()
    target_classes = set(args.asset_classes.split(","))

    rows = []
    fail = 0
    started = time.time()
    for asset_class, spec in cfg["asset_classes"].items():
        if asset_class not in target_classes:
            continue
        tickers = list(spec["tickers"])
        if args.max_tickers:
            tickers = tickers[:args.max_tickers]
        for i, ticker in enumerate(tickers):
            t0 = time.time()
            try:
                card = predict_ticker(ticker, asset_class)
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"  [{i+1:3d}] {asset_class}/{ticker:<14s} FAIL {type(exc).__name__}: {str(exc)[:60]}")
                continue
            if card is None:
                fail += 1
                print(f"  [{i+1:3d}] {asset_class}/{ticker:<14s} no data")
                continue
            d = card.to_dict()
            pillars = d.get("pillars") or {}
            rows.append({
                "asof": d.get("asof"),
                "ticker": d.get("ticker"),
                "asset_class": d.get("asset_class"),
                "regime": d.get("regime"),
                "proba_up": d.get("proba_up"),
                "blended_p": d.get("blended_p"),
                "conviction": d.get("conviction"),
                "direction": d.get("direction"),
                "passes_gate": d.get("passes_gate"),
                "expected_move_pct": d.get("expected_move_pct"),
                "data_quality_score": d.get("data_quality_score"),
                "horizon_days": d.get("horizon_days", 5),
                "model_kind": d.get("model_kind"),
                "model_trained_at": d.get("model_trained_at"),
                # NEW: per-pillar columns so the snapshot is fully introspectable
                "technical_score": pillars.get("technical"),
                "fundamental_score": pillars.get("fundamental"),
                "sentiment_score": pillars.get("sentiment"),
                "macro_score": pillars.get("macro"),
                "microstructure_score": pillars.get("microstructure"),
                "oi_score": pillars.get("oi"),
                "rubric_total": d.get("rubric_total"),
                "logged_at": datetime.now(timezone.utc).isoformat(),
            })
            elapsed = time.time() - t0
            conv = float(d.get("conviction", 0) or 0)
            dirn = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(int(d.get("direction", 0)), "?")
            print(f"  [{i+1:3d}] {asset_class}/{ticker:<14s} "
                  f"p={d.get('proba_up'):.3f} conv={conv:>5.1f} {dirn:<5s} "
                  f"({elapsed:.1f}s)")

    total_time = time.time() - started
    print()
    print(f"=== Done. {len(rows)} predictions, {fail} failures in {total_time:.1f}s ===")
    if not rows:
        print("No predictions to write.")
        return 1

    df = pd.DataFrame(rows)
    print()
    print("=== Distribution stats ===")
    for col in ("proba_up", "blended_p", "conviction"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                print(f"  {col:<20s} n={len(s):3d}  "
                      f"min={s.min():>7.3f}  med={s.median():>7.3f}  "
                      f"max={s.max():>7.3f}  std={s.std():>6.4f}")
    print()
    n_pass = int(df["passes_gate"].sum()) if "passes_gate" in df.columns else 0
    print(f"  Gate emit rate: {n_pass}/{len(df)} = {100*n_pass/max(1,len(df)):.0f}%")

    if args.skip_write:
        print()
        print("--skip-write set; not persisting.")
        return 0

    out_dir = Path("data/lake/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = out_dir / f"{today}.parquet"
    # Use the hardened atomic-write that validates before swap.
    from omni.orchestrator.persist import _atomic_write_parquet
    _atomic_write_parquet(df, out_path)
    print()
    print(f"=== Wrote {out_path} ({out_path.stat().st_size:,} bytes) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
