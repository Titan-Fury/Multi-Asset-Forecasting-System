"""Data quality probe (Phase 24.16) — answers "is my data noisy?".

Walks the existing prediction snapshots and feature artifacts to surface:
  - row counts and date coverage per snapshot
  - distribution of `proba_up`, `conviction`, `data_quality_score`
  - regime label distribution
  - missing/NaN proportion in critical columns
  - day-over-day stability (do the same tickers stay around the same conviction?)

Run:
    python scripts/check_data_quality.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _describe(s: pd.Series, name: str) -> None:
    """Print a one-line summary stats line."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        print(f"  {name:<24}  EMPTY")
        return
    print(
        f"  {name:<24}  n={len(s):4d}  "
        f"min={s.min():>7.3f}  p25={s.quantile(0.25):>7.3f}  "
        f"med={s.median():>7.3f}  p75={s.quantile(0.75):>7.3f}  "
        f"max={s.max():>7.3f}  std={s.std():>6.3f}"
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pred_dir = root / "data" / "lake" / "predictions"
    files = sorted(pred_dir.glob("*.parquet"))
    if not files:
        print(f"No prediction snapshots in {pred_dir}", file=sys.stderr)
        return 1

    _section(f"Prediction snapshots in {pred_dir}")
    print(f"  found {len(files)} files")
    for f in files[-5:]:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}  {size_kb:.1f} KB")

    # Load all into one frame for cross-day comparison
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["__snapshot"] = f.stem
            frames.append(df)
        except Exception as e:  # noqa: BLE001
            print(f"  failed to read {f.name}: {e}")
    if not frames:
        return 1
    big = pd.concat(frames, ignore_index=True)

    _section("Schema")
    print(f"  rows total: {len(big)}")
    print(f"  columns:    {sorted(c for c in big.columns if c != '__snapshot')}")

    # Latest snapshot summary
    latest = frames[-1]
    _section(f"Latest snapshot ({files[-1].name})")
    print(f"  rows: {len(latest)}")
    if "ticker" in latest.columns:
        print(f"  unique tickers: {latest['ticker'].nunique()}")
        print(f"  asset classes:  "
              f"{latest['asset_class'].value_counts().to_dict() if 'asset_class' in latest else 'n/a'}")
    if "regime" in latest.columns:
        print(f"  regime mix:     {latest['regime'].value_counts().to_dict()}")
    if "passes_gate" in latest.columns:
        emit_rate = float(latest["passes_gate"].mean()) * 100
        print(f"  gate emit rate: {emit_rate:.1f}% "
              f"({int(latest['passes_gate'].sum())} of {len(latest)})")

    _section("Distribution stats (latest snapshot)")
    for col in ("proba_up", "blended_p", "conviction",
                 "expected_move_pct", "realized_sharpe_1y",
                 "data_quality_score"):
        if col in latest.columns:
            _describe(latest[col], col)
        else:
            print(f"  {col:<24}  ABSENT")

    # Missing data audit
    _section("Missing / NaN / Zero audit (latest snapshot)")
    critical = [c for c in
                ("proba_up", "blended_p", "conviction",
                 "expected_move_pct", "regime", "ticker", "direction")
                if c in latest.columns]
    for c in critical:
        nan = int(latest[c].isna().sum())
        if pd.api.types.is_numeric_dtype(latest[c]):
            zero = int((pd.to_numeric(latest[c], errors="coerce") == 0).sum())
        else:
            zero = int((latest[c].astype(str) == "").sum())
        print(f"  {c:<24}  NaN={nan:4d}  zero/empty={zero:4d}  "
              f"({100*nan/len(latest):.1f}% / {100*zero/len(latest):.1f}%)")

    # Cross-day stability — only meaningful if we have >=2 snapshots
    _section("Day-over-day stability")
    if len(frames) >= 2:
        prev, curr = frames[-2], frames[-1]
        if "ticker" in prev.columns and "ticker" in curr.columns:
            # De-duplicate by taking the first row per ticker so reindex
            # doesn't choke on snapshots that have multiple asset classes.
            prev1 = prev.drop_duplicates(subset=["ticker"], keep="first")
            curr1 = curr.drop_duplicates(subset=["ticker"], keep="first")
            common = sorted(set(prev1["ticker"]) & set(curr1["ticker"]))
            print(f"  common tickers in last 2 snapshots: {len(common)}")
            if common and "conviction" in curr1.columns and "conviction" in prev1.columns:
                p = prev1.set_index("ticker")["conviction"].astype(float)
                c = curr1.set_index("ticker")["conviction"].astype(float)
                d = (c.reindex(common) - p.reindex(common)).dropna()
                _describe(d, "delta_conviction (curr-prev)")
                if not d.empty:
                    pct_big_swing = float((d.abs() > 30).mean()) * 100
                    print(f"  tickers with |delta_conviction| > 30: {pct_big_swing:.1f}%")
                # Direction flip rate
                if "direction" in prev1.columns and "direction" in curr1.columns:
                    pdir = prev1.set_index("ticker")["direction"]
                    cdir = curr1.set_index("ticker")["direction"]
                    flips = int((pdir.reindex(common) != cdir.reindex(common)).sum())
                    flip_rate = 100 * flips / max(1, len(common))
                    print(f"  direction flip rate: {flip_rate:.1f}% "
                          f"({flips} of {len(common)})")
    else:
        print("  only 1 snapshot — cannot measure stability")

    # Verdict
    _section("Verdict")
    issues: list[str] = []
    if "data_quality_score" in latest.columns:
        med_dq = float(pd.to_numeric(
            latest["data_quality_score"], errors="coerce").dropna().median())
        if med_dq < 60:
            issues.append(f"median data_quality_score is {med_dq:.0f}/100 — "
                          f"upstream feeds are degraded")
        else:
            print(f"  data_quality_score median: {med_dq:.0f}/100 OK")
    if "conviction" in latest.columns:
        c = pd.to_numeric(latest["conviction"], errors="coerce").dropna()
        if not c.empty:
            near_50 = float((c.between(40, 60)).mean()) * 100
            if near_50 > 60:
                issues.append(f"{near_50:.0f}% of convictions cluster around 50 — "
                              f"model is uncertain")
            else:
                print(f"  conviction spread: {near_50:.0f}% in 40-60 band (healthy if <60%)")
    if "proba_up" in latest.columns:
        p = pd.to_numeric(latest["proba_up"], errors="coerce").dropna()
        if not p.empty:
            std = float(p.std())
            if std < 0.05:
                issues.append(f"proba_up std is {std:.3f} — model output collapsed "
                              f"(likely calibrator broken or features degenerate)")
            else:
                print(f"  proba_up dispersion: std={std:.3f} (>0.05 healthy) OK")

    print()
    if issues:
        print("  ISSUES FOUND:")
        for it in issues:
            print(f"    - {it}")
        return 2
    print("  No major data-quality issues detected at the snapshot level.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
