"""Daily drift monitor — compute PSI per feature between training-time
baseline and the most recent N days of features, surface alarms.

Usage:
    python scripts/monitor_drift.py
    python scripts/monitor_drift.py --window 30 --top 25 --alarm-threshold 0.25

Reads:
    data/lake/training/*.parquet  (latest = baseline)
    data/lake/predictions/*.parquet  (latest 3 snapshots = current)

Writes:
    data/monitoring/drift_<YYYY-MM-DD>.parquet
    stdout: human-readable top-N table
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

from omni.monitoring.drift import psi_per_feature
from omni.utils.logger import log


def _load_baseline() -> pd.DataFrame:
    cands = sorted(Path("data/lake/training").glob("*.parquet"))
    if not cands:
        raise RuntimeError("no training parquet at data/lake/training")
    return pd.read_parquet(cands[-1])


def _load_current(window: int, baseline_cols: set[str] | None = None
                    ) -> pd.DataFrame:
    """Concatenate the last `window` days of feature snapshots, if they
    share schema with the baseline. Otherwise fall back to the most recent
    `window` business-days slice of the training set itself, which is the
    correct definition of "recent" for offline monitoring anyway.
    """
    pred_dir = Path("data/lake/predictions")
    if pred_dir.exists():
        files = sorted(pred_dir.glob("*.parquet"))
        if files:
            parts = []
            for f in files[-window:]:
                try:
                    parts.append(pd.read_parquet(f))
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"skipping {f}: {exc}")
            if parts:
                cur = pd.concat(parts, ignore_index=True)
                # Use prediction snapshots only when their schema overlaps
                # the baseline meaningfully (>= 50% of numeric features).
                if baseline_cols is not None:
                    overlap = len(set(cur.columns) & baseline_cols)
                    if overlap >= max(20, int(len(baseline_cols) * 0.5)):
                        return cur
                    log.info(f"prediction snapshot schema overlap {overlap} "
                              f"too small; using training-set tail instead")
                else:
                    return cur
    base = _load_baseline()
    if "date" in base.columns:
        cutoff = pd.to_datetime(base["date"]).max() - pd.tseries.offsets.BDay(window)
        sub = base[pd.to_datetime(base["date"]) >= cutoff].reset_index(drop=True)
        if not sub.empty:
            return sub
    return base.tail(min(len(base), 5000)).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=30,
                    help="number of recent days to use as the 'current' sample")
    p.add_argument("--top", type=int, default=25,
                    help="how many top-PSI features to print")
    p.add_argument("--alarm-threshold", type=float, default=0.25,
                    help="PSI value at which a feature is flagged 'alarm'")
    p.add_argument("--save", action="store_true",
                    help="write data/monitoring/drift_<date>.parquet")
    args = p.parse_args()

    print("[1/3] loading baseline + current samples")
    baseline_full = _load_baseline()
    # Compute candidate numeric feature set from the full set first
    drop = {"date", "ticker", "asset_class", "display_name", "regime"}
    target_prefixes = ("fwd_ret_", "fwd_dir_", "fwd_q_", "fwd_rank_",
                       "tb_label_", "tb_first_")
    baseline_numeric_cols = {
        c for c in baseline_full.columns
        if c not in drop
        and not any(c.startswith(p) for p in target_prefixes)
        and pd.api.types.is_numeric_dtype(baseline_full[c])
    }
    current = _load_current(args.window, baseline_cols=baseline_numeric_cols)

    # If current came from the training set (the typical fallback in dev
    # environments where prediction snapshots don't share schema), we
    # split baseline as the OLDER portion of the training set so the two
    # sides are time-disjoint. Otherwise use the full training set as
    # baseline against the live snapshot.
    if id(current) != id(baseline_full) and "date" in baseline_full.columns:
        # Use everything STRICTLY before the current window's first date as baseline
        if "date" in current.columns:
            cur_min = pd.to_datetime(current["date"]).min()
            baseline = baseline_full[pd.to_datetime(baseline_full["date"]) < cur_min]
            if baseline.empty:
                # fall back: no time-prior portion -> use the full training set
                baseline = baseline_full
        else:
            baseline = baseline_full
    else:
        # current was sliced FROM baseline_full; carve baseline into the
        # complement so we're comparing OLDER vs NEWER training data.
        if "date" in baseline_full.columns and "date" in current.columns:
            cur_min = pd.to_datetime(current["date"]).min()
            baseline = baseline_full[pd.to_datetime(baseline_full["date"]) < cur_min]
            if baseline.empty:
                baseline = baseline_full
        else:
            baseline = baseline_full

    print(f"  baseline: {baseline.shape}")
    print(f"  current : {current.shape}")

    # Drop obvious id / target / date columns
    drop = {"date", "ticker", "asset_class", "display_name", "regime"}
    target_prefixes = ("fwd_ret_", "fwd_dir_", "fwd_q_", "fwd_rank_",
                       "tb_label_", "tb_first_")
    feats = [c for c in baseline.columns
              if c not in drop
              and not any(c.startswith(p) for p in target_prefixes)
              and pd.api.types.is_numeric_dtype(baseline[c])
              and c in current.columns]

    print(f"[2/3] computing PSI on {len(feats)} numeric features")
    drift = psi_per_feature(baseline, current, features=feats)

    n_alarm = int((drift["psi"] >= args.alarm_threshold).sum())
    n_watch = int(((drift["psi"] >= 0.10) & (drift["psi"] < args.alarm_threshold)).sum())
    print()
    print(f"  features in alarm  (PSI >= {args.alarm_threshold}): {n_alarm}")
    print(f"  features on watch  (0.10 <= PSI < {args.alarm_threshold}): {n_watch}")
    print()
    head = drift.head(args.top)
    print(f"  Top {len(head)} drifted features:")
    print(f"  {'feature':<30s} {'psi':>8s}  {'severity':<8s}  n_base    n_curr")
    for _, r in head.iterrows():
        print(f"  {r['feature']:<30s} {r['psi']:>8.4f}  {r['severity']:<8s}  "
              f"{r['n_baseline']:>7,d}  {r['n_current']:>7,d}")

    if args.save:
        out_dir = Path("data/monitoring")
        out_dir.mkdir(parents=True, exist_ok=True)
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        path = out_dir / f"drift_{today}.parquet"
        drift.to_parquet(path, index=False)
        print(f"\n[3/3] saved -> {path}")
    else:
        print("\n[3/3] (use --save to persist a drift snapshot)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
