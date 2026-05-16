"""Train per-regime stacks (Phase 25 / Commit E).

For each regime label in {bull, chop, bear, crisis}, slice the training
set to rows whose date matches that regime, then fit a stack via
`omni.models.stack.fit_stack`. Saves a single bundle to
`data/models/regime_stacks_h{horizon}.joblib`.

Usage:
    python scripts/train_regime_stacks.py --horizon 5 --n-folds 5

Inference is gated by `OMNI_REGIME_STACKS=1` so the existing single-stack
production path remains the default until validated.
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

import numpy as np
import pandas as pd

from omni.features.build import is_target_col
from omni.models import regime_stack as _rs
from omni.models.stack import fit_stack
from omni.models.xgboost_5d import _DROP_ALWAYS
from omni.utils.logger import log


REGIME_PARQUET = Path("data/regime/latest.parquet")


def _load_training() -> pd.DataFrame:
    cands = sorted(Path("data/lake/training").glob("*.parquet"))
    if not cands:
        raise RuntimeError("no training parquet found at data/lake/training")
    df = pd.read_parquet(cands[-1])
    if "date" not in df.columns:
        raise RuntimeError("training parquet missing 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_regimes() -> pd.DataFrame:
    if not REGIME_PARQUET.exists():
        raise RuntimeError(f"regime labels missing at {REGIME_PARQUET}; "
                            f"run scripts/save_regime_labels.py first")
    rg = pd.read_parquet(REGIME_PARQUET)
    rg["date"] = pd.to_datetime(rg["date"])
    return rg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--target", default="dir", choices=("dir",),
                    help="binary fwd_dir target. Other targets not supported here.")
    p.add_argument("--min-rows", type=int, default=2000,
                    help="skip regimes with fewer than this many training rows")
    args = p.parse_args()

    print(f"[1/4] loading training set + regime labels")
    df = _load_training()
    rg = _load_regimes()
    df = df.merge(rg, on="date", how="left")
    n_unlabelled = int(df["regime"].isna().sum())
    if n_unlabelled:
        log.warning(f"{n_unlabelled} training rows have no regime label "
                    f"(date outside [{rg['date'].min()}, {rg['date'].max()}]); "
                    f"they will be dropped.")
    df = df.dropna(subset=["regime"]).reset_index(drop=True)

    target = f"fwd_dir_{args.horizon}d"
    if target not in df.columns:
        print(f"ERROR: target {target} not in training set")
        return 1
    df = df.dropna(subset=[target]).reset_index(drop=True)
    df["_y"] = (df[target].astype(int) > 0).astype(int)

    print(f"  rows after merge/dropna: {len(df):,}")
    print(f"  regime distribution:")
    for lbl, n in df["regime"].value_counts().items():
        print(f"    {lbl:<8s} {n:>8,}")

    feat_cols = [c for c in df.columns
                  if c not in _DROP_ALWAYS
                  and not is_target_col(c)
                  and c not in ("regime", "_y")
                  and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  feature cols: {len(feat_cols)}")

    print(f"[2/4] training one stack per regime (horizon={args.horizon})")
    stacks: dict = {}
    aucs: dict[str, float] = {}
    n_per: dict[str, int] = {}
    for label in df["regime"].unique():
        sub = df[df["regime"] == label].sort_values("date").reset_index(drop=True)
        if len(sub) < args.min_rows:
            print(f"  {label}: only {len(sub)} rows; skip (min={args.min_rows})")
            continue
        print(f"  {label}: rows={len(sub):,} ...")
        X = sub[feat_cols].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        y = sub["_y"].astype(int).to_numpy()
        try:
            sr = fit_stack(
                X=X, y=y, dates=sub["date"],
                horizon_days=args.horizon, n_folds=args.n_folds,
                tickers=sub["ticker"] if "ticker" in sub.columns else None,
            )
            stacks[label] = sr
            aucs[label] = float(sr.oof_meta_auc)
            n_per[label] = int(len(sub))
            print(f"    -> oof_meta_auc={sr.oof_meta_auc:.4f}")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"regime={label} stack fit failed: {exc}")

    if not stacks:
        print("ERROR: no per-regime stacks trained")
        return 1

    print(f"[3/4] assembling bundle")
    bundle = _rs.RegimeStackBundle(
        stacks=stacks,
        feature_cols=feat_cols,
        horizon_days=args.horizon,
        oof_aucs=aucs,
        n_rows_per_regime=n_per,
        trained_at=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    out_path = _rs.default_path(args.horizon)
    _rs.RegimeStackBundle.save(bundle, out_path)

    print(f"[4/4] done.")
    print(f"  saved -> {out_path}")
    print(f"  oof_aucs: {aucs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
