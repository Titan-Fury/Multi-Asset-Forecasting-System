"""Train and save the split-conformal interval calibrator.

Fits a `ConformalIntervalCalibrator` on the last ~20% of the training
window and saves it to `data/models/conformal_h5.joblib` so the
prediction pipeline can replace the heuristic ±1σ envelope with a
calibrated 80%-coverage interval.

Usage:
    set PYTHONPATH=C:\\Omni_Stocks
    python scripts/train_conformal.py [--alpha 0.20] [--horizon 5]
                                       [--cal-frac 0.20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from omni.utils.logger import log
from omni.models.conformal import (
    ConformalIntervalCalibrator,
    fit_interval_calibrator,
)
from omni.ondemand.model_registry import load_horizon as _load_universal


def _build_residuals(df: pd.DataFrame, *, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (residuals_pct, sigma_pct) on the supplied frame.

    mu_pred is computed from the universal model's calibrated probability
    via the same edge*sigma rule used at inference (predict.py line 610).
    sigma is realized vol scaled to the horizon, in percent.
    """
    um = _load_universal(horizon=horizon)
    if um is None:
        raise RuntimeError(f"universal model for horizon={horizon} not loaded")

    target = f"fwd_ret_{horizon}d"
    if target not in df.columns:
        raise KeyError(f"missing target column {target}")

    # vol_20 is daily realized vol stored in the training frame
    if "vol_20" not in df.columns:
        raise KeyError("missing vol_20 (daily realized vol) in training frame")

    df = df.dropna(subset=[target, "vol_20"]).copy()

    # Build feature matrix aligned to the universal model's feature_cols
    feat_cols = um.feature_cols
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        log.warning(f"conformal: filling {len(missing)} missing feature cols with 0")
        for c in missing:
            df[c] = 0.0
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    medians = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(medians).fillna(0.0)

    # Calibrated probability -> edge -> mu_pred
    p = um.calibrator.predict_proba(X.to_numpy(dtype=float))[:, 1]
    edge = (p - 0.5) * 2.0  # in [-1, 1]
    sigma_d = pd.to_numeric(df["vol_20"], errors="coerce").fillna(0.02).to_numpy()
    sigma_pct = sigma_d * np.sqrt(horizon) * 100.0

    y_pct = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy() * 100.0
    mu_pred = edge * sigma_pct

    residuals = y_pct - mu_pred
    return residuals, sigma_pct


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit split-conformal interval calibrator.")
    ap.add_argument("--alpha", type=float, default=0.20,
                    help="Miscoverage rate (default 0.20 = 80%% coverage).")
    ap.add_argument("--horizon", type=int, default=5,
                    help="Forecast horizon in days.")
    ap.add_argument("--cal-frac", type=float, default=0.20,
                    help="Fraction of training data to use as calibration set.")
    ap.add_argument("--training-set",
                    default="data/lake/training/training_set.parquet")
    ap.add_argument("--out-path",
                    default="data/models/conformal_h5.joblib")
    ap.add_argument("--asset-class", default=None,
                    help="If set, restrict to one asset class (e.g. india).")
    args = ap.parse_args()

    print(f"[1/4] loading training set: {args.training_set}")
    df = pd.read_parquet(args.training_set)
    if "date" not in df.columns:
        raise KeyError("training set missing date column")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if args.asset_class and "asset_class" in df.columns:
        df = df[df["asset_class"] == args.asset_class].reset_index(drop=True)
        print(f"      filtered to asset_class={args.asset_class}: {len(df)} rows")
    else:
        print(f"      total rows: {len(df)}")

    # Time-based calibration split: last cal_frac dates are calibration
    n = len(df)
    n_cal = max(1000, int(n * args.cal_frac))
    cal_df = df.tail(n_cal).copy()
    print(f"[2/4] calibration set: last {n_cal} rows  "
          f"({cal_df['date'].min().date()} -> {cal_df['date'].max().date()})")

    print(f"[3/4] computing residuals & sigmas (horizon={args.horizon})")
    residuals, sigmas = _build_residuals(cal_df, horizon=args.horizon)

    cal: ConformalIntervalCalibrator = fit_interval_calibrator(
        residuals, sigmas, alpha=args.alpha,
    )

    # Re-evaluate empirical coverage on the calibration set itself (will be
    # ~1-alpha by construction, but verify).
    half = cal.quantile * np.where(sigmas > 0, sigmas, 1e-9)
    inside = (residuals >= -half) & (residuals <= half)
    emp_cov = float(inside.mean())

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, out_path)
    print("[4/4] done.")
    print(f"      alpha:       {cal.alpha}  (target coverage = {1 - cal.alpha:.0%})")
    print(f"      quantile:    {cal.quantile:.3f}")
    print(f"      empirical coverage on cal set: {emp_cov:.3f}")
    print(f"      saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
