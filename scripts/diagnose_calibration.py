"""A/B compare isotonic vs beta calibration on the universal training set.

For each horizon in {1, 5, 21}, fits an XGB classifier on the standard
training set, then fits both isotonic and beta calibrators on a held-out
inner-validation slice, and reports:
  - Brier score (lower = better)
  - ECE (Expected Calibration Error, 10 equal-width bins)

Usage:
    python scripts/diagnose_calibration.py
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from pathlib import Path

from omni.calibration import BetaCalibrator
from omni.utils.logger import log


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, 10 equal-width bins."""
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=int).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(p)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not mask.any():
            continue
        bin_conf = p[mask].mean()
        bin_acc = y[mask].mean()
        e += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(e)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=int).ravel()
    return float(np.mean((p - y) ** 2))


def _train_set() -> pd.DataFrame:
    p = Path("data/lake/training/training_set.parquet")
    if p.exists():
        return pd.read_parquet(p)
    # fallback to most recent training parquet
    cands = sorted(Path("data/lake/training").glob("*.parquet"))
    if not cands:
        raise RuntimeError("no training parquet found")
    return pd.read_parquet(cands[-1])


def main() -> int:
    from sklearn.isotonic import IsotonicRegression
    from xgboost import XGBClassifier

    df = _train_set()
    print(f"loaded training set: {df.shape}")
    print()

    drop = {"date", "ticker", "asset_class", "display_name"}
    target_prefixes = ("fwd_ret_", "fwd_dir_", "fwd_q_")

    print(f"{'horizon':>8s} | {'brier_iso':>10s} {'brier_beta':>10s} | "
          f"{'ece_iso':>8s} {'ece_beta':>8s} | winner")
    print("-" * 78)

    for h in (1, 5, 21):
        target = f"fwd_dir_{h}d"
        if target not in df.columns:
            print(f"  h={h}: target {target} missing, skip")
            continue
        sub = df.dropna(subset=[target]).reset_index(drop=True)
        # fwd_dir is in {-1, 0, 1}; binarise to up vs not-up so XGBClassifier
        # has a 2-class target.
        sub["_y_bin"] = (sub[target].astype(int) > 0).astype(int)
        feats = [c for c in sub.columns
                  if c not in drop and not c.startswith(target_prefixes)
                  and c != "_y_bin"
                  and pd.api.types.is_numeric_dtype(sub[c])]
        # Time-ordered train/cal/test split: 60/20/20
        n = len(sub)
        i1, i2 = int(n * 0.6), int(n * 0.8)
        tr, ca, te = sub.iloc[:i1], sub.iloc[i1:i2], sub.iloc[i2:]

        def _xy(part, feats=feats):
            X = part[feats].apply(pd.to_numeric, errors="coerce")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e9, 1e9).values
            y = part["_y_bin"].astype(int).values
            return X, y

        Xtr, ytr = _xy(tr)
        Xca, yca = _xy(ca)
        Xte, yte = _xy(te)

        clf = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=-1, verbosity=0,
        )
        clf.fit(Xtr, ytr)
        raw_ca = clf.predict_proba(Xca)[:, 1]
        raw_te = clf.predict_proba(Xte)[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_ca, yca)
        bet = BetaCalibrator().fit(raw_ca, yca)

        p_iso = iso.predict(raw_te)
        p_bet = bet.predict(raw_te)

        b_iso, b_bet = brier(p_iso, yte), brier(p_bet, yte)
        e_iso, e_bet = ece(p_iso, yte), ece(p_bet, yte)
        winner = "beta" if (b_bet < b_iso and e_bet < e_iso) else (
            "iso" if (b_iso <= b_bet and e_iso <= e_bet) else "mixed")
        print(f"  h={h:>5d} | {b_iso:>10.4f} {b_bet:>10.4f} | "
              f"{e_iso:>8.4f} {e_bet:>8.4f} | {winner}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
