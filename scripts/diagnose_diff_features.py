"""Diagnostic A/B: OOF AUC with vs without diff_lagged features.

Loads the latest training set, computes diff/lag columns on top per
ticker, then runs a 60/20/20 time-ordered XGB fit and reports OOF AUC
with and without the new columns for h ∈ {1, 5, 21}.

Decision rule: commit only if the new features lift AUC by >= 0.005
on at least one horizon and don't regress others.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from omni.features.diff_lagged import add_diff_and_lagged


def _train_set() -> pd.DataFrame:
    cands = sorted(Path("data/lake/training").glob("*.parquet"))
    if not cands:
        raise RuntimeError("no training parquet found")
    return pd.read_parquet(cands[-1])


def _augment(df: pd.DataFrame, horizons=(1, 5, 21)) -> pd.DataFrame:
    """Apply diff_lagged per-ticker so cross-ticker rows don't pollute shifts."""
    parts = []
    if "ticker" not in df.columns:
        return add_diff_and_lagged(df.copy(), horizons=horizons)
    for tk, sub in df.groupby("ticker", sort=False):
        sub2 = add_diff_and_lagged(sub.copy(), horizons=horizons)
        parts.append(sub2)
    return pd.concat(parts, ignore_index=True)


def _xy(df, feats, target):
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e9, 1e9).values
    y = (df[target].astype(int) > 0).astype(int).values
    return X, y


def main() -> int:
    df = _train_set()
    print(f"loaded training set: {df.shape}")
    df_plus = _augment(df)

    new_cols = [c for c in df_plus.columns if c not in df.columns]
    print(f"new columns added: {new_cols}")
    print()

    drop = {"date", "ticker", "asset_class", "display_name"}
    target_prefixes = ("fwd_ret_", "fwd_dir_", "fwd_q_", "fwd_rank_",
                       "tb_label_", "tb_first_")

    print(f"{'h':>3s} | {'auc_base':>9s} {'auc_plus':>9s} {'delta':>7s}")
    print("-" * 40)

    for h in (1, 5, 21):
        target = f"fwd_dir_{h}d"
        if target not in df_plus.columns:
            print(f"  h={h}: target missing; skip")
            continue

        sub_p = df_plus.dropna(subset=[target]).reset_index(drop=True)
        sub_b = df.dropna(subset=[target]).reset_index(drop=True)

        feats_b = [c for c in sub_b.columns
                   if c not in drop and not c.startswith(target_prefixes)
                   and pd.api.types.is_numeric_dtype(sub_b[c])]
        feats_p = [c for c in sub_p.columns
                   if c not in drop and not c.startswith(target_prefixes)
                   and pd.api.types.is_numeric_dtype(sub_p[c])]

        # Time-ordered 80/20 split
        for sub, feats, name in ((sub_b, feats_b, "base"), (sub_p, feats_p, "plus")):
            n = len(sub)
            i1 = int(n * 0.8)
            tr, te = sub.iloc[:i1], sub.iloc[i1:]
            Xtr, ytr = _xy(tr, feats, target)
            Xte, yte = _xy(te, feats, target)
            clf = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                n_jobs=-1, verbosity=0,
            )
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            try:
                auc = float(roc_auc_score(yte, p))
            except Exception:
                auc = float("nan")
            if name == "base":
                auc_b = auc
            else:
                auc_p = auc
        delta = auc_p - auc_b
        print(f"  h={h:>1d} | {auc_b:>9.4f} {auc_p:>9.4f} {delta:>+7.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
