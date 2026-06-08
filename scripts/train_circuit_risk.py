"""Train the lower-circuit risk classifier on the India training set.

Usage:
    python scripts/train_circuit_risk.py
    python scripts/train_circuit_risk.py --threshold-pct 0.095 --horizon 5
    python scripts/train_circuit_risk.py --threshold-pct 0.05 --suffix _5pct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omni.lake import paths as lake_paths  # noqa: E402
from omni.models import circuit_risk as _crisk  # noqa: E402
from omni.utils.logger import log  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train lower-circuit risk classifier (India equities)."
    )
    parser.add_argument(
        "--threshold-pct", type=float, default=0.095,
        help="Drawdown threshold defining a 'circuit hit' (default 0.095 ≈ 10%% circuit)",
    )
    parser.add_argument("--horizon", type=int, default=5,
                          help="Look-forward window in trading days (default 5)")
    parser.add_argument("--n-folds", type=int, default=5,
                          help="Purged K-fold count")
    parser.add_argument(
        "--training-set",
        default=str(lake_paths.lake_root() / "training" / "training_set.parquet"),
        help="Path to the parquet training set (built by build_training_set.py)",
    )
    parser.add_argument("--suffix", default="",
                          help="Suffix appended to the saved-model filename "
                                "(e.g. '_5pct'). Default writes circuit_risk_india.pkl")
    args = parser.parse_args()

    p = Path(args.training_set)
    if not p.exists():
        print(f"ERROR: training set not found: {p}")
        print("Run scripts/build_training_set.py first.")
        return 1

    print(f"[1/3] loading training frame: {p}")
    df = pd.read_parquet(p)
    print(f"      shape={df.shape}")

    if "asset_class" in df.columns:
        n_india = (df["asset_class"].astype(str).str.lower() == "india").sum()
        print(f"      india rows: {n_india:,} ({n_india/len(df):.1%})")

    print(f"[2/3] fitting circuit-risk model "
          f"(threshold={args.threshold_pct}, horizon={args.horizon}d)")
    model = _crisk.fit(
        df,
        threshold_pct=args.threshold_pct,
        horizon=args.horizon,
        n_folds=args.n_folds,
    )

    if model.base is None:
        print("ERROR: training aborted (no positives in training set)")
        return 2

    out_name = f"circuit_risk_india{args.suffix}.pkl"
    out_path = Path("data/models") / out_name
    _crisk.save(model, out_path)

    print("[3/3] done.")
    print(f"      rows trained:   {model.n_train_rows:,}")
    print(f"      base rate:      {model.base_rate:.4f}  (= P(circuit-hit) in data)")
    print(f"      OOF PR-AUC:     {model.pr_auc:.4f}")
    print(f"      lift vs base:   {model.pr_auc / max(model.base_rate, 1e-6):.2f}×")
    print(f"      per-fold AP:    {[f'{x:.4f}' for x in model.folds_pr_auc]}")
    print(f"      n features:     {len(model.feature_cols)}")
    print(f"      saved to:       {out_path}")

    log.info(
        f"circuit_risk train complete: pr_auc={model.pr_auc:.4f} "
        f"base_rate={model.base_rate:.4f} rows={model.n_train_rows}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
