"""Train the learned rubric weights on the historical training set.

Walks the training set per ticker through `score_frame` to produce the
9 pillar scores per row, joins to fwd_dir_5d (or tb_label_5d if
present), then fits a per-regime logistic regression.

Output: data/models/learned_rubric_weights.json with shape
{regime -> {pillar_key -> normalised coefficient}}.

Inference is opt-in via env flag `OMNI_LEARNED_RUBRIC=1`.
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

from omni.rubric.learned import save_learned_weights, train_learned_weights
from omni.rubric.scorer import score_frame
from omni.utils.logger import log


def _load_training() -> pd.DataFrame:
    cands = sorted(Path("data/lake/training").glob("*.parquet"))
    if not cands:
        raise RuntimeError("no training parquet at data/lake/training")
    return pd.read_parquet(cands[-1])


def _load_regimes() -> pd.DataFrame:
    p = Path("data/regime/latest.parquet")
    if not p.exists():
        raise RuntimeError(f"regime labels missing at {p}; "
                            f"run scripts/save_regime_labels.py first")
    return pd.read_parquet(p)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--target", default="auto",
                    choices=("auto", "fwd_dir", "tb_label"),
                    help="auto picks tb_label_<H>d if present, else fwd_dir_<H>d")
    p.add_argument("--min-rows-per-regime", type=int, default=500)
    p.add_argument("--out", default=None,
                    help="output JSON path; defaults to "
                         "data/models/learned_rubric_weights.json")
    args = p.parse_args()

    print("[1/4] loading training set + regime labels")
    df = _load_training()
    rg = _load_regimes()
    df["date"] = pd.to_datetime(df["date"])
    rg["date"] = pd.to_datetime(rg["date"])
    df = df.merge(rg, on="date", how="left")
    df = df.dropna(subset=["regime"]).reset_index(drop=True)

    target_col_pref = (
        f"tb_label_{args.horizon}d" if args.target in ("auto", "tb_label")
        else f"fwd_dir_{args.horizon}d"
    )
    if target_col_pref not in df.columns and args.target == "auto":
        log.warning(f"{target_col_pref} missing; falling back to fwd_dir_{args.horizon}d")
        target_col_pref = f"fwd_dir_{args.horizon}d"
    if target_col_pref not in df.columns:
        log.error(f"target column {target_col_pref} not in training set")
        return 1
    print(f"  target column: {target_col_pref}")
    print(f"  rows: {len(df):,}, regime distribution:")
    for lbl, n in df["regime"].value_counts().items():
        print(f"    {lbl:<8s} {n:>8,}")

    print("[2/4] computing pillar scores per row")
    # score_frame requires per-row asset_class; default to whatever's present
    if "asset_class" not in df.columns:
        df["asset_class"] = "us"
    # Build a regime Series indexed by row index for score_frame
    rg_series = pd.Series(df["regime"].astype(str).values, index=df.index)
    scored = score_frame(df, regimes=rg_series)
    # score_frame side-effects 9 *_score columns onto its return value;
    # ensure our regime + target columns survive the call.
    for col in ("regime", target_col_pref):
        if col not in scored.columns and col in df.columns:
            scored[col] = df[col].values

    print("[3/4] fitting per-regime logistic regression")
    weights = train_learned_weights(
        scored,
        target=target_col_pref,
        horizon=args.horizon,
        regime_col="regime",
        min_rows_per_regime=args.min_rows_per_regime,
    )
    if not weights:
        log.error("no per-regime weights produced (all regimes dropped)")
        return 1

    out_path = args.out or "data/models/learned_rubric_weights.json"
    print("[4/4] saving")
    save_learned_weights(weights, out_path)
    print(f"  -> {out_path}")
    print()
    print("Per-regime top-3 pillars by learned weight:")
    for regime, w in weights.items():
        top3 = sorted(w.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {regime:<8s}: {top3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
