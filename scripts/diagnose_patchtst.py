"""Phase 25.1 — diagnose PatchTST as a 4th stack base learner.

Trains the stack twice on the TB-augmented training set (h=5):
  1. Without PatchTST (baseline = current production)
  2. With PatchTST  (OMNI_ENABLE_PATCHTST=1)

Reports per-base OOF AUC and meta AUC for each. Decision rule:
  - Ship PatchTST if its base OOF AUC >= 0.51 AND it lifts meta AUC by
    at least +0.005 (the threshold previously used for diff features).
  - Otherwise leave it gated off (mirror NHITS pattern).

Output:
  data/models/patchtst_diagnostic.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from omni.utils.logger import log
from omni.models.stack import fit_stack

TRAIN = Path("data/lake/training/training_set.parquet")
HORIZON = 5
TB_COL = f"tb_label_{HORIZON}d"


def _prep():
    df = pd.read_parquet(TRAIN)
    df = df.drop_duplicates(["ticker", "date"], keep="last")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if TB_COL not in df.columns:
        raise RuntimeError(
            f"{TB_COL} missing — run scripts/augment_triple_barrier.py first"
        )
    df = df.dropna(subset=[TB_COL])
    log.info(f"prep: {len(df):,} rows, {df['ticker'].nunique()} tickers")

    drop_cols = {"date", "ticker", "asset_class", "run_date", "asof_date"}
    drop_cols |= {c for c in df.columns
                  if c.startswith(("fwd_", "tb_label_", "tb_first_", "fwd_rank_"))}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    # Sanitise: replace +/-inf with NaN so XGBoost / LightGBM don't choke.
    n_inf = int(np.isinf(X.to_numpy(dtype=float, na_value=np.nan)).sum())
    if n_inf:
        log.info(f"prep: clipping {n_inf:,} inf values to NaN")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = (df[TB_COL].astype(int) == 1).astype(int).to_numpy()
    dates = pd.to_datetime(df["date"])
    tickers = df["ticker"].astype(str)

    # PatchTST needs closes; reconstruct from log_ret_1d
    if "log_ret_1d" in df.columns:
        closes = np.exp(df.groupby("ticker")["log_ret_1d"]
                          .cumsum().fillna(0)).to_numpy()
    else:
        closes = np.ones(len(df))
    return X, y, dates, tickers, pd.Series(closes)


def _run_stack(X, y, dates, tickers, closes, *, enable_patchtst: bool):
    os.environ["OMNI_ENABLE_PATCHTST"] = "1" if enable_patchtst else "0"
    log.info(f"== fit_stack(patchtst={enable_patchtst}) ==")
    t0 = time.time()
    rec = fit_stack(
        X, y, dates,
        horizon_days=HORIZON, n_folds=5, embargo_days=HORIZON,
        tickers=tickers, closes=closes,
    )
    return {
        "patchtst": enable_patchtst,
        "elapsed_s": round(time.time() - t0, 1),
        "oof_aucs": rec.oof_aucs,
        "meta_auc": rec.oof_meta_auc,
        "base_models": list(rec.base_models.keys()),
    }


def main() -> int:
    X, y, dates, tickers, closes = _prep()

    baseline = _run_stack(X, y, dates, tickers, closes, enable_patchtst=False)
    log.info(f"baseline: {baseline}")

    with_pt = _run_stack(X, y, dates, tickers, closes, enable_patchtst=True)
    log.info(f"with patchtst: {with_pt}")

    pt_base_auc = with_pt["oof_aucs"].get("patchtst", 0.0)
    meta_lift = with_pt["meta_auc"] - baseline["meta_auc"]
    decision = (pt_base_auc >= 0.51 and meta_lift >= 0.005)

    out = {
        "horizon": HORIZON,
        "baseline": baseline,
        "with_patchtst": with_pt,
        "patchtst_base_auc": pt_base_auc,
        "meta_auc_lift": meta_lift,
        "decision": "SHIP" if decision else "GATE_OFF",
    }
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/models/patchtst_diagnostic.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print("  PatchTST diagnostic — h=5")
    print("=" * 78)
    print(f"  baseline base AUCs:     {baseline['oof_aucs']}")
    print(f"  baseline meta AUC:      {baseline['meta_auc']:.4f}")
    print(f"  with-patchtst AUCs:     {with_pt['oof_aucs']}")
    print(f"  with-patchtst meta:     {with_pt['meta_auc']:.4f}")
    print(f"  patchtst base AUC:      {pt_base_auc:.4f}  "
          f"(threshold 0.51)")
    print(f"  meta AUC lift:          {meta_lift:+.4f}  "
          f"(threshold +0.005)")
    print(f"  DECISION:               {out['decision']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
