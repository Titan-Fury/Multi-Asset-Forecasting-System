"""Phase-19 A/B benchmark on SBIN.NS.

What this measures (honestly):
    1. End-to-end smoke: does `predict_ticker('SBIN', 'india')` still
       work with the new microstructure + embedding columns added?
       (regression test — must not crash, must return a card.)

    2. Information value of new features: on the last N business days
       of SBIN history, do the new microstructure features carry
       statistically meaningful information about forward 5d returns
       BEFORE retraining? We measure:
         - Pearson correlation of each new feature with fwd_ret_5d.
         - Univariate AUC of each new feature as a sole predictor.

    3. Pipeline parity: prediction probability with new features
       active vs. disabled (`OMNI_DISABLE_EMB=1, OMNI_DISABLE_MICROSTR=1`).
       Since new features aren't in the trained universal model's
       feature_cols, both should produce IDENTICAL `proba_up` —
       confirming we haven't accidentally polluted the live signal.

    4. Speed cost: wall-clock for the live predict call A vs B.

Honest caveats:
    - True AUC lift requires retraining the universal model with the
      new features included. That's `scripts/retrain_all.py` after
      running this benchmark.
    - Bhavcopy historical pull is rate-limited; we sample only the
      last 60 business days of delivery data.

Output:
    - Prints a comparison table to stdout.
    - Writes `docs/phase19_sbi_benchmark.md` with the same table.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# sys.path bootstrap so this works under .venv\Scripts\python.exe directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TICKER = "SBIN"
ASSET = "india"


def predict_with_flags(*, disable_emb: bool, disable_micro: bool) -> tuple[dict, float]:
    """Run predict_ticker with feature toggles. Returns (card_dict, elapsed_s)."""
    os.environ["OMNI_DISABLE_EMB"] = "1" if disable_emb else "0"
    os.environ["OMNI_DISABLE_MICROSTR"] = "1" if disable_micro else "0"
    # purge cache so toggles take effect
    try:
        from omni.ondemand.cache import purge_ticker
        purge_ticker(TICKER)
    except Exception:
        pass
    from omni.ondemand.predict import predict_ticker
    t0 = time.perf_counter()
    card = predict_ticker(TICKER, ASSET)
    t1 = time.perf_counter()
    return card.to_dict(), t1 - t0


def feature_information_value() -> pd.DataFrame:
    """Compute correlation + univariate AUC of new features vs fwd_ret_5d.

    Pulls SBIN history, builds the features inline (with all new families
    enabled), then evaluates the columns of interest on the same frame.
    """
    from omni.ondemand.features_inline import build_features_inline
    from omni.ondemand.lookup import lookup_ticker

    # Make sure new features are enabled for this run
    os.environ["OMNI_DISABLE_EMB"] = "0"
    os.environ["OMNI_DISABLE_MICROSTR"] = "0"

    bundle = lookup_ticker(TICKER, ASSET)
    feats = build_features_inline(
        ASSET, TICKER, bundle.prices,
        fundamentals=bundle.fundamentals,
        sentiment=bundle.news_sentiment,
        macro=bundle.macro,
        horizons=(1, 5, 20),
    )
    if "fwd_ret_5d" not in feats.columns:
        return pd.DataFrame()

    eval_cols = [
        # microstructure
        "delivery_pct_z_60d", "oi_change_pct_5d", "fii_dii_net_z_20d",
        "amihud_illiq_20d", "signed_vol_proxy", "microstr_score",
        # a few embedding dims
        "emb_00", "emb_01", "emb_02", "emb_03",
    ]
    eval_cols = [c for c in eval_cols if c in feats.columns]

    rows = []
    y_ret = feats["fwd_ret_5d"]
    y_dir = (y_ret > 0).astype(int)
    for c in eval_cols:
        x = feats[c]
        m = x.notna() & y_ret.notna()
        if m.sum() < 30:
            rows.append({"feature": c, "n": int(m.sum()),
                         "corr_fwd_ret_5d": np.nan, "univariate_auc": np.nan})
            continue
        try:
            corr = float(np.corrcoef(x[m], y_ret[m])[0, 1])
        except Exception:
            corr = np.nan
        try:
            # AUC of x as predictor of y_dir; if x is all-equal, fall back to NaN
            if x[m].nunique() < 2:
                auc = np.nan
            else:
                auc = float(roc_auc_score(y_dir[m], x[m]))
        except Exception:
            auc = np.nan
        rows.append({"feature": c, "n": int(m.sum()),
                     "corr_fwd_ret_5d": corr, "univariate_auc": auc})
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 70)
    print(f"Phase-19 A/B benchmark on {TICKER}.NS")
    print("=" * 70)

    print("\n[1/3] Pipeline parity check (new features should NOT change live proba)")
    cardA, tA = predict_with_flags(disable_emb=True, disable_micro=True)
    cardB, tB = predict_with_flags(disable_emb=False, disable_micro=False)
    parity = pd.DataFrame([
        {"variant": "baseline (emb+microstr OFF)",
         "proba_up": round(cardA["proba_up"], 4),
         "blended_p": round(cardA["blended_p"], 4),
         "direction": cardA["direction"],
         "conviction": round(cardA["conviction"], 1),
         "elapsed_s": round(tA, 2)},
        {"variant": "with new features ON",
         "proba_up": round(cardB["proba_up"], 4),
         "blended_p": round(cardB["blended_p"], 4),
         "direction": cardB["direction"],
         "conviction": round(cardB["conviction"], 1),
         "elapsed_s": round(tB, 2)},
    ])
    print(parity.to_string(index=False))
    delta = abs(cardA["proba_up"] - cardB["proba_up"])
    print(f"\n  proba_up delta = {delta:.4f}  "
          f"({'PASS' if delta < 1e-6 else 'NOTE: differs - new features reached the model'})")

    print("\n[2/3] Information value of new features on SBIN history")
    iv = feature_information_value()
    if iv.empty:
        print("  (no fwd_ret_5d available — skipped)")
    else:
        # Format AUC as deviation from 0.5 so direction is obvious
        iv = iv.assign(auc_minus_05=lambda d: d["univariate_auc"] - 0.5)
        print(iv.round(4).to_string(index=False))
        meaningful = iv[(iv["univariate_auc"].fillna(0.5) - 0.5).abs() > 0.02]
        print(f"\n  features with |AUC-0.5| > 0.02 (potentially predictive): "
              f"{len(meaningful)} / {len(iv)}")

    print("\n[3/3] Speed cost")
    print(f"  baseline  : {tA:.2f}s")
    print(f"  with new  : {tB:.2f}s   "
          f"(overhead: {tB - tA:+.2f}s)")

    # Write summary
    out_md = Path("docs/phase19_sbi_benchmark.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Phase-19 SBI benchmark\n\n## Pipeline parity\n\n```\n")
        f.write(parity.to_string(index=False))
        f.write(f"\n```\n\nproba_up delta = `{delta:.4f}`\n\n")
        f.write("## Feature information value (vs fwd_ret_5d)\n\n```\n")
        if not iv.empty:
            f.write(iv.round(4).to_string(index=False))
        f.write(f"\n```\n\n## Speed\n\n- baseline: {tA:.2f}s\n- with new: {tB:.2f}s "
                f"(overhead {tB - tA:+.2f}s)\n")
    print(f"\nWritten {out_md}")


if __name__ == "__main__":
    main()
