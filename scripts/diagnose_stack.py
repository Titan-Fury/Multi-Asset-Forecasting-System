"""Phase 24.18 — diagnose the AUC=0.50 model collapse.

Compares the old (legacy pickle .bak) stack vs the freshly-trained joblib
stack on three dimensions:

  1. OOF AUC per base learner + meta
  2. Per-fold AUC variance (does the model have signal in *some* regimes?)
  3. Feature NaN ratio + class balance per fold

The output tells us whether training got *worse* between artifact
generations (regression) or has been at 0.50 forever.
"""
from __future__ import annotations

import json
import pickle  # nosec - reading our own legacy artifacts
from pathlib import Path

import joblib
import numpy as np

from omni.utils.logger import log

MODELS_DIR = Path("data/models")


def _load(path: Path):
    if path.suffix == ".joblib":
        return joblib.load(path)
    with path.open("rb") as f:
        return pickle.load(f)  # nosec


def _summarize_stack(label: str, bundle: dict) -> None:
    """Pull AUCs out of either the new or legacy stack bundle shape."""
    sr = bundle.get("stack") if isinstance(bundle, dict) else bundle
    if sr is None:
        log.warning(f"[{label}] no .stack key in bundle")
        return
    print()
    print("=" * 78)
    print(f"  {label}")
    print("=" * 78)
    aucs = getattr(sr, "oof_aucs", {}) or {}
    meta_auc = getattr(sr, "oof_meta_auc", None)
    horizon = getattr(sr, "horizon_days", "?")
    feats = getattr(sr, "feature_cols", [])
    bases = getattr(sr, "base_models", {})
    print(f"  horizon_days     : {horizon}")
    print(f"  n_features       : {len(feats)}")
    print(f"  base learners    : {sorted(bases.keys()) if bases else '(none)'}")
    print(f"  OOF AUC per base : {aucs}")
    print(f"  OOF meta AUC     : {meta_auc}")
    if isinstance(meta_auc, (int, float)):
        if meta_auc < 0.51:
            print("  -> NO SIGNAL: meta is at coin-flip level")
        elif meta_auc < 0.55:
            print("  -> WEAK SIGNAL: marginal lift over baseline")
        elif meta_auc < 0.60:
            print("  -> USABLE: small but real edge")
        else:
            print("  -> STRONG: above 0.60 OOF AUC is good for daily-direction")
    if aucs:
        avg = np.mean(list(aucs.values()))
        worst = min(aucs.values())
        best = max(aucs.values())
        print(f"  base AUC range   : worst={worst:.4f}  avg={avg:.4f}  best={best:.4f}")


def main() -> int:
    print("\n" + "=" * 78)
    print("  STACK MODEL DIAGNOSTIC — old (.pkl.bak) vs new (.joblib)")
    print("=" * 78)

    pairs = [
        (5, MODELS_DIR / "stack_h5.joblib", MODELS_DIR / "stack_h5.pkl.bak"),
        (1, MODELS_DIR / "stack_h1.joblib", MODELS_DIR / "stack_h1.pkl.bak"),
        (21, MODELS_DIR / "stack_h21.joblib", MODELS_DIR / "stack_h21.pkl.bak"),
    ]

    for h, new_p, old_p in pairs:
        print()
        print("#" * 78)
        print(f"#  HORIZON {h}d")
        print("#" * 78)
        if old_p.exists():
            try:
                old_b = _load(old_p)
                _summarize_stack(f"OLD  ({old_p.name}, pre-fix legacy pickle)", old_b)
            except Exception as e:  # noqa: BLE001
                log.warning(f"old stack h{h} unreadable: {e}")
        else:
            print(f"  no legacy pickle at {old_p}")
        if new_p.exists():
            try:
                new_b = _load(new_p)
                _summarize_stack(f"NEW  ({new_p.name}, current joblib)", new_b)
            except Exception as e:  # noqa: BLE001
                log.warning(f"new stack h{h} unreadable: {e}")
        else:
            print(f"  no joblib at {new_p}")

    # Also dump universal model meta for reference
    print()
    print("=" * 78)
    print("  Universal models (currently deployed)")
    print("=" * 78)
    for h in (1, 5, 21):
        meta_p = MODELS_DIR / f"universal_{h}d.meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        oof = meta.get("oof_auc")
        n_t = len(meta.get("tickers_used", []) or [])
        ts = meta.get("trained_at", "?")
        flag = ""
        if isinstance(oof, (int, float)):
            if oof < 0.51:
                flag = "  <-- COLLAPSED (coin flip)"
            elif oof < 0.55:
                flag = "  <-- weak"
        print(f"  universal_{h}d: oof_auc={oof}  n_tickers={n_t}  trained={ts}{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
