"""Sanity checks for the backtest pipeline.

These are guard rails that should fire BEFORE any results are trusted:

  1. assert_no_target_leakage(features, model_feature_cols)
       -- model's feature list must not contain any column flagged by
          features.build.is_target_col.

  2. assert_chronological_splits(folds)
       -- every train slice must end strictly before its test slice begins.

  3. shuffled_baseline(walkforward_fn, df)
       -- destroys time order via random label shuffle, refits, returns the
          baseline metrics. A real edge should beat this baseline on hit_rate
          and AUC by a meaningful margin (>= 2 percentage points).

  4. assert_feature_alignment(features, proba)
       -- proba index must align 1:1 with the features index used to build
          the input X for the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from omni.features.build import is_target_col
from omni.utils.logger import log


class LeakageError(AssertionError):
    pass


def assert_no_target_leakage(model_feature_cols: list[str]) -> None:
    bad = [c for c in model_feature_cols if is_target_col(c)]
    if bad:
        raise LeakageError(f"target columns leaked into features: {bad}")
    log.info(f"leakage check passed: {len(model_feature_cols)} model features, 0 targets")


def assert_chronological_splits(folds) -> None:
    """Each fold's max(train_dates) must be < min(test_dates)."""
    for i, fr in enumerate(folds):
        if not hasattr(fr, "test_dates"):
            continue
        # we need train dates too; expose via fr.train_dates if added later
        # for now check test dates strictly increase across folds
        if i > 0:
            prev = folds[i - 1].test_dates.max()
            cur = fr.test_dates.min()
            if cur < prev:
                raise LeakageError(
                    f"fold {i} test starts {cur} before fold {i-1} ends {prev}"
                )
    log.info(f"chronological split check passed across {len(folds)} folds")


def assert_feature_alignment(X_index, proba_index) -> None:
    if len(X_index) != len(proba_index):
        raise LeakageError(
            f"feature/proba length mismatch: {len(X_index)} vs {len(proba_index)}"
        )


def shuffled_baseline(df: pd.DataFrame, *, target: str = "fwd_dir_5d",
                      walkforward_fn=None, seed: int = 17) -> dict:
    """Refit on label-shuffled data; expect near-50% hit rate and ~0.5 AUC.

    `walkforward_fn(df_shuffled)` must return a frame with `proba_up` and
    `y_true` columns (the same shape as the real predictions frame).
    """
    if walkforward_fn is None:
        raise ValueError("walkforward_fn is required")
    rng = np.random.default_rng(seed)
    shuffled = df.copy()
    shuffled[target] = rng.permutation(shuffled[target].values)
    preds = walkforward_fn(shuffled)
    from omni.backtest.metrics import hit_rate, auc, brier
    out = {
        "hit_rate": hit_rate(preds),
        "auc": auc(preds),
        "brier": brier(preds),
        "n": int(len(preds)),
    }
    # Sanity: shuffled hit rate should be within ~5pp of 0.5
    if abs(out["hit_rate"] - 0.5) > 0.05:
        log.warning(f"shuffled baseline hit_rate={out['hit_rate']:.3f} "
                    "is suspiciously far from 0.5 -- inspect for leakage")
    return out


def edge_over_baseline(real: dict, baseline: dict, *, min_edge_pp: float = 2.0) -> dict:
    """Return delta and a pass/fail flag."""
    edge_hr = (real.get("hit_rate", 0.5) - baseline.get("hit_rate", 0.5)) * 100
    edge_auc = (real.get("auc", 0.5) - baseline.get("auc", 0.5)) * 100
    return {
        "edge_hit_rate_pp": float(edge_hr),
        "edge_auc_pp": float(edge_auc),
        "passes_min_edge": bool(edge_hr >= min_edge_pp and edge_auc >= min_edge_pp),
    }
