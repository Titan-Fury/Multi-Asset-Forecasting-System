"""Difference encoding + lagged-label features (Phase 25 / Commit D).

Two additions to the per-ticker feature frame:

1. Difference encoding: tree-based learners pick up on momentum more
   reliably when given *raw price changes* alongside the existing
   percentage-return features. We add:
     - delta_close_5d   = adj_close - adj_close.shift(5)
     - delta_close_20d  = adj_close - adj_close.shift(20)
     - delta_atr_5d     = atr_14 - atr_14.shift(5)   (vol regime change)
     - delta_rsi_5d     = rsi_14 - rsi_14.shift(5)
   These are not redundant with `ret_5d`/`ret_20d` because the magnitude
   of the underlying series matters at the cross-sectional level (a
   5-rupee move in a Rs.50 stock is different from a Rs.5000 stock even
   if the percentage return is identical).

2. Lagged realised labels: a *fully-realised* label from the past is a
   leakage-free feature at scoring time. For horizon h, we use the label
   computed h+1 trading days ago (so its realisation window ends BEFORE
   today's row date). For h=5 that's `fwd_dir_5d.shift(6)`.

   This gives the model access to its own recent prediction performance
   without cheating: at scoring time t, we know whether the label that
   was generated at t-6 (and realised at t-1) came in as +1 or -1. This
   is a strong, simple signal for momentum persistence.

Empirical lift on the training set (scripts/diagnose_diff_features.py,
lake snapshot 2026-05-14, time-ordered 80/20 split):

  | horizon | auc_base | auc_plus | delta   |
  | ------- | -------- | -------- | ------- |
  | h=1     | 0.5319   | 0.5346   | +0.0027 |
  | h=5     | 0.5682   | 0.5687   | +0.0005 |
  | h=21    | 0.6277   | 0.6287   | +0.0010 |

Lift is consistently positive but modest (below the +0.005 threshold).
We keep the wiring because (a) no horizon regresses, (b) the runtime
cost is essentially zero (7 extra columns; computed once per row), and
(c) the lift is largest on h=1 — the horizon closest to coin-flip,
where any structural signal counts.

API:
    out = add_diff_and_lagged(out, horizons=(1, 5, 21))

Idempotent — adds columns; doesn't overwrite an existing populated value.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

_DIFF_COLS = ("delta_close_5d", "delta_close_20d",
              "delta_atr_5d", "delta_rsi_5d")


def _ensure_cols(out: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan


def add_diff_and_lagged(out: pd.DataFrame, *,
                          horizons: Iterable[int] = (1, 5, 21)) -> pd.DataFrame:
    """Append diff-encoded technical features and lagged realised labels.

    `out` must already contain `adj_close` (and ideally `atr_14`,
    `rsi_14`). For each horizon `h`, if `fwd_dir_{h}d` exists in `out`
    we add `lag_real_fwd_dir_{h}d = fwd_dir_{h}d.shift(h + 1)`.
    """
    if out is None or out.empty:
        return out
    out = out.sort_values("date").reset_index(drop=True)

    _ensure_cols(out, _DIFF_COLS)

    if "adj_close" in out.columns:
        ac = out["adj_close"].astype(float)
        # Phase 24.22 — express as PERCENTAGE delta, not raw rupee delta.
        # The previous version emitted absolute price differences which are
        # not ticker-invariant: a Rs 5 move is huge for a Rs 50 stock and
        # nothing for a Rs 5000 stock, but the model receives the same
        # value. With a ticker-invariant denominator the feature works
        # cross-sectionally and trees can split meaningfully.
        prev5 = ac.shift(5)
        prev20 = ac.shift(20)
        out["delta_close_5d"] = (ac - prev5) / prev5.replace(0, np.nan)
        out["delta_close_20d"] = (ac - prev20) / prev20.replace(0, np.nan)
    # Phase 24.22 — was looking for `atr_14` but technical.py emits
    # `atr_pct_14` (the normalised version) so the branch was dead and
    # `delta_atr_5d` was always NaN. Use the normalised column.
    atr_col = "atr_pct_14" if "atr_pct_14" in out.columns else (
        "atr_14" if "atr_14" in out.columns else None)
    if atr_col is not None:
        a = out[atr_col].astype(float)
        out["delta_atr_5d"] = a - a.shift(5)
    if "rsi_14" in out.columns:
        r = out["rsi_14"].astype(float)
        out["delta_rsi_5d"] = r - r.shift(5)

    # Lagged REALISED labels — strict no-leakage shift of (h + 1).
    # The +1 is the embargo so the realisation window of the label has
    # closed before today's row date.
    for h in horizons:
        col = f"fwd_dir_{h}d"
        if col in out.columns:
            new = f"lag_real_fwd_dir_{h}d"
            out[new] = out[col].shift(int(h) + 1)
    return out
