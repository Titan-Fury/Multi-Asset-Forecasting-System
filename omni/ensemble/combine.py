"""Probability-rubric blending.

Inputs per (ticker, date):
  proba_up   -- calibrated probability of fwd 5d return > 0  (XGBoost output)
  rubric_total  -- 0..100 rubric score from omni.rubric.scorer

We blend these into a single conviction in [0, 100]:

  rubric_p   = rubric_total / 100                          # in [0, 1]
  blended    = w_model * proba_up + w_rubric * rubric_p     # in [0, 1]
  conviction = blended * 100                                # in [0, 100]

Default weights: model 0.6, rubric 0.4 — favour the calibrated model but keep
the rubric as a sanity check. The two should disagree often enough that the
gate (next step) is meaningful; that's the whole point of a hybrid.

Direction is implied by `blended`:
  +1 if blended > 0.5 + neutral_band
  -1 if blended < 0.5 - neutral_band
   0 otherwise

The conviction strength is `abs(blended - 0.5) * 200` (0 = no conviction,
100 = certain). This is what the gate compares to `min_conviction`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BlendConfig:
    w_model: float = 0.6
    w_rubric: float = 0.4
    neutral_band: float = 0.02  # +/- around 0.5

    def __post_init__(self):
        s = self.w_model + self.w_rubric
        if abs(s - 1.0) > 1e-6:
            self.w_model = self.w_model / s
            self.w_rubric = self.w_rubric / s


def blend(df: pd.DataFrame, *, cfg: BlendConfig | None = None) -> pd.DataFrame:
    """`df` must have columns `proba_up` and `rubric_total`.

    Returns df augmented with: blended_p, conviction, direction.
    """
    cfg = cfg or BlendConfig()
    if "proba_up" not in df.columns or "rubric_total" not in df.columns:
        raise KeyError("blend() requires 'proba_up' and 'rubric_total' columns")
    out = df.copy()
    p = out["proba_up"].astype(float)
    r = (out["rubric_total"].astype(float) / 100.0).clip(0, 1)
    blended = cfg.w_model * p + cfg.w_rubric * r
    out["blended_p"] = blended
    out["conviction"] = (blended.sub(0.5).abs() * 200.0).clip(0, 100)
    out["direction"] = np.where(blended > 0.5 + cfg.neutral_band, 1,
                       np.where(blended < 0.5 - cfg.neutral_band, -1, 0))
    return out
