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


# Per-state model weight when blending. Values < 0.5 tilt toward the rubric.
# Used for both hard-state lookup and probability-weighted soft mixing.
_REGIME_MODEL_WEIGHT = {
    "bull":   0.60,
    "chop":   0.60,
    "bear":   0.55,
    "crisis": 0.45,
}


@dataclass
class BlendConfig:
    w_model: float = 0.6
    w_rubric: float = 0.4
    neutral_band: float = 0.02  # +/- around 0.5
    # Phase 28.10 — optional base-rate correction. `blend()` treats 0.5 as the
    # neutral point, which assumes `proba_up` is centered at 0.5. That holds
    # for a model trained on a ~balanced target, but a model trained on an
    # imbalanced label (e.g. triple-barrier with ~33% "up") emits CALIBRATED
    # probabilities centered at its base rate (~0.33), so a fixed 0.5 neutral
    # reads almost everything as "down" and the gate degenerates to ~all-short
    # (see docs/phase_28_9_gated_backtest_notes.md). When set to the model's
    # training positive base rate, proba_up is re-centered to a 0.5 prior via
    # prior-correction BEFORE blending. Default None = no correction = current
    # behaviour preserved byte-for-byte for the deployed ~balanced model.
    model_base_rate: float | None = None

    def __post_init__(self):
        s = self.w_model + self.w_rubric
        if abs(s - 1.0) > 1e-6:
            self.w_model = self.w_model / s
            self.w_rubric = self.w_rubric / s


def _recenter_to_uniform_prior(p: np.ndarray, base_rate: float) -> np.ndarray:
    """Prior-correct probabilities from a `base_rate` prior to a 0.5 prior.

    Standard class-prior-shift adjustment: p_adj = p*(1-b) / (p*(1-b) + (1-p)*b).
    Maps b -> 0.5, is monotone in p, and stays in [0, 1]. NaNs pass through.
    """
    b = float(base_rate)
    num = p * (1.0 - b)
    den = num + (1.0 - p) * b
    with np.errstate(invalid="ignore", divide="ignore"):
        adj = np.where(den > 0, num / den, p)
    return adj


def _soft_model_weight(out: pd.DataFrame, cfg: BlendConfig) -> np.ndarray:
    """Probability-weighted blend weight using HMM soft state distribution.

    Looks for `regime_p_bull / regime_p_chop / regime_p_bear / regime_p_crisis`
    columns. Falls back to hard-state lookup if soft probabilities are not
    present. Smooths transitions at regime boundaries — currently the hard
    blend jumps 0.60 -> 0.45 the day the HMM flips, which is too sharp.
    """
    soft_cols = [f"regime_p_{s}" for s in _REGIME_MODEL_WEIGHT]
    if all(c in out.columns for c in soft_cols):
        # Each row has p_bull + p_chop + p_bear + p_crisis ≈ 1
        probs = out[soft_cols].astype(float).fillna(0.0).to_numpy()
        weights = np.array([_REGIME_MODEL_WEIGHT[s] for s in _REGIME_MODEL_WEIGHT])
        return probs @ weights
    if "regime" in out.columns:
        regime_lc = out["regime"].astype(str).str.lower()
        return np.array([
            _REGIME_MODEL_WEIGHT.get(r, cfg.w_model) for r in regime_lc
        ], dtype=float)
    return np.full(len(out), cfg.w_model, dtype=float)


def blend(df: pd.DataFrame, *, cfg: BlendConfig | None = None) -> pd.DataFrame:
    """`df` must have columns `proba_up` and `rubric_total`.

    Returns df augmented with: blended_p, conviction, direction, plus the
    Phase 24.21 confidence-decomposition columns:
      - blend_w_model        per-row model weight used in the blend
      - blend_w_rubric       per-row rubric weight (= 1 - w_model)
      - blend_model_contrib  w_model * proba_up   (in [0, 1])
      - blend_rubric_contrib w_rubric * (rubric_total / 100)
      - pre_gate_conviction  conviction BEFORE any gate / regime multiplier

    The decomposition columns are pure information — they do not change the
    blended_p / conviction / direction logic. Downstream consumers
    (waterfall chart, explainability panel) read them by name; consumers
    that don't know about them are unaffected.

    Regime-aware re-weighting:
        The default 60/40 (model/rubric) weighting assumes the model
        generalises well. In stress regimes (bear / crisis) historical
        models tend to underperform their CV AUC because the training
        distribution was dominated by bull / chop. We tilt toward the
        rubric in bear (55/45) and more in crisis (45/55).

        When HMM soft probabilities are present (`regime_p_<state>`),
        we blend weights *probabilistically* — eliminates the hard
        step at regime boundaries.
    """
    cfg = cfg or BlendConfig()
    if "proba_up" not in df.columns or "rubric_total" not in df.columns:
        raise KeyError("blend() requires 'proba_up' and 'rubric_total' columns")
    out = df.copy()
    p = out["proba_up"].astype(float)
    # Phase 28.10 — optional base-rate correction (default off; see BlendConfig).
    if cfg.model_base_rate is not None and 0.0 < cfg.model_base_rate < 1.0 \
            and abs(cfg.model_base_rate - 0.5) > 1e-9:
        p = pd.Series(_recenter_to_uniform_prior(p.to_numpy(), cfg.model_base_rate),
                      index=p.index)
    r = (out["rubric_total"].astype(float) / 100.0).clip(0, 1)

    w_model_per_row = _soft_model_weight(out, cfg)
    w_rubric_per_row = 1.0 - w_model_per_row

    p_arr = p.to_numpy()
    r_arr = r.to_numpy()
    model_contrib = w_model_per_row * p_arr
    rubric_contrib = w_rubric_per_row * r_arr
    blended = model_contrib + rubric_contrib

    out["blended_p"] = blended
    out["conviction"] = (pd.Series(blended).sub(0.5).abs() * 200.0).clip(0, 100).values
    out["direction"] = np.where(blended > 0.5 + cfg.neutral_band, 1,
                       np.where(blended < 0.5 - cfg.neutral_band, -1, 0))

    # Phase 24.21 — confidence decomposition. Cheap to compute (no extra
    # passes) and immediately surfaces "what dragged conviction down" via
    # the waterfall chart in the dashboard.
    out["blend_w_model"] = w_model_per_row
    out["blend_w_rubric"] = w_rubric_per_row
    out["blend_model_contrib"] = model_contrib
    out["blend_rubric_contrib"] = rubric_contrib
    out["pre_gate_conviction"] = out["conviction"].values
    return out

