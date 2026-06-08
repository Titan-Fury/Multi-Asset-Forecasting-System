"""End-to-end signal pipeline.

Glue layer that joins the parts written in phases 4 and 5 into a single
callable. Used by the daily orchestrator (phase 9) and the backtester
(phase 7).

For one (ticker, asof) the pipeline is:
   features -> regime label -> model proba -> rubric score -> blend -> gate

This module does NOT train models or fit HMMs -- it consumes already-fitted
artifacts and produces the final signal frame. Training happens in the
walk-forward fit step.
"""
from __future__ import annotations

import pandas as pd

from omni.ensemble.combine import BlendConfig, blend
from omni.ensemble.gate import GateConfig
from omni.ensemble.gate import apply as gate_apply
from omni.regime.hmm import RegimeModel
from omni.rubric.scorer import score_frame as rubric_score_frame
from omni.utils.logger import log


def assemble_signals(features: pd.DataFrame,
                     *,
                     proba_up: pd.Series,
                     regime_model: RegimeModel | None = None,
                     regime_inputs: pd.DataFrame | None = None,
                     regime_override: pd.Series | None = None,
                     blend_cfg: BlendConfig | None = None,
                     gate_cfg: GateConfig | None = None) -> pd.DataFrame:
    """Combine features + model proba into final signal frame.

    Args:
      features: feature frame (must have ticker, date, asset_class).
      proba_up: aligned with features.index, calibrated p(fwd_dir_5d > 0).
      regime_model: fitted RegimeModel; predicts label per row in `regime_inputs`.
      regime_inputs: regime feature frame indexed by date; must cover all dates
        in `features`.
      regime_override: optional Series (date -> label) to bypass HMM (e.g. backtest).
      blend_cfg / gate_cfg: optional overrides.

    Output columns:
      ticker, date, asset_class, proba_up, rubric_total, regime,
      blended_p, conviction, direction, passes_gate, reasons
    """
    df = features.copy()
    df["proba_up"] = proba_up.values if hasattr(proba_up, "values") else proba_up

    # Attach regime
    if regime_override is not None:
        rmap = regime_override.copy()
        rmap.index = pd.to_datetime(rmap.index)
        df["regime"] = pd.to_datetime(df["date"]).map(rmap).fillna("chop")
    elif regime_model is not None and regime_inputs is not None:
        ri = regime_inputs.set_index("date")
        labels = regime_model.predict(ri)
        rmap = pd.Series(labels, index=ri.index)
        df["regime"] = pd.to_datetime(df["date"]).map(rmap).fillna("chop")
    else:
        df["regime"] = "chop"

    # Rubric
    rub = rubric_score_frame(df, regimes=df["regime"])
    # Blend
    blended = blend(rub, cfg=blend_cfg)
    # Gate (keep all rows; flag pass/fail with reasons)
    gated = gate_apply(blended, cfg=gate_cfg)

    keep = ["ticker", "date", "asset_class", "regime",
            "proba_up", "rubric_total", "blended_p", "conviction",
            "direction", "passes_gate", "reasons"]
    out = gated[[c for c in keep if c in gated.columns]].copy()
    log.info(f"pipeline: rows={len(out)} pass={int(out['passes_gate'].sum())}")
    return out
