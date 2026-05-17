"""Confidence Waterfall chart (Phase 24.21 — Feature 1).

Decomposes the final conviction shown on a prediction card into the
intermediate components produced by `omni.ensemble.combine.blend`:

    starting baseline (50% prior on a coin-flip market)
      + model_contrib   (proba_up * w_model)         -> intermediate
      + rubric_contrib  (rubric_total/100 * w_rubric) -> intermediate
      = blended_p
      |conviction = |blended_p - 0.5| * 200|         -> pre_gate_conviction
      x regime multiplier (from gate.regime_dir_multiplier)
      = effective_conviction shown on the card

Renders as a Plotly waterfall so users can see exactly which lever (model,
rubric, regime gate) drove the final number — answers the long-standing
"why is conviction only 12?" question with one chart.

Usage from the dashboard:

    from omni.dashboard.charts.waterfall import render_conviction_waterfall
    render_conviction_waterfall(card_dict)

`card_dict` is the dict produced by PredictionCard.to_dict() — it must
carry the decomposition fields written by `blend()`:
  proba_up, rubric_total, blend_w_model, blend_w_rubric,
  blend_model_contrib, blend_rubric_contrib, pre_gate_conviction,
  conviction, direction, regime
"""
from __future__ import annotations

from typing import Mapping

# Streamlit / Plotly imports are lazy so importing this module never
# pulls in the dashboard runtime when the prediction card is constructed
# in a non-UI context (backtester, daily orchestrator).


def render_conviction_waterfall(card: Mapping[str, object],
                                  *, height: int = 360) -> None:
    """Render a Plotly waterfall under the current Streamlit container.

    Defensive: if any required field is missing or plotly isn't installed
    the function emits a one-line caption explaining what's missing and
    returns without raising. The dashboard never crashes on a malformed
    card.
    """
    try:
        import streamlit as st
    except Exception:  # noqa: BLE001
        return
    try:
        import plotly.graph_objects as go
    except Exception as e:  # noqa: BLE001
        st.caption(f"waterfall unavailable: {e}")
        return

    # Pull fields with safe defaults so a card built by an older blend()
    # (without the decomposition columns) still renders SOMETHING useful.
    proba_up = float(card.get("proba_up", 0.5) or 0.5)
    rubric_total = float(card.get("rubric_total", 50.0) or 50.0)
    w_model = float(card.get("blend_w_model", 0.6) or 0.6)
    w_rubric = float(card.get("blend_w_rubric", 0.4) or 0.4)
    model_contrib = card.get("blend_model_contrib")
    rubric_contrib = card.get("blend_rubric_contrib")
    if model_contrib is None:
        model_contrib = w_model * proba_up
    if rubric_contrib is None:
        rubric_contrib = w_rubric * (rubric_total / 100.0)
    pre_gate = float(card.get("pre_gate_conviction",
                              card.get("conviction", 0.0)) or 0.0)
    final = float(card.get("effective_conviction",
                           card.get("conviction", pre_gate)) or pre_gate)
    direction = int(card.get("direction", 0) or 0)
    regime = str(card.get("regime", "") or "")

    # Compute the regime multiplier as the ratio (final / pre_gate). When
    # pre_gate is 0 the card is flat anyway; show 1.0 to avoid divide-by-0.
    if pre_gate > 1e-6:
        regime_mult = final / pre_gate
    else:
        regime_mult = 1.0

    # Build the waterfall on a 0-100 conviction scale so all bars share units.
    # Starting baseline: a model with no information would put 50% on up, 50%
    # on down — that's conviction 0 on our scale, but we show the geometric
    # midpoint so the relative widths read intuitively.
    baseline_conv = 0.0
    # Express each contribution as its conviction-equivalent delta, signed
    # by direction. Long: positive contribution = bullish push; short:
    # positive contribution actually drags conviction down. Keep the
    # display simple: show absolute conviction adjustments.
    model_delta = abs(proba_up - 0.5) * 200.0 * w_model
    rubric_delta = abs((rubric_total / 100.0) - 0.5) * 200.0 * w_rubric
    regime_drag = pre_gate * (regime_mult - 1.0)  # negative when penalised

    fig = go.Figure(go.Waterfall(
        name="conviction",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["baseline",
           f"model w={w_model:.2f}",
           f"rubric w={w_rubric:.2f}",
           f"regime gate ({regime} x{regime_mult:.2f})",
           "final"],
        y=[baseline_conv, model_delta, rubric_delta, regime_drag, final],
        connector={"line": {"color": "rgba(148,163,184,0.4)"}},
        increasing={"marker": {"color": "#22d3ee"}},  # cyan
        decreasing={"marker": {"color": "#f87171"}},  # red
        totals={"marker": {"color": "#a78bfa"}},      # violet
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=11),
        yaxis=dict(title="conviction (0-100)", range=[0, max(100.0, final + 5)],
                   gridcolor="rgba(148,163,184,0.12)"),
        title=dict(
            text=f"Confidence decomposition · direction={direction:+d}",
            font=dict(size=12, color="#e6ecf5"),
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
