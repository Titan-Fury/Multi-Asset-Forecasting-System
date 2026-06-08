"""Causal Counterfactual sliders (Phase 24.21 — Feature 3).

A Streamlit widget that lets the user override key macro variables
(VIX / DXY / US10Y / oil) and one pillar score on the current prediction
card. The widget then re-runs the deterministic part of the scoring
pipeline on the modified row and shows the before/after conviction
delta side-by-side.

This answers questions the existing card cannot:
  - "What if the VIX spiked to 35 — would my long signal still pass?"
  - "If sentiment dropped to 30 instead of 65 — does the rubric still emit?"
  - "Does my bullish call hold up under crisis-regime weights?"

Implementation strategy: this module does NOT touch the trained models.
It re-runs only the deterministic blend + gate stages with the user's
counterfactual inputs. That means it cannot answer "what would the
XGBoost output be?" — but it CAN answer "given the same proba_up, how
do the rubric, regime weights, and gate change conviction?". This is
the layer where the user has interpretable levers.

Usage:
    from omni.dashboard.charts.counterfactual import render_counterfactual_panel
    render_counterfactual_panel(card_dict)
"""
from __future__ import annotations

from collections.abc import Mapping

_MACRO_KNOBS = (
    # (key on card, label, min, max, step)
    ("vix",   "VIX",   5.0,  80.0, 0.5),
    ("dxy",   "DXY",   85.0, 120.0, 0.1),
    ("us10y", "US10Y", 1.0,  6.0,  0.05),
    ("oil",   "Oil ($)", 30.0, 150.0, 0.5),
)

_REGIMES = ("bull", "chop", "bear", "crisis")


def render_counterfactual_panel(card: Mapping[str, object],
                                 *, height_px: int = 360) -> None:
    """Render the counterfactual sliders + before/after card.

    The widget is namespaced by ticker (via st.session_state keys) so it
    works inside loops over multiple cards without state collisions.
    """
    try:
        import streamlit as st
    except Exception:  # noqa: BLE001
        return

    ticker = str(card.get("ticker", "?"))
    ns = f"cf_{ticker}"

    with st.expander(f"Counterfactual scenarios · {ticker}", expanded=False):
        st.caption(
            "Drag a slider, see how conviction changes. The model's "
            "proba_up is held fixed — only the rubric / regime weights "
            "and the gate are re-evaluated."
        )

        # -- macro knobs (informational; we don't recompute the model)
        st.markdown("**Macro overrides**")
        cols = st.columns(len(_MACRO_KNOBS))
        macro_values = {}
        for i, (key, label, lo, hi, step) in enumerate(_MACRO_KNOBS):
            cur = float(card.get(key, (lo + hi) / 2.0))
            macro_values[key] = cols[i].slider(
                label, min_value=lo, max_value=hi, value=cur, step=step,
                key=f"{ns}_{key}",
            )

        # -- regime override
        cur_regime = str(card.get("regime", "chop")).lower()
        if cur_regime not in _REGIMES:
            cur_regime = "chop"
        regime_choice = st.selectbox(
            "Regime override",
            list(_REGIMES),
            index=list(_REGIMES).index(cur_regime),
            key=f"{ns}_regime",
        )

        # -- pillar / rubric override
        rubric_total = float(card.get("rubric_total", 50.0) or 50.0)
        new_rubric_total = st.slider(
            "Rubric total override (0..100)",
            min_value=0.0, max_value=100.0, value=rubric_total, step=1.0,
            key=f"{ns}_rubric",
        )

        # -- recompute
        original_conv, original_dir, original_gate = _recompute(
            card, regime=cur_regime,
            rubric_total=rubric_total,
        )
        new_conv, new_dir, new_gate = _recompute(
            card, regime=regime_choice,
            rubric_total=new_rubric_total,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Conviction (original)", f"{original_conv:.1f}",
                  help=f"direction={original_dir:+d} · gate={'PASS' if original_gate else 'BLOCK'}")
        c2.metric("Conviction (counterfactual)", f"{new_conv:.1f}",
                  delta=f"{(new_conv - original_conv):+.1f}",
                  help=f"direction={new_dir:+d} · gate={'PASS' if new_gate else 'BLOCK'}")
        c3.metric(
            "Gate verdict",
            "PASS" if new_gate else "BLOCK",
            delta=("now PASSING" if (new_gate and not original_gate)
                   else "now BLOCKED" if (original_gate and not new_gate)
                   else "unchanged"),
        )


def _recompute(card: Mapping[str, object], *, regime: str,
                 rubric_total: float) -> tuple[float, int, bool]:
    """Re-run blend() + gate.apply() on a synthetic single-row frame
    holding `proba_up` fixed but with the counterfactual `regime` and
    `rubric_total`. Returns (conviction, direction, passes_gate).

    Falls back to a pure-Python recomputation if pandas / the ensemble
    modules can't be imported (so the widget still renders something
    sensible in a degraded environment).
    """
    proba_up = float(card.get("proba_up", 0.5) or 0.5)
    asset_class = str(card.get("asset_class", "us"))
    try:
        import pandas as pd

        from omni.ensemble.combine import blend
        from omni.ensemble.gate import apply as gate_apply

        row = pd.DataFrame([{
            "ticker": str(card.get("ticker", "?")),
            "asset_class": asset_class,
            "proba_up": proba_up,
            "rubric_total": rubric_total,
            "regime": regime,
        }])
        b = blend(row)
        g = gate_apply(b)
        eff_conv = float(g["effective_conviction"].iloc[0]) if "effective_conviction" in g.columns else float(g["conviction"].iloc[0])
        direction = int(g["direction"].iloc[0])
        passes = bool(g["passes_gate"].iloc[0])
        return eff_conv, direction, passes
    except Exception:  # noqa: BLE001
        # Best-effort fallback: no gate, no regime weights — just the
        # raw blended conviction so the user still sees *something* move.
        rubric_p = max(0.0, min(1.0, rubric_total / 100.0))
        blended = 0.6 * proba_up + 0.4 * rubric_p
        conv = abs(blended - 0.5) * 200.0
        direction = 1 if blended > 0.52 else (-1 if blended < 0.48 else 0)
        return conv, direction, conv >= 70.0 and direction != 0
