"""Regime Transition Heatmap (Phase 24.21 — Feature 2).

Surfaces the HMM state dynamics that the existing regime model already
computes but never shows the user:

  1. Transition matrix      P(next_regime | current_regime)
     Renders as a 4x4 heatmap. Values close to 1.0 on the diagonal mean
     the regime is sticky; off-diagonal mass tells you which regime is
     the most likely next state.

  2. Posterior probability time-series of being in each regime, smoothed
     across the last ~120 trading days. Lets the user spot the slow
     ramp into a regime change *before* the labelled state flips.

  3. Crisis early-warning chip when P(crisis | today) crosses a threshold
     (default 15 %). The label may still read 'bull' but the posterior
     leakage tells you the model is hedging.

This module is read-only — it never refits the HMM. It pulls posteriors
from `RegimeModel.predict_proba(features)` and the transition matrix from
the underlying hmmlearn model's `transmat_`.

Usage:
    from omni.dashboard.charts.regime_heatmap import (
        render_transition_matrix,
        render_posterior_timeseries,
        crisis_warning_chip,
    )
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_STATE_ORDER = ("bull", "chop", "bear", "crisis")


def _safe_transmat(model) -> np.ndarray | None:
    """Pull the transition matrix off whichever underlying object the
    `RegimeModel` is wrapping (hmmlearn.GaussianHMM by default). Returns
    None if not exposed.
    """
    for attr in ("transmat_", "_hmm"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        if attr == "_hmm":
            tm = getattr(obj, "transmat_", None)
            if tm is not None:
                return np.asarray(tm)
        else:
            return np.asarray(obj)
    return None


def render_transition_matrix(model, *,
                             state_labels: Sequence[str] | None = None,
                             height: int = 320) -> None:
    """Render the HMM transition matrix as a 4x4 heatmap. Defensive: falls
    through with a one-line caption if hmmlearn / plotly aren't installed
    or the model exposes no transmat_."""
    try:
        import streamlit as st
    except Exception:  # noqa: BLE001
        return
    try:
        import plotly.graph_objects as go
    except Exception as e:  # noqa: BLE001
        st.caption(f"transition heatmap unavailable: {e}")
        return

    tm = _safe_transmat(model)
    if tm is None:
        st.caption("regime model has no transition matrix exposed")
        return
    labels = list(state_labels or _STATE_ORDER)[:tm.shape[0]]
    fig = go.Figure(go.Heatmap(
        z=tm,
        x=labels, y=labels,
        zmin=0, zmax=1,
        colorscale="Viridis",
        text=[[f"{v:.2f}" for v in row] for row in tm],
        texttemplate="%{text}",
        textfont=dict(size=11, color="#e6ecf5"),
        colorbar=dict(title="P(j|i)", thickness=10, len=0.9),
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=11),
        title=dict(text="Regime transition probabilities",
                   font=dict(size=12, color="#e6ecf5")),
        xaxis=dict(title="next regime"),
        yaxis=dict(title="current regime", autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_posterior_timeseries(probs: pd.DataFrame, *,
                                lookback: int = 120,
                                height: int = 280) -> None:
    """Render P(state) over the last `lookback` rows.

    `probs` must have one row per date and at least the four state columns
    (bull/chop/bear/crisis). A 'date' column or DatetimeIndex is used for
    the x-axis if present.
    """
    try:
        import streamlit as st
    except Exception:  # noqa: BLE001
        return
    try:
        import plotly.graph_objects as go
    except Exception as e:  # noqa: BLE001
        st.caption(f"posterior chart unavailable: {e}")
        return
    if probs is None or probs.empty:
        st.caption("no regime posterior data yet")
        return

    df = probs.tail(lookback).copy()
    if "date" in df.columns:
        x = pd.to_datetime(df["date"])
    else:
        x = pd.to_datetime(df.index)

    palette = {
        "bull":   "#22d3ee",  # cyan
        "chop":   "#a78bfa",  # violet
        "bear":   "#fb923c",  # orange
        "crisis": "#f87171",  # red
    }
    fig = go.Figure()
    for state in _STATE_ORDER:
        col = state if state in df.columns else f"regime_p_{state}"
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=x, y=df[col].astype(float),
            mode="lines",
            stackgroup="one",
            name=state,
            line=dict(width=0.5, color=palette[state]),
            hovertemplate=f"{state}: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=11),
        title=dict(text=f"Regime posteriors · last {lookback} sessions",
                   font=dict(size=12, color="#e6ecf5")),
        yaxis=dict(range=[0, 1], gridcolor="rgba(148,163,184,0.12)"),
        legend=dict(orientation="h", y=-0.18),
    )
    st.plotly_chart(fig, use_container_width=True)


def crisis_warning_chip(probs: pd.DataFrame, *,
                        threshold: float = 0.15) -> tuple[bool, float]:
    """Return (warn, latest_p_crisis). When `warn=True` the dashboard
    should surface a coloured chip — the labelled regime may still be
    benign, but the posterior probability of crisis has crept above the
    threshold which is a known leading indicator (Apr-2025 tariff shock,
    Mar-2020 COVID).

    Defensive: returns (False, 0.0) if data is missing.
    """
    if probs is None or probs.empty:
        return False, 0.0
    col = "crisis" if "crisis" in probs.columns else "regime_p_crisis"
    if col not in probs.columns:
        return False, 0.0
    try:
        latest = float(probs[col].iloc[-1])
    except Exception:  # noqa: BLE001
        return False, 0.0
    return latest >= float(threshold), latest
