"""Backtest plots.

matplotlib-based plot helpers; each function returns the Figure so callers
can save / display as needed. No side effects unless `path=...` is provided.

Plots:
  calibration_plot(predictions)   -- reliability diagram + perfect-calibration line
  decile_lift_plot(predictions)   -- bar chart of mean realized return per proba decile
  equity_curve(signals)           -- cumulative signed-return curve from gated signals
  fold_hit_rate_plot(fold_metrics)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from omni.backtest.metrics import calibration_table, decile_lift


def _fig():
    import matplotlib.pyplot as plt
    return plt


def calibration_plot(preds: pd.DataFrame, *, n_bins: int = 10,
                     path: str | Path | None = None):
    plt = _fig()
    tab = calibration_table(preds, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.scatter(tab["bin_mean_p"], tab["bin_mean_y"], s=tab["n"] / max(1, tab["n"].max()) * 200)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Calibration (size ∝ bin n)")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def decile_lift_plot(preds: pd.DataFrame, *, path: str | Path | None = None):
    plt = _fig()
    tab = decile_lift(preds)
    if tab.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(tab["decile"], tab["mean_fwd_ret"] * 100)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Probability decile (1=lowest, 10=highest)")
    ax.set_ylabel("Mean realized 5d return (%)")
    ax.set_title("Decile lift")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def equity_curve(signals: pd.DataFrame, *, path: str | Path | None = None):
    """Daily cumulative signed return from gated signals only."""
    plt = _fig()
    if signals.empty or "fwd_ret_5d" not in signals.columns:
        return None
    sub = signals[signals.get("passes_gate", True)].copy()
    if sub.empty:
        return None
    sub["signed"] = sub["direction"] * sub["fwd_ret_5d"]
    daily = sub.groupby("date")["signed"].mean().sort_index()
    equity = (1 + daily.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(equity.index, equity.values)
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative signed return (gated)")
    ax.set_title("Gated-signal equity curve (5d non-overlapping approx)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def fold_hit_rate_plot(fold_metrics: list[dict], *, path: str | Path | None = None):
    plt = _fig()
    if not fold_metrics:
        return None
    folds = [fm["fold"] for fm in fold_metrics]
    hits = [fm["hit_rate"] for fm in fold_metrics]
    aucs = [fm.get("auc", np.nan) for fm in fold_metrics]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(folds, hits, "o-", label="hit_rate")
    ax.plot(folds, aucs, "s--", label="auc")
    ax.axhline(0.5, color="gray", linewidth=0.8)
    ax.set_ylim(0.3, 0.8)
    ax.set_xlabel("fold")
    ax.set_title("Per-fold metrics")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig
