"""Backtest metrics.

All metrics here operate on a frame with at minimum:
  proba_up   -- predicted probability fwd_dir_5d > 0
  y_true     -- 0/1 ground truth
  fwd_ret_5d -- realized 5d forward return (for PnL-aware metrics)

Computed:
  hit_rate         -- accuracy at p=0.5 cutoff
  brier            -- mean squared error of probabilities (lower is better)
  log_loss         -- binary cross-entropy
  auc              -- ROC AUC (rank-based, threshold-independent)
  calibration      -- table of (bin_mean_p, bin_mean_y, n) for reliability plot
  top_decile       -- precision in the top-10% predicted-up bucket
  bottom_decile    -- precision in the bottom-10% predicted-up bucket (i.e.
                      hit-rate of shorts when we predict down)
  decile_lift      -- mean realized return per decile of proba (1..10)

These are intended to be reported per fold and aggregated across all folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hit_rate(df: pd.DataFrame, *, threshold: float = 0.5) -> float:
    pred = (df["proba_up"] >= threshold).astype(int)
    return float((pred == df["y_true"]).mean())


def brier(df: pd.DataFrame) -> float:
    return float(((df["proba_up"] - df["y_true"]) ** 2).mean())


def log_loss(df: pd.DataFrame, *, eps: float = 1e-12) -> float:
    p = df["proba_up"].clip(eps, 1 - eps)
    y = df["y_true"]
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def auc(df: pd.DataFrame) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    if df["y_true"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(df["y_true"], df["proba_up"]))


def calibration_table(df: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    cuts = pd.cut(df["proba_up"], bins=bins, include_lowest=True)
    g = df.groupby(cuts, observed=True)
    return pd.DataFrame({
        "bin_mean_p": g["proba_up"].mean(),
        "bin_mean_y": g["y_true"].mean(),
        "n": g.size(),
    }).reset_index(drop=True)


def top_decile(df: pd.DataFrame, *, q: float = 0.9) -> dict:
    cutoff = df["proba_up"].quantile(q)
    sub = df[df["proba_up"] >= cutoff]
    if sub.empty:
        return {"cutoff": float(cutoff), "n": 0, "precision": float("nan"),
                "mean_fwd_ret": float("nan")}
    return {
        "cutoff": float(cutoff),
        "n": int(len(sub)),
        "precision": float(sub["y_true"].mean()),
        "mean_fwd_ret": float(sub["fwd_ret_5d"].mean()) if "fwd_ret_5d" in sub.columns else float("nan"),
    }


def bottom_decile(df: pd.DataFrame, *, q: float = 0.1) -> dict:
    cutoff = df["proba_up"].quantile(q)
    sub = df[df["proba_up"] <= cutoff]
    if sub.empty:
        return {"cutoff": float(cutoff), "n": 0, "precision_short": float("nan"),
                "mean_fwd_ret": float("nan")}
    return {
        "cutoff": float(cutoff),
        "n": int(len(sub)),
        "precision_short": float((sub["y_true"] == 0).mean()),
        "mean_fwd_ret": float(sub["fwd_ret_5d"].mean()) if "fwd_ret_5d" in sub.columns else float("nan"),
    }


def decile_lift(df: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    if "fwd_ret_5d" not in df.columns:
        return pd.DataFrame()
    qcut = pd.qcut(df["proba_up"], n_bins, labels=False, duplicates="drop")
    g = df.assign(decile=qcut + 1).groupby("decile")
    return pd.DataFrame({
        "decile": g.size().index,
        "n": g.size().values,
        "mean_proba": g["proba_up"].mean().values,
        "hit_rate": g["y_true"].mean().values,
        "mean_fwd_ret": g["fwd_ret_5d"].mean().values,
    }).reset_index(drop=True)


def gated_metrics(df: pd.DataFrame) -> dict:
    """Metrics for the rows that passed the conviction gate.

    Expects `passes_gate`, `direction`, and `fwd_ret_5d` columns.
    """
    if "passes_gate" not in df.columns:
        return {}
    sub = df[df["passes_gate"]].copy()
    n = len(sub)
    if n == 0:
        return {"n_emitted": 0}
    # Realized direction success: direction==+1 & fwd_ret>0  OR  direction==-1 & fwd_ret<0
    if "fwd_ret_5d" in sub.columns:
        success = ((sub["direction"] == 1) & (sub["fwd_ret_5d"] > 0)) | \
                  ((sub["direction"] == -1) & (sub["fwd_ret_5d"] < 0))
        signed_ret = sub["direction"] * sub["fwd_ret_5d"]
    else:
        success = pd.Series([], dtype=bool)
        signed_ret = pd.Series([], dtype=float)
    return {
        "n_emitted": n,
        "emit_rate": float(n / max(1, len(df))),
        "directional_hit_rate": float(success.mean()) if n > 0 else float("nan"),
        "mean_signed_fwd_ret": float(signed_ret.mean()) if n > 0 else float("nan"),
    }


def summary(df: pd.DataFrame) -> dict:
    out = {
        "n": int(len(df)),
        "hit_rate": hit_rate(df),
        "brier": brier(df),
        "log_loss": log_loss(df),
        "auc": auc(df),
    }
    out["top_decile"] = top_decile(df)
    out["bottom_decile"] = bottom_decile(df)
    if "passes_gate" in df.columns:
        out["gated"] = gated_metrics(df)
    return out
