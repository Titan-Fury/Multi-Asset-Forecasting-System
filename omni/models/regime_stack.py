"""Regime-conditional stack routing (Phase 25 / Commit E).

Idea: train one StackResult per regime label (bull / chop / bear /
crisis), then at inference blend their predictions by the HMM's
per-state posterior `proba_today`. Captures regime-specific feature
relationships that a single global stack averages over.

This is a thin wrapper around `omni.models.stack.StackResult` — each
sub-stack is a regular stack that was fit on a regime-sliced subset of
the training rows. The `RegimeStackBundle` exposes the same
`predict_proba(X)` signature plus an optional `regime_proba` dict for
soft routing.

Training: see `scripts/train_regime_stacks.py`.

Inference: gated by `OMNI_REGIME_STACKS=1` so the existing single-stack
production path remains the default until empirically validated.

API:
    bundle = RegimeStackBundle.load("data/models/regime_stacks_h5.joblib")
    p = bundle.predict_proba(X, regime_proba={"bull": 0.7, "chop": 0.3, ...})
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omni.utils.logger import log

_REGIME_LABELS = ("bull", "chop", "bear", "crisis")


@dataclass
class RegimeStackBundle:
    """Holds N stacks keyed by regime label; routes by regime_proba."""

    stacks: dict[str, Any]                                # label -> StackResult
    feature_cols: list[str]                                # union across sub-stacks (for alignment)
    horizon_days: int = 5
    oof_aucs: dict[str, float] = field(default_factory=dict)
    n_rows_per_regime: dict[str, int] = field(default_factory=dict)
    trained_at: str = ""

    def predict_proba(self, X: pd.DataFrame, *,
                       regime_proba: dict[str, float] | None = None,
                       regime_label: str | None = None) -> np.ndarray:
        """Predict P(y=1) by routing through regime-specific stacks.

        - If `regime_proba` is provided (preferred), each sub-stack
          contributes its prediction weighted by the corresponding
          posterior. Missing labels are treated as 0 weight.
        - Else if `regime_label` is provided, only that sub-stack runs.
        - Else uniform 1/N average across available sub-stacks.
        """
        # Compute each sub-stack prediction (lazy: skip if weight ~0)
        weights = self._weights(regime_proba=regime_proba, regime_label=regime_label)
        # Normalise once
        total = sum(weights.values())
        if total <= 0:
            return np.full(len(X), 0.5, dtype=float)

        accum = np.zeros(len(X), dtype=float)
        for label, w in weights.items():
            if w <= 0:
                continue
            stk = self.stacks.get(label)
            if stk is None:
                continue
            try:
                p = stk.predict_proba(X)
                accum += (w / total) * np.asarray(p, dtype=float).ravel()
            except Exception as exc:  # noqa: BLE001
                log.warning(f"regime_stack[{label}] predict failed: {exc}")
                # Fall back to neutral 0.5 for this slice so the bundle
                # doesn't silently drop a regime's contribution.
                accum += (w / total) * 0.5
        return accum

    def _weights(self, *, regime_proba: dict[str, float] | None,
                  regime_label: str | None) -> dict[str, float]:
        if regime_proba:
            return {lbl: float(regime_proba.get(lbl, 0.0)) for lbl in self.stacks.keys()}
        if regime_label:
            return {lbl: (1.0 if lbl == regime_label else 0.0)
                     for lbl in self.stacks.keys()}
        # Uniform fallback
        n = max(1, len(self.stacks))
        return {lbl: 1.0 / n for lbl in self.stacks.keys()}

    @staticmethod
    def save(bundle: RegimeStackBundle, path: str | Path) -> Path:
        import joblib
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, p)
        log.info(f"regime_stacks saved -> {p} "
                  f"(regimes={list(bundle.stacks.keys())}, aucs={bundle.oof_aucs})")
        return p

    @staticmethod
    def load(path: str | Path) -> RegimeStackBundle | None:
        import joblib
        p = Path(path)
        if not p.exists():
            return None
        try:
            return joblib.load(p)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"regime_stacks load failed: {exc}")
            return None


def default_path(horizon: int = 5) -> Path:
    return Path("data") / "models" / f"regime_stacks_h{horizon}.joblib"
