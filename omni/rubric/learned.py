"""Learned rubric weights — logistic regression over the 9 pillar scores.

The hand-tuned weights in `config/rubric.yaml` were never fit to data.
A simple logistic on (pillar_scores -> fwd_dir_5d) recovers the
per-regime predictive contribution of each pillar from history.

Trained per regime so each regime gets its own coefficient vector
(crisis cares more about macro; bull cares more about momentum).

API (used at training time only):

    weights = train_learned_weights(df_train, horizon=5,
                                       target="fwd_dir_5d")
    # weights: {regime -> {pillar -> float coefficient}}
    save_learned_weights(weights, path)

API (used at inference time, opt-in):

    from omni.rubric.learned import load_learned_weights, blend_weights
    learned = load_learned_weights()  # cached; None if file missing
    if learned and OMNI_LEARNED_RUBRIC:
        weights = blend_weights(yaml_weights, learned[regime], alpha=0.5)

When opted in, the YAML weights and learned weights are blended 50/50
so a bad fit on a small training set can't catastrophically zero out
a pillar that has a sound prior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from omni.utils.logger import log
from omni.utils.paths import local_data_root


_PILLAR_COLS = (
    "technical_score", "fundamental_score", "sentiment_score",
    "macro_score", "microstructure_score", "oi_score",
    "relative_strength_score", "insider_score", "earnings_momentum_score",
)
# YAML pillar key -> internal column name in the scored frame
_PILLAR_KEYS = (
    "technical", "fundamental", "sentiment", "macro",
    "microstructure", "oi", "relative_strength", "insider",
    "earnings_momentum",
)


def _path() -> Path:
    return local_data_root() / "models" / "learned_rubric_weights.json"


def train_learned_weights(scored_df: pd.DataFrame,
                           *,
                           target: str = "fwd_dir_5d",
                           horizon: int = 5,
                           regime_col: str = "regime",
                           min_rows_per_regime: int = 500,
                           ) -> dict:
    """Fit per-regime logistic regression on pillar scores -> binary target.

    `scored_df` must have:
      - all 9 pillar score columns (e.g. technical_score)
      - the target column (default fwd_dir_5d)
      - the regime column (default 'regime')

    Returns a dict {regime: {pillar_key: coefficient}}. Coefficients are
    L1-normalised within each regime so they sum to 1.0 in absolute value
    — directly drop-in compatible with the YAML weights schema.
    """
    from sklearn.linear_model import LogisticRegression

    needed = set(_PILLAR_COLS) | {target, regime_col}
    missing = needed - set(scored_df.columns)
    if missing:
        raise ValueError(f"learned-rubric: missing columns {sorted(missing)}")

    df = scored_df.dropna(subset=[target, regime_col]).copy()
    df["_y"] = (df[target].astype(int) > 0).astype(int)

    out: dict[str, dict[str, float]] = {}
    for regime, sub in df.groupby(regime_col):
        if len(sub) < min_rows_per_regime:
            log.warning(f"learned-rubric: {regime} only {len(sub)} rows < min "
                          f"{min_rows_per_regime}; skipping")
            continue
        X = sub[list(_PILLAR_COLS)].apply(pd.to_numeric, errors="coerce") \
            .fillna(50.0).clip(0, 100).to_numpy()
        # Center on 50 so coefficients reflect "above-neutral lifts the
        # probability"; magnitude is what matters for the rubric weights.
        X = (X - 50.0) / 50.0
        y = sub["_y"].to_numpy()
        if len(np.unique(y)) < 2:
            log.warning(f"learned-rubric: {regime} single-class y; skipping")
            continue
        # L2 regularisation; balanced class weights so up-skewed regimes
        # don't dominate.
        lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=300,
                                  class_weight="balanced")
        lr.fit(X, y)
        coefs = np.asarray(lr.coef_, dtype=float).ravel()
        # Clip to non-negative — pillar scores are designed so HIGHER means
        # MORE positive expected direction. A negative learned coefficient
        # means the pillar is anti-predictive, which we treat as "zero
        # weight" rather than flip the sign.
        coefs = np.maximum(coefs, 0.0)
        # L1-normalise to sum 1.0 (matching YAML weight convention)
        s = coefs.sum()
        if s <= 0:
            log.warning(f"learned-rubric: {regime} all-zero coefs; using uniform")
            coefs = np.ones(len(_PILLAR_COLS)) / len(_PILLAR_COLS)
        else:
            coefs = coefs / s
        out[str(regime)] = {k: float(c) for k, c in zip(_PILLAR_KEYS, coefs)}
        log.info(f"learned-rubric[{regime}] n={len(sub)}: "
                  f"top3="
                  f"{sorted(out[str(regime)].items(), key=lambda kv: -kv[1])[:3]}")
    return out


def save_learned_weights(weights: dict, path: Path | str | None = None) -> Path:
    """Persist {regime -> {pillar_key -> coef}} to a JSON file."""
    import json
    p = Path(path) if path else _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(weights, indent=2))
    log.info(f"learned-rubric saved -> {p}")
    # Phase 24.22 — invalidate the in-process cache so a process that
    # re-trains and re-saves picks up the new weights on the very next
    # load_learned_weights() call. Previously the cache was permanent and
    # returned stale weights for the rest of the process lifetime.
    clear_cache()
    return p


_CACHE: dict | None = None
_CACHE_MTIME: float | None = None  # Phase 24.22 — file mtime sentinel


def load_learned_weights(path: Path | str | None = None) -> dict | None:
    """Load saved learned weights. Returns None if the file is missing.

    Phase 24.22 — caches are validated against the on-disk mtime so a
    long-running process (e.g. the dashboard) picks up retrained weights
    without restart. The previous version cached forever.
    """
    global _CACHE, _CACHE_MTIME
    import json
    p = Path(path) if path else _path()
    if not p.exists():
        return None
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = None
    if _CACHE is not None and _CACHE_MTIME == mtime:
        return _CACHE
    try:
        _CACHE = json.loads(p.read_text())
        _CACHE_MTIME = mtime
        return _CACHE
    except Exception as exc:  # noqa: BLE001
        log.warning(f"learned-rubric load failed: {exc}")
        return None


def clear_cache() -> None:
    global _CACHE, _CACHE_MTIME
    _CACHE = None
    _CACHE_MTIME = None


def blend_weights(yaml_weights: dict, learned_weights: dict, *,
                    alpha: float = 0.5) -> dict:
    """Blend the YAML prior with the learned posterior.

    `alpha` is the learned weight (0 = pure YAML, 1 = pure learned).
    Both inputs are dicts of pillar_key -> float; output sums to 1.0.
    """
    keys = set(yaml_weights) | set(learned_weights)
    out: dict[str, float] = {}
    for k in keys:
        y = float(yaml_weights.get(k, 0.0))
        l = float(learned_weights.get(k, 0.0))
        out[k] = (1 - alpha) * y + alpha * l
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out
