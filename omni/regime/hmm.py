"""Hidden Markov Model regime detector.

A 4-state Gaussian HMM trained on a small set of macro/market features whose
states we then *label* (post-fit) as one of: bull / bear / chop / crisis.

Inputs (per business day, US session):
  - log return of SPX (ret)
  - 20d realized vol (rv)
  - 20d breadth proxy: pct of S&P 500 stocks above 50dma (or fallback: rolling
    mean(sign(ret_1d)) over the universe)
  - VIX level (vix)

Labelling heuristic (applied to the fitted state means):
  crisis = highest vol AND highest VIX
  bear   = lowest mean return among the rest
  bull   = highest mean return among the rest
  chop   = remaining state

The HMM is fit ONCE on a training window (walk-forward by the caller). Predicted
states are then mapped to labels using the heuristic above. We expose:
  fit(features) -> RegimeModel
  predict(features) -> array of labels per row
  predict_proba(features) -> per-state posterior (smoothed)

Anti-leakage: when the caller uses this in a walk-forward backtest, fit on
[t0, t_train_end] only and predict on [t_train_end+1, t_test_end].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from omni.utils.logger import log

LABELS = ("bull", "bear", "chop", "crisis")


@dataclass
class RegimeModel:
    hmm: object  # hmmlearn.hmm.GaussianHMM
    feature_cols: list[str]
    state_to_label: dict[int, str]
    converged: bool = True   # Phase 24.16 — surface convergence status
    fallback: bool = False   # True when HMM failed and we returned a stub

    def predict(self, X: pd.DataFrame, *, smooth_window: int = 5) -> np.ndarray:
        """Predict regime labels.

        Phase 24.16 — `smooth_window` adds a min-duration constraint so a
        single-day spike doesn't cause a regime flip. The default 5
        trading days mirrors the typical HMM stickiness in finance lit
        (Hamilton 1989, Hardy 2001). Set 0 to disable smoothing.
        """
        x = X[self.feature_cols].to_numpy(dtype=float)
        states = self.hmm.predict(x)
        if smooth_window and len(states) >= smooth_window:
            states = _smooth_states(states, window=smooth_window)
        return np.array([self.state_to_label[s] for s in states])

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.feature_cols].to_numpy(dtype=float)
        post = self.hmm.predict_proba(x)
        cols = [self.state_to_label[i] for i in range(post.shape[1])]
        # Collapse duplicate label columns (rare but possible) by max-proba
        df = pd.DataFrame(post, columns=cols)
        # pandas 2.2+: axis=1 in groupby is removed; transpose-groupby-transpose
        return df.T.groupby(level=0).max().T


def _smooth_states(states: np.ndarray, *, window: int = 5) -> np.ndarray:
    """Min-duration smoothing — collapse runs shorter than `window` into
    the dominant neighbouring state. Reduces regime jitter without losing
    real transitions.
    """
    out = states.copy()
    n = len(out)
    if n < 2:
        return out
    i = 0
    while i < n:
        j = i
        while j + 1 < n and out[j + 1] == out[i]:
            j += 1
        run_len = j - i + 1
        if run_len < window and i > 0:
            # Replace short run with the previous state
            out[i:j + 1] = out[i - 1]
        i = j + 1
    return out


def _label_states(means: np.ndarray, feature_cols: list[str]) -> dict[int, str]:
    """means: shape (n_states, n_features). Map state idx -> label.

    Phase 24.16 — handles tied means. Previously when two states shared
    the same mean return (common with degenerate / pre-shuffled data),
    `bull_state` and `bear_state` would resolve to the same state index
    and the remaining state would end up UNLABELED. Calling predict()
    on that unlabelled state then raised KeyError. Now every state idx
    gets a label, with deterministic tie-breaks by index.
    """
    n = means.shape[0]
    if n != 4:
        # Fallback: just sort by mean of feature[0] (return)
        order = np.argsort(means[:, 0])
        labels = ["bear", "chop", "chop", "bull"][:n]
        return {int(s): labels[i] for i, s in enumerate(order)}

    ret_idx = feature_cols.index("ret") if "ret" in feature_cols else 0
    vix_idx = feature_cols.index("vix") if "vix" in feature_cols else -1
    rv_idx = feature_cols.index("rv") if "rv" in feature_cols else -1

    # Crisis: highest VIX (or highest realized vol if VIX missing).
    # On ties, np.argmax picks the lowest index (deterministic).
    # Defensive: if BOTH vix and rv are missing, the previous code silently
    # used means[:, -1] which indexes the last column — wrong. Fall back
    # to the absolute return magnitude as a severity proxy instead.
    if vix_idx >= 0:
        severity = means[:, vix_idx]
    elif rv_idx >= 0:
        severity = means[:, rv_idx]
    else:
        severity = np.abs(means[:, ret_idx])
    crisis_state = int(np.argmax(severity))

    others = sorted(s for s in range(n) if s != crisis_state)
    # Sort the remaining states by their return mean. The lowest-return
    # state is "bear", the highest is "bull", the middle is "chop".
    # If ties exist, ordering by state index keeps the assignment stable.
    others_sorted = sorted(
        others,
        key=lambda s: (float(means[s, ret_idx]), s),
    )
    bear_state = others_sorted[0]
    chop_state = others_sorted[1]
    bull_state = others_sorted[2]
    out = {
        crisis_state: "crisis",
        bear_state: "bear",
        chop_state: "chop",
        bull_state: "bull",
    }
    # Sanity: every state index 0..n-1 must be in the map.
    if set(out.keys()) != set(range(n)):
        # Should be impossible, but guard so we never return a partial map.
        log.warning(f"_label_states partial map {out}; filling missing")
        for s in range(n):
            out.setdefault(s, "chop")
    return out


def fit(features: pd.DataFrame, *, n_states: int = 4,
        feature_cols: tuple[str, ...] = ("ret", "rv", "vix"),
        seed: int = 42, n_iter: int = 200) -> RegimeModel:
    """Fit a Gaussian HMM. Caller must restrict `features` to the train window.

    Phase 24.16 — surfaces non-convergence and emits a metric. Returns a
    `RegimeModel` with `.converged` set so downstream code can decide
    whether to trust the labels. Falls back to a single-state "chop"
    classifier if the HMM hard-fails (e.g. singular covariance).
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise RuntimeError(
            "hmmlearn not installed. Add hmmlearn to deps and reinstall."
        ) from e

    cols = [c for c in feature_cols if c in features.columns]
    if not cols:
        raise ValueError(f"none of {feature_cols} present in features")
    X = features[cols].dropna().to_numpy(dtype=float)
    if len(X) < 100:
        raise ValueError(f"too few samples to fit HMM: {len(X)}")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=seed,
        tol=1e-3,
    )
    try:
        model.fit(X)
    except Exception as e:  # noqa: BLE001
        # HMM convergence can fail on degenerate data (constant features,
        # singular covariance). Don't crash the whole pipeline — return a
        # fallback that maps everything to "chop" so downstream rubric
        # weights stay neutral.
        log.warning(f"HMM fit failed: {e}; returning fallback chop-only model")
        try:
            from omni.observability.metrics import metrics
            metrics.inc("regime_label_flips_total")  # treat as a "noise event"
        except Exception:  # noqa: BLE001
            pass
        return _fallback_model(cols)

    # Phase 24.16 — `fit()` may "succeed" with non-PSD covariances on
    # degenerate inputs; the failure then surfaces at predict() time as
    # `'covars' must be symmetric, positive-definite`. Probe predict on a
    # tiny sample so we catch this BEFORE returning the model and route
    # to the fallback instead of crashing the orchestrator stage.
    try:
        _ = model.predict(X[:5])
    except Exception as e:  # noqa: BLE001
        log.warning(f"HMM predict probe failed ({e}); using fallback chop-only model")
        try:
            from omni.observability.metrics import metrics
            metrics.inc("regime_label_flips_total")
        except Exception:  # noqa: BLE001
            pass
        return _fallback_model(cols)

    # Convergence check — hmmlearn exposes monitor_.converged_
    converged = bool(getattr(getattr(model, "monitor_", None),
                                "converged_", True))
    if not converged:
        log.warning(f"HMM did not converge after {n_iter} iters — "
                     f"label trust is reduced; consider raising n_iter or "
                     f"shortening the train window")

    state_to_label = _label_states(model.means_, cols)
    log.info(f"HMM fit: states={n_states} converged={converged} "
              f"labels={state_to_label}")
    return RegimeModel(hmm=model, feature_cols=cols,
                        state_to_label=state_to_label,
                        converged=converged, fallback=False)


def _fallback_model(cols: list[str]) -> RegimeModel:
    """Return a stub RegimeModel that always predicts 'chop'. Used when
    the real HMM fails to fit so downstream code keeps working with a
    neutral signal."""

    class _StubHMM:
        n_components = 1

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.ones((len(X), 1))

    return RegimeModel(
        hmm=_StubHMM(),
        feature_cols=cols,
        state_to_label={0: "chop"},
        converged=False,
        fallback=True,
    )


def build_regime_features(macro: pd.DataFrame, spx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble regime input features from macro + (optional) SPX prices.

    macro: output of features.macro.compute() — must have date, vix_level (and ideally us10y).
    spx: OHLCV frame for SPX (^GSPC). If None, callers should pass a market proxy.
    """
    if macro is None or macro.empty:
        return pd.DataFrame(columns=["date", "ret", "rv", "vix"])
    out = pd.DataFrame({"date": pd.to_datetime(macro["date"])})
    if "vix_level" in macro.columns:
        out["vix"] = macro["vix_level"].values
    if spx is not None and not spx.empty:
        s = spx.sort_values("date").copy()
        s["date"] = pd.to_datetime(s["date"])
        ac = s["adj_close"].astype(float)
        s["ret"] = np.log(ac / ac.shift(1))
        s["rv"] = s["ret"].rolling(20, min_periods=20).std() * np.sqrt(252)
        out = out.merge(s[["date", "ret", "rv"]], on="date", how="left")
    return out.dropna().reset_index(drop=True)
