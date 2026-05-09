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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = X[self.feature_cols].to_numpy(dtype=float)
        states = self.hmm.predict(x)
        return np.array([self.state_to_label[s] for s in states])

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X[self.feature_cols].to_numpy(dtype=float)
        post = self.hmm.predict_proba(x)
        cols = [self.state_to_label[i] for i in range(post.shape[1])]
        # Collapse duplicate label columns (rare but possible) by max-proba
        df = pd.DataFrame(post, columns=cols)
        # pandas 2.2+: axis=1 in groupby is removed; transpose-groupby-transpose
        return df.T.groupby(level=0).max().T


def _label_states(means: np.ndarray, feature_cols: list[str]) -> dict[int, str]:
    """means: shape (n_states, n_features). Map state idx -> label."""
    n = means.shape[0]
    if n != 4:
        # Fallback: just sort by mean of feature[0] (return)
        order = np.argsort(means[:, 0])
        labels = ["bear", "chop", "chop", "bull"][:n]
        return {int(s): labels[i] for i, s in enumerate(order)}

    ret_idx = feature_cols.index("ret") if "ret" in feature_cols else 0
    vix_idx = feature_cols.index("vix") if "vix" in feature_cols else -1
    rv_idx = feature_cols.index("rv") if "rv" in feature_cols else -1

    # Crisis: highest VIX (or highest realized vol if VIX missing)
    severity = means[:, vix_idx] if vix_idx >= 0 else means[:, rv_idx]
    crisis_state = int(np.argmax(severity))

    others = [s for s in range(n) if s != crisis_state]
    rets = {s: means[s, ret_idx] for s in others}
    bull_state = max(rets, key=rets.get)
    bear_state = min(rets, key=rets.get)
    chop_state = [s for s in others if s not in (bull_state, bear_state)][0]
    return {crisis_state: "crisis", bull_state: "bull",
            bear_state: "bear", chop_state: "chop"}


def fit(features: pd.DataFrame, *, n_states: int = 4,
        feature_cols: tuple[str, ...] = ("ret", "rv", "vix"),
        seed: int = 42, n_iter: int = 200) -> RegimeModel:
    """Fit a Gaussian HMM. Caller must restrict `features` to the train window."""
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
    model.fit(X)
    state_to_label = _label_states(model.means_, cols)
    log.info(f"HMM fit: states={n_states} labels={state_to_label}")
    return RegimeModel(hmm=model, feature_cols=cols, state_to_label=state_to_label)


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
