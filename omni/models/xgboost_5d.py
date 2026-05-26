"""Walk-forward XGBoost classifier for the 5d direction target.

Design:
  - Target column: `fwd_dir_5d` (mapped from {-1, 0, +1} to {0, 1} where +1 -> 1
    and {-1, 0} -> 0  — i.e. "is the 5d forward return positive").
  - Features: ALL columns except those flagged by `omni.features.build.is_target_col`,
    plus we drop identity / date columns and any all-NaN columns.
  - Walk-forward CV: time-ordered folds of (train_end, test_window). NEVER a
    random split.
  - Calibration: each fold trains XGBoost on the inner-train slice, then fits
    `CalibratedClassifierCV(method='isotonic', cv='prefit')` on a held-out
    inner-validation slice (last 20% of train window). The prefit calibrator
    is what produces the per-row probability used downstream.
  - Output: a dict[fold_id -> {model, calibrator, test_predictions_df}].

Public API:
  prepare_xy(df) -> (X, y, feature_cols)
  walkforward_fit(df, n_folds=..., test_size=...) -> WalkForwardResult
  predict_proba(model, calibrator, X) -> np.ndarray
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from omni.features.build import is_target_col
from omni.utils.logger import log

# Columns that are identity/metadata, never features
_DROP_ALWAYS = {
    "date", "ticker", "asset_class", "run_date", "source",
    # Phase 24.22 — raw OHLCV columns must NOT become features. They are
    # absolute price levels in each stock's own currency unit, so a tree
    # learns to split on price-level proxies which leak the ticker's
    # identity and destroy any chance of cross-ticker generalization.
    # The technical/diff_lagged feature builders already produce the
    # ticker-invariant transforms (returns, ratios, z-scores) we want the
    # model to consume.
    "open", "high", "low", "close", "adj_close", "volume",
}


def prepare_xy(df: pd.DataFrame, *, target: str = "fwd_dir_5d"
               ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Strip target columns and metadata, return (X, y, feature_cols).

    y is binary: 1 if fwd_dir_5d > 0 else 0. Rows where the target is NaN are
    dropped.
    """
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in frame")
    work = df.dropna(subset=[target]).copy()
    y = (work[target] > 0).astype(int)

    feat_cols = [c for c in work.columns
                 if c not in _DROP_ALWAYS
                 and not is_target_col(c)]
    X = work[feat_cols].apply(pd.to_numeric, errors="coerce")
    # Drop columns that are entirely NaN within this slice
    X = X.dropna(axis=1, how="all")
    # Replace ±inf with NaN so XGBoost's missing=NaN handling kicks in
    # (new vol-adjusted / z-score features can produce inf on near-zero denoms).
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y, list(X.columns)


def walkforward_splits(dates: pd.Series, *, n_folds: int = 5,
                       min_train: int = 252,
                       embargo_days: int = 0) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) tuples in chronological order.

    Each fold's test window is the next slab; train is everything before it.

    Phase 28 — `embargo_days` skips a buffer between train_end and test_start
    so that fwd-H labels at the train boundary don't leak into the test fold.
    Default 0 keeps backward compatibility; `walkforward_fit` defaults to
    `horizon_days + 1` (typically 6 for 5d targets) which is the textbook
    purged-CV embargo.
    """
    order = np.argsort(dates.values)
    n = len(order)
    if n <= min_train + n_folds:
        raise ValueError(f"not enough rows ({n}) for {n_folds} folds")
    fold_size = (n - min_train) // n_folds
    embargo = max(0, int(embargo_days))
    for k in range(n_folds):
        train_end = min_train + k * fold_size
        test_start = train_end + embargo
        test_end = min(n, test_start + fold_size)
        if test_end <= test_start:
            break
        # Drop the last `embargo` rows from the train side too — those rows'
        # forward labels span into the test window and would leak across the
        # gap if we kept them.
        train_idx = order[: max(0, train_end - embargo)]
        test_idx = order[test_start:test_end]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    test_dates: pd.Series
    test_tickers: pd.Series  # NEW: ticker per test row, parallel to test_dates
    proba: np.ndarray
    y_true: np.ndarray
    model: object = None
    calibrator: object = None


@dataclass
class WalkForwardResult:
    folds: list[FoldResult] = field(default_factory=list)
    feature_cols: list[str] = field(default_factory=list)

    def predictions_frame(self) -> pd.DataFrame:
        rows = []
        for fr in self.folds:
            df = pd.DataFrame({
                "fold": fr.fold,
                "date": fr.test_dates.values,
                "ticker": fr.test_tickers.values,
                "proba_up": fr.proba,
                "y_true": fr.y_true,
            })
            rows.append(df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _build_monotone_constraints(feat_cols: Iterable[str]) -> tuple[int, ...]:
    """Return positional XGBoost monotone_constraints tuple aligned to
    `feat_cols` order. Only the features whose sign is unambiguous get
    constrained; everything else stays at 0 (unconstrained).

    Phase 24.22 — architectural insurance against the CIANAGRO
    failure mode. Even with limited training data the trees can never
    learn the wrong sign on these features.
    """
    rules: dict[str, int] = {
        # Lower-circuit pressure: more hits / direction toward downside =>
        # lower probability that the next 5 days are positive.
        "circuit_lower_hits_5d": -1,
        "circuit_lower_hits_20d": -1,
        # Upper-circuit demand: more hits => higher probability of continuation.
        "circuit_upper_hits_20d": +1,
        # Narrative distress signals.
        "panic_keyword_ratio": -1,
        "sebi_action_flag": -1,
    }
    return tuple(rules.get(c, 0) for c in feat_cols)


def _magnitude_weights(df: pd.DataFrame, *, target: str) -> np.ndarray | None:
    """Build per-row sample weights proportional to |realised forward return|.

    Phase 24.22 — focuses XGBoost on the days that mattered. A trading
    signal that flips sign correctly on a +0.1% bar costs nothing in the
    real world; flipping correctly on a -8% bar is the entire game.
    Without this weighting the binary objective treats both equally.

    Phase 28.3 — also recognises ``tb_label_<H>d`` targets so triple-
    barrier training paths still get magnitude weighting (previously
    they fell through to None and lost the focus signal).

    Falls back to None (uniform weighting) when ``fwd_ret_<H>d`` isn't
    in the frame, so existing callers that don't provide it still work.
    """
    h = "5"
    if target.endswith("d"):
        # Extract horizon from either fwd_dir_<H>d OR tb_label_<H>d.
        for prefix in ("fwd_dir_", "tb_label_"):
            if target.startswith(prefix):
                try:
                    h = target[len(prefix):-len("d")] or "5"
                except Exception:  # noqa: BLE001
                    h = "5"
                break
    ret_col = f"fwd_ret_{h}d"
    if ret_col not in df.columns:
        return None
    r = pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0).abs().to_numpy()
    # Clip extreme outliers (data errors) so they don't dominate the loss.
    # 25% in 5 trading days is already an extraordinary move; cap at 50%.
    r = np.clip(r, 0.0, 0.50)
    # Add a small floor so sub-percent moves still contribute to learning
    # the bulk of the distribution.
    w = 0.25 + 4.0 * r
    return w


def walkforward_fit(df: pd.DataFrame, *, target: str = "fwd_dir_5d",
                    n_folds: int = 5, min_train: int = 504,
                    inner_val_frac: float = 0.2,
                    xgb_params: dict | None = None,
                    keep_models: bool = False,
                    embargo_days: int | None = None,
                    sample_weight: np.ndarray | None = None) -> WalkForwardResult:
    """Walk-forward fit. Requires `df` to have a `date` column.

    Phase 28 — `embargo_days` defaults to `horizon_days + 1` (inferred from
    the target column name), preventing label leakage at the train/test
    boundary that previously inflated fold AUCs.

    Phase 28.3 — `sample_weight` is the caller-supplied per-row weight
    vector (e.g. recency weights from the trainer). When provided, it is
    combined MULTIPLICATIVELY with the magnitude-derived weights so a row
    gets BOTH "this day mattered" magnitude focus AND "this is recent
    enough to trust" recency emphasis. Length must equal len(df) BEFORE
    the prepare_xy NaN-drop; alignment to X.index happens internally.
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError as e:
        raise RuntimeError("xgboost / scikit-learn missing from env") from e

    # sklearn 1.6+ removed cv='prefit'; use FrozenEstimator if available.
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn>=1.6
        _HAS_FROZEN = True
    except ImportError:
        _HAS_FROZEN = False

    if "date" not in df.columns:
        raise KeyError("walkforward_fit requires a 'date' column")

    X, y, feat_cols = prepare_xy(df, target=target)
    dates = pd.to_datetime(df.loc[X.index, "date"])
    tickers = (df.loc[X.index, "ticker"]
               if "ticker" in df.columns
               else pd.Series(["__single__"] * len(X), index=X.index))

    # Phase 24.22 — magnitude-aware sample weights (focus on bars that move).
    full_weights = _magnitude_weights(df.loc[X.index].copy(), target=target)

    # Phase 28.3 — combine with caller-supplied recency weights when given.
    # Element-wise product: a row only gets "high attention" from XGBoost if
    # BOTH (a) its realised move was meaningful AND (b) it falls in a recent
    # enough regime. Aligned to X.index (post-NaN-drop) the same way
    # full_weights is.
    if sample_weight is not None:
        try:
            recency_aligned = np.asarray(sample_weight, dtype=float)
            # If caller passed pre-NaN-drop weights, slice to X.index.
            if len(recency_aligned) == len(df):
                # df.index might not be a positional range; use loc-by-label.
                recency_aligned = (pd.Series(recency_aligned, index=df.index)
                                     .loc[X.index].to_numpy())
            elif len(recency_aligned) != len(X):
                from omni.utils.logger import log as _log
                _log.warning(
                    f"walkforward_fit: sample_weight length {len(recency_aligned)} "
                    f"matches neither df ({len(df)}) nor X ({len(X)}); ignoring"
                )
                recency_aligned = None

            if recency_aligned is not None:
                if full_weights is None:
                    full_weights = recency_aligned
                else:
                    full_weights = full_weights * recency_aligned
        except Exception as exc:  # noqa: BLE001
            from omni.utils.logger import log as _log
            _log.warning(f"walkforward_fit: sample_weight merge failed: {exc}")

    # Phase 24.22 — hyperparameter overhaul:
    #  - learning_rate 0.05 -> 0.03 with proportional n_estimators bump
    #    (slower commitment, better generalisation on small training subsets)
    #  - min_child_weight 1 -> 5 (stop the tree splitting on noise rows;
    #    helps on sparse distressed-name data)
    #  - reg_alpha 0 -> 0.5 (L1; ignores noise features as catalogue grew)
    #  - monotone_constraints (architectural insurance against wrong-sign
    #    learning on the circuit-distress features)
    params = dict(
        n_estimators=800, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.5, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42,
        monotone_constraints=_build_monotone_constraints(feat_cols),
    )
    # Phase 22 — opt-in GPU: set OMNI_XGB_DEVICE=cuda to use the local GPU.
    # Defaults to CPU so existing CI / non-GPU envs keep working.
    import os as _os
    _dev = _os.environ.get("OMNI_XGB_DEVICE", "").strip().lower()
    if _dev in ("cuda", "gpu"):
        params["device"] = "cuda"
    if xgb_params:
        # Caller-provided overrides win; if they pass their own
        # monotone_constraints we honour theirs verbatim.
        params.update(xgb_params)

    # Phase 28 — derive horizon from the target column for the default
    # embargo (e.g. fwd_dir_5d -> 5d -> embargo 6). Caller can override
    # with embargo_days explicitly when needed (0 = legacy behaviour).
    if embargo_days is None:
        import re as _re
        m = _re.search(r"_(\d+)d$", str(target))
        horizon = int(m.group(1)) if m else 5
        embargo_days = horizon + 1

    res = WalkForwardResult(feature_cols=feat_cols)
    for k, (tr_idx, te_idx) in enumerate(walkforward_splits(
            dates, n_folds=n_folds, min_train=min_train,
            embargo_days=embargo_days)):
        # Inner split: last 20% of train window for calibration (time-ordered)
        n_inner = max(1, int(len(tr_idx) * inner_val_frac))
        inner_train = tr_idx[:-n_inner]
        inner_val = tr_idx[-n_inner:]

        Xt, yt = X.iloc[inner_train], y.iloc[inner_train]
        Xv, yv = X.iloc[inner_val], y.iloc[inner_val]
        Xs, ys = X.iloc[te_idx], y.iloc[te_idx]

        clf = XGBClassifier(**params)
        fit_kwargs: dict = {"verbose": False}
        if full_weights is not None:
            fit_kwargs["sample_weight"] = full_weights[inner_train]
        clf.fit(Xt.values, yt.values, **fit_kwargs)

        # Phase 25 — env-controlled calibration: OMNI_CAL_METHOD=beta opts
        # into Kull-Filho-Flach beta calibration. Default (isotonic) keeps
        # the existing CalibratedClassifierCV path so persisted models
        # behave identically.
        from omni.calibration.factory import get_method
        if get_method() == "beta":
            from omni.calibration.factory import fit_calibrator
            raw_v = clf.predict_proba(Xv.values)[:, 1]
            cal = fit_calibrator(raw_v, yv.values, method="beta")
            # Wrap so .predict_proba(X) keeps the original 2-D shape.
            # Build a small adapter that calls the underlying classifier
            # for raw scores, then runs the calibrator on them.

            class _BetaCalProxy:
                __slots__ = ("clf", "cal")

                def __init__(self, clf, cal):
                    self.clf = clf
                    self.cal = cal

                def predict_proba(self, X):
                    raw = self.clf.predict_proba(X)[:, 1]
                    p = self.cal.predict_proba(raw)
                    # Phase 24.22 — guarantee a 2-D (N, 2) output. The caller
                    # at line 213 indexes [:, 1] which raises IndexError if a
                    # beta-calibrator implementation returns the positive
                    # probability as a 1-D array. Normalise both shapes here.
                    p = np.asarray(p)
                    if p.ndim == 1:
                        return np.column_stack([1.0 - p, p])
                    return p

            cal = _BetaCalProxy(clf, cal)
        else:
            if _HAS_FROZEN:
                cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
            else:
                cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
            cal.fit(Xv.values, yv.values)

        proba = cal.predict_proba(Xs.values)[:, 1]
        log.info(f"fold={k} train={len(inner_train)} val={len(inner_val)} "
                 f"test={len(te_idx)} mean_proba={proba.mean():.3f}")

        res.folds.append(FoldResult(
            fold=k,
            train_size=len(inner_train),
            test_size=len(te_idx),
            test_dates=dates.iloc[te_idx],
            test_tickers=tickers.iloc[te_idx],
            proba=proba,
            y_true=ys.values,
            model=clf if keep_models else None,
            calibrator=cal if keep_models else None,
        ))
    return res
