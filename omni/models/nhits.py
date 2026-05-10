"""N-HiTS base learner wrapper for the stacking ensemble.

Phase 23 — adds a 4th tier-2 base alongside XGBoost / LightGBM / HGB / CatBoost.

N-HiTS (Neural Hierarchical Interpolation for Time Series) is a forecasting
model from `neuralforecast`. We expose a *binary classification* facade so
``omni.models.stack.fit_stack`` can plug it in alongside the gradient-boosted
bases without changing its protocol.

Contract (matches sklearn-style classifier):
    fit(X: np.ndarray, y: np.ndarray, *, dates, tickers, closes, horizon_days)
        - X: tabular features (ignored — N-HiTS uses the price panel only).
        - y: binary up/down labels (used for sigmoid calibration).
        - dates / tickers / closes: long-format panel info from the trainer.
    predict_proba(X) -> ndarray of shape (n, 2)
        Column 1 is the probability of an up move at the trained horizon.

Design notes:
    - We deliberately ignore the tabular features for the N-HiTS leg. Its job is
      to capture pure price-history signal that the trees miss; the meta LR
      decides how much to weight it. Trying to feed it 88-d tabular features
      tends to hurt — N-HiTS likes clean target panels.
    - Probabilities come from sigmoid(k * forecast_return). k is auto-fit on
      training so the proba distribution roughly matches base-rate of y.
    - GPU is used when ``OMNI_NHITS_DEVICE=cuda`` is set in env.
    - Graceful degradation: if neuralforecast or pytorch-lightning are missing,
      the class still constructs but ``fit`` raises and ``predict_proba``
      returns 0.5; the stack drops it from base_specs in that case.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from omni.utils.logger import log


def _try_import_nf():
    try:
        from neuralforecast import NeuralForecast  # noqa: F401
        from neuralforecast.models import NHITS  # noqa: F401
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning(f"neuralforecast import failed: {exc}")
        return False


_NF_AVAILABLE = _try_import_nf()


class NHiTSBinaryClassifier:
    """Sklearn-style classifier wrapping a NeuralForecast(NHITS) forecaster.

    The wrapper trains one global NHITS model on a (unique_id=ticker, ds=date,
    y=close) panel, then converts H-day forecast returns into pseudo-probabilities
    via a learnt sigmoid scale.
    """

    def __init__(
        self,
        *,
        horizon_days: int = 5,
        input_size: int = 64,
        max_steps: int = 400,
        accelerator: str | None = None,
        n_freq_downsample: tuple[int, int, int] = (24, 12, 1),
    ) -> None:
        self.horizon_days = int(horizon_days)
        self.input_size = int(input_size)
        self.max_steps = int(max_steps)
        # Resolve accelerator: env override > arg > auto cpu
        env_dev = os.environ.get("OMNI_NHITS_DEVICE", "").strip().lower()
        if env_dev in ("cuda", "gpu"):
            self.accelerator = "gpu"
        else:
            self.accelerator = accelerator or "cpu"
        self.n_freq_downsample = n_freq_downsample
        self._nf: Any = None
        self._sigmoid_k: float = 50.0
        self._global_mean_proba: float = 0.5
        self._row_proba: dict[tuple[str, pd.Timestamp], float] = {}

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _build_panel(dates, tickers, closes) -> pd.DataFrame:
        df = pd.DataFrame({
            "unique_id": pd.Series(tickers).astype(str).values,
            "ds": pd.to_datetime(pd.Series(dates).values),
            "y": pd.to_numeric(pd.Series(closes), errors="coerce").astype(float).values,
        })
        df = df.dropna(subset=["unique_id", "ds", "y"])
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        # neuralforecast requires a regular daily frequency; we drop dups and
        # let NHITS handle gaps via its own padding internally.
        df = df.drop_duplicates(subset=["unique_id", "ds"], keep="last")
        return df

    # ----------------------------------------------------------------- fit
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        *,
        dates=None,
        tickers=None,
        closes=None,
        **_: Any,
    ) -> "NHiTSBinaryClassifier":
        if not _NF_AVAILABLE:
            raise RuntimeError("neuralforecast not installed")
        if dates is None or tickers is None or closes is None:
            raise ValueError(
                "NHiTSBinaryClassifier.fit needs dates, tickers, closes "
                "(stack.fit_stack must thread them through)."
            )
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS

        panel = self._build_panel(dates, tickers, closes)
        if panel.empty or panel["unique_id"].nunique() < 2:
            log.warning("NHITS: panel too small; fit skipped")
            self._global_mean_proba = float(np.clip(np.mean(y), 0.05, 0.95))
            return self

        # Keep only series with enough history to be forecastable.
        min_len = self.input_size + self.horizon_days + 5
        keep = panel.groupby("unique_id").size()
        keep = keep[keep >= min_len].index
        panel = panel[panel["unique_id"].isin(keep)].copy()
        if panel.empty:
            log.warning("NHITS: no series long enough; fit skipped")
            self._global_mean_proba = float(np.clip(np.mean(y), 0.05, 0.95))
            return self

        log.info(
            f"NHITS fit: tickers={panel['unique_id'].nunique()} "
            f"rows={len(panel):,} h={self.horizon_days} "
            f"accel={self.accelerator} max_steps={self.max_steps}"
        )

        model = NHITS(
            h=self.horizon_days,
            input_size=self.input_size,
            max_steps=self.max_steps,
            n_freq_downsample=list(self.n_freq_downsample),
            accelerator=self.accelerator,
            devices=1,
            scaler_type="robust",
            random_seed=42,
        )
        # neuralforecast picks freq automatically when ds is datetime-like;
        # equities aren't strictly business-day in our lake, so use 'D' to be
        # tolerant of holiday gaps. NHITS handles them internally.
        self._nf = NeuralForecast(models=[model], freq="D")
        try:
            self._nf.fit(panel)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"NHITS fit failed: {exc}; falling back to mean proba")
            self._nf = None
            self._global_mean_proba = float(np.clip(np.mean(y), 0.05, 0.95))
            return self

        # Forecast horizon for every series; build a (ticker, last_ds) -> proba map.
        try:
            fc = self._nf.predict()
        except Exception as exc:  # noqa: BLE001
            log.warning(f"NHITS predict on training panel failed: {exc}")
            self._global_mean_proba = float(np.clip(np.mean(y), 0.05, 0.95))
            return self

        # Last close per series (anchor)
        last_close = (
            panel.groupby("unique_id")["y"].last().rename("last_close").reset_index()
        )
        fc = fc.reset_index() if "unique_id" not in fc.columns else fc
        fc = fc.merge(last_close, on="unique_id", how="left")
        fc["fwd_return"] = (fc["NHITS"] - fc["last_close"]) / fc["last_close"].replace(0, np.nan)

        # Calibrate sigmoid scale so proba distribution roughly matches y mean.
        target = float(np.clip(np.mean(y), 0.05, 0.95))
        rets = fc["fwd_return"].dropna().to_numpy()
        if rets.size:
            # Find k in [5, 200] s.t. sigmoid(k*rets).mean() ~= target.
            from scipy.special import expit
            ks = np.linspace(5.0, 200.0, 40)
            errs = [abs(expit(k * rets).mean() - target) for k in ks]
            self._sigmoid_k = float(ks[int(np.argmin(errs))])
        self._global_mean_proba = target

        # Cache (ticker, anchor_date) -> proba so per-row inference is O(1)
        from scipy.special import expit
        for _, r in fc.iterrows():
            anchor = pd.Timestamp(r.get("ds", pd.NaT))
            tkr = str(r["unique_id"])
            ret = float(r.get("fwd_return", 0.0))
            self._row_proba[(tkr, anchor)] = float(expit(self._sigmoid_k * ret))

        log.info(
            f"NHITS calibrated: k={self._sigmoid_k:.1f} "
            f"target_proba={target:.3f} row_cache={len(self._row_proba):,}"
        )
        return self

    # ------------------------------------------------------- predict_proba
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        # Wrapper used by stack.predict at inference time — caller doesn't
        # provide dates/tickers, so we return the global mean. The stack's
        # OOF training path uses _predict_panel below directly.
        p = np.full(n, self._global_mean_proba, dtype=float)
        return np.column_stack([1.0 - p, p])

    def predict_panel(
        self, *, dates, tickers, closes
    ) -> np.ndarray:
        """Per-row probabilities given the same panel triple used at fit time."""
        if self._nf is None or not self._row_proba:
            n = len(dates)
            return np.full(n, self._global_mean_proba, dtype=float)
        out = np.empty(len(dates), dtype=float)
        for i, (d, t) in enumerate(zip(dates, tickers)):
            key = (str(t), pd.Timestamp(d))
            out[i] = self._row_proba.get(key, self._global_mean_proba)
        return out


__all__ = ["NHiTSBinaryClassifier", "_NF_AVAILABLE"]
