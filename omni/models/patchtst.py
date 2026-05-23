"""PatchTST base learner — patch-based Transformer for time-series forecasting.

Phase 25.1 — adds a 4th tier-2 sequence model alongside XGBoost / LightGBM /
HGB / CatBoost. NHITS was disabled at OOF AUC 0.4974 (anti-predictive);
PatchTST is structurally different (attention vs interpolation hierarchy)
and may capture different signal.

Mirrors the contract of ``omni.models.nhits.NHiTSBinaryClassifier`` so the
stack can plug it in alongside the gradient-boosted bases without changes
to ``omni.models.stack.fit_stack``.

Design notes
------------
- We deliberately ignore tabular X for the PatchTST leg (same as NHITS).
  PatchTST consumes the price panel only. The meta LR decides how much to
  weight it relative to the tree bases.
- Probabilities come from sigmoid(k * forecast_return). k is auto-fit on
  training so the proba distribution roughly matches base-rate of y.
- Default-disabled — opt in via ``OMNI_ENABLE_PATCHTST=1``. Mirrors the
  NHITS pattern: ship the wrapper, default off, only enable after a
  diagnostic shows OOF AUC >= 0.51.
- GPU is used when ``OMNI_PATCHTST_DEVICE=cuda`` is set.
- Graceful degradation: if neuralforecast or pytorch-lightning are
  missing, ``fit`` raises and ``predict_proba`` returns the calibrated
  global mean — the stack drops it from base_specs in that case.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from omni.utils.logger import log


def _try_import_patchtst() -> bool:
    try:
        from neuralforecast import NeuralForecast  # noqa: F401
        from neuralforecast.models import PatchTST  # noqa: F401
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug(f"neuralforecast.PatchTST import failed: {exc}")
        return False


_PATCHTST_AVAILABLE = _try_import_patchtst()


class PatchTSTBinaryClassifier:
    """sklearn-style classifier wrapping a NeuralForecast(PatchTST) forecaster.

    Trains one global PatchTST model on a (unique_id=ticker, ds=date,
    y=close) panel, then converts H-day forecast returns into pseudo-
    probabilities via a learnt sigmoid scale.
    """

    def __init__(
        self,
        *,
        horizon_days: int = 5,
        input_size: int = 96,
        max_steps: int = 400,
        accelerator: str | None = None,
        patch_len: int = 16,
        stride: int = 8,
        n_heads: int = 4,
    ) -> None:
        self.horizon_days = int(horizon_days)
        self.input_size = int(input_size)
        self.max_steps = int(max_steps)
        env_dev = os.environ.get("OMNI_PATCHTST_DEVICE", "").strip().lower()
        if env_dev in ("cuda", "gpu"):
            self.accelerator = "gpu"
        else:
            self.accelerator = accelerator or "cpu"
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.n_heads = int(n_heads)
        self._nf: Any = None
        self._sigmoid_k: float = 50.0
        self._global_mean_proba: float = 0.5
        self._row_proba: dict[tuple[str, pd.Timestamp], float] = {}

    @staticmethod
    def _build_panel(dates, tickers, closes) -> pd.DataFrame:
        df = pd.DataFrame({
            "unique_id": pd.Series(tickers).astype(str).values,
            "ds": pd.to_datetime(pd.Series(dates).values),
            "y": pd.to_numeric(pd.Series(closes), errors="coerce").astype(float).values,
        })
        df = df.dropna(subset=["unique_id", "ds", "y"])
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        df = df.drop_duplicates(subset=["unique_id", "ds"], keep="last")
        return df

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        *,
        dates=None,
        tickers=None,
        closes=None,
        **_: Any,
    ) -> "PatchTSTBinaryClassifier":
        if not _PATCHTST_AVAILABLE:
            raise RuntimeError("neuralforecast.PatchTST not installed")
        if dates is None or tickers is None or closes is None:
            raise ValueError(
                "PatchTSTBinaryClassifier.fit needs dates, tickers, closes "
                "(stack.fit_stack must thread them through)."
            )
        from neuralforecast import NeuralForecast
        from neuralforecast.models import PatchTST

        panel = self._build_panel(dates, tickers, closes)
        if panel.empty or panel["unique_id"].nunique() < 2:
            log.warning("PatchTST: panel too small; fit skipped")
            self._global_mean_proba = float(np.clip(np.mean(y), 0.05, 0.95))
            return self

        # Keep only series with enough history. PatchTST needs more
        # context than NHITS because of the patch-tokenization.
        min_len = self.input_size + self.horizon_days + 5
        keep = panel.groupby("unique_id").size()
        keep = keep[keep >= min_len].index
        panel = panel[panel["unique_id"].isin(keep)].copy()
        if panel.empty:
            log.warning("PatchTST: no series long enough; fit skipped")
            self._global_mean_proba = float(np.clip(np.mean(y), 0.05, 0.95))
            return self

        log.info(
            f"PatchTST fit: tickers={panel['unique_id'].nunique()} "
            f"rows={len(panel):,} h={self.horizon_days} "
            f"accel={self.accelerator} max_steps={self.max_steps} "
            f"patch_len={self.patch_len} stride={self.stride}"
        )

        try:
            model = PatchTST(
                h=self.horizon_days,
                input_size=self.input_size,
                max_steps=self.max_steps,
                patch_len=self.patch_len,
                stride=self.stride,
                n_heads=self.n_heads,
                accelerator=self.accelerator,
                devices=1,
                scaler_type="robust",
                random_seed=42,
            )
        except TypeError as exc:
            # Older neuralforecast may not accept some of these kwargs;
            # fall back to defaults rather than die.
            log.warning(f"PatchTST init kwargs rejected ({exc}); retrying with minimal kwargs")
            model = PatchTST(
                h=self.horizon_days,
                input_size=self.input_size,
                max_steps=self.max_steps,
                accelerator=self.accelerator,
            )

        self._nf = NeuralForecast(models=[model], freq="D")
        try:
            self._nf.fit(panel)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"PatchTST fit failed: {exc}; falling back to mean proba")
            self._nf = None
            self._global_mean_proba = float(np.clip(np.mean(y), 0.05, 0.95))
            return self

        try:
            fc = self._nf.predict()
        except Exception as exc:  # noqa: BLE001
            log.warning(f"PatchTST predict on training panel failed: {exc}")
            self._global_mean_proba = float(np.clip(np.mean(y), 0.05, 0.95))
            return self

        last_close = (
            panel.groupby("unique_id")["y"].last().rename("last_close").reset_index()
        )
        fc = fc.reset_index() if "unique_id" not in fc.columns else fc
        # NeuralForecast names the prediction column after the model class
        pred_col = next((c for c in fc.columns
                          if c not in ("unique_id", "ds")), None)
        if pred_col is None:
            log.warning("PatchTST predict frame missing prediction column")
            self._global_mean_proba = float(np.clip(np.mean(y), 0.05, 0.95))
            return self
        fc = fc.merge(last_close, on="unique_id", how="left")
        fc["fwd_return"] = (fc[pred_col] - fc["last_close"]) / fc["last_close"].replace(0, np.nan)

        target = float(np.clip(np.mean(y), 0.05, 0.95))
        rets = fc["fwd_return"].dropna().to_numpy()
        if rets.size:
            from scipy.special import expit
            ks = np.linspace(5.0, 200.0, 40)
            errs = [abs(expit(k * rets).mean() - target) for k in ks]
            self._sigmoid_k = float(ks[int(np.argmin(errs))])
        self._global_mean_proba = target

        from scipy.special import expit
        for _, r in fc.iterrows():
            anchor = pd.Timestamp(r.get("ds", pd.NaT))
            tkr = str(r["unique_id"])
            ret = float(r.get("fwd_return", 0.0))
            self._row_proba[(tkr, anchor)] = float(expit(self._sigmoid_k * ret))

        log.info(
            f"PatchTST calibrated: k={self._sigmoid_k:.1f} "
            f"target_proba={target:.3f} row_cache={len(self._row_proba):,}"
        )
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        p = np.full(n, self._global_mean_proba, dtype=float)
        return np.column_stack([1.0 - p, p])

    def predict_panel(self, *, dates, tickers, closes) -> np.ndarray:
        """Per-row probabilities given the same panel triple used at fit time."""
        if self._nf is None or not self._row_proba:
            n = len(dates)
            return np.full(n, self._global_mean_proba, dtype=float)
        out = np.empty(len(dates), dtype=float)
        for i, (d, t) in enumerate(zip(dates, tickers)):
            key = (str(t), pd.Timestamp(d))
            out[i] = self._row_proba.get(key, self._global_mean_proba)
        return out


__all__ = ["PatchTSTBinaryClassifier", "_PATCHTST_AVAILABLE"]
