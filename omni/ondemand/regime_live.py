"""Live regime detector for the on-demand path.

Fetches SPX (^GSPC) + macro_live, fits a 4-state Gaussian HMM, returns
the model + the label of the most recent day. Cached 24h.

The model itself is not pickled across processes (hmmlearn objects are
large); we re-fit on cache miss. Fit is fast (~3-6s for 10y daily).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from omni.ondemand.cache import TTL, cached
from omni.ondemand.macro_live import _fetch_yf_series, compute_live
from omni.regime import hmm as _hmm
from omni.utils.logger import log


@dataclass
class RegimeOutput:
    label_today: str               # bull / chop / bear / crisis
    proba_today: dict[str, float]  # per-state posterior on the latest day
    asof: pd.Timestamp
    n_train: int


def _build_regime_inputs(lookback_years: int = 10) -> pd.DataFrame:
    """Assemble (date, ret, rv, vix) from live sources."""
    macro = compute_live(lookback_years=lookback_years)
    spx = _fetch_yf_series("^GSPC", lookback_years=lookback_years)
    if spx.empty:
        return pd.DataFrame(columns=["date", "ret", "rv", "vix"])
    spx = spx.sort_values("date").copy()
    spx["date"] = pd.to_datetime(spx["date"])
    ac = spx["value"].astype(float)
    spx["ret"] = np.log(ac / ac.shift(1))
    spx["rv"] = spx["ret"].rolling(20, min_periods=20).std() * np.sqrt(252)

    out = pd.DataFrame({"date": pd.to_datetime(macro["date"])}) if not macro.empty \
          else pd.DataFrame({"date": spx["date"]})
    if not macro.empty and "vix_level" in macro.columns:
        out = out.merge(macro[["date", "vix_level"]].rename(columns={"vix_level": "vix"}),
                         on="date", how="left")
    out = out.merge(spx[["date", "ret", "rv"]], on="date", how="left")
    return out.dropna().reset_index(drop=True)


@cached(ttl=TTL.regime, key="regime_live:v1")
def fit_or_load() -> tuple[pd.DataFrame, RegimeOutput] | None:
    """Fit a regime model on live data and return today's label.

    Returns (regime_features_frame, RegimeOutput) on success, or None if
    not enough data (e.g. yfinance failed).
    """
    feats = _build_regime_inputs()
    if feats.empty or len(feats) < 200:
        log.warning(f"regime_live: insufficient data ({len(feats)} rows)")
        return None
    try:
        model = _hmm.fit(feats, n_states=4)
        proba = model.predict_proba(feats)
        labels = model.predict(feats)
    except Exception as e:
        log.warning(f"regime_live fit failed: {e}")
        return None

    last_idx = len(feats) - 1
    last_row = proba.iloc[last_idx].to_dict()
    out = RegimeOutput(
        label_today=str(labels[last_idx]),
        proba_today={k: float(v) for k, v in last_row.items()},
        asof=pd.Timestamp(feats.iloc[last_idx]["date"]),
        n_train=len(feats),
    )
    log.info(f"regime_live: today={out.label_today} n={out.n_train}")
    # Attach the per-row labels onto the feature frame so callers can inspect
    feats = feats.assign(regime=labels)
    return feats, out
