"""Circuit-bar detection for Indian equities.

NSE stocks have circuit limits (typically +/- 5%, +/- 10%, or +/- 20%
depending on category and futures-eligibility). When a stock locks at
circuit, the bar's high == low == close == price-at-circuit, and the
intraday volume is artificially capped because no offer/bid is
available on the side that hit the circuit.

For training labels this is a problem: a +5% upper-circuit day usually
means the *real* demand wanted +8% or +12%, but the price couldn't
move further. Including such bars in `fwd_dir_<H>d` labels biases the
model toward "circuit-day pattern -> bounded +5% return", which is
exactly the wrong inference. The signal in the underlying state was
strong-directional.

Detection heuristic:

    is_circuit_bar = (|close/prev_close - 1| >= circuit_thresh)
                      AND (range / open <= range_thresh)

Where range = high - low. A locked bar has near-zero intraday range.
We use a tolerance band (`circuit_thresh = 0.045`) so we catch the
+5% / -5% lock, the +10% / -10% lock, and the +20% / -20% lock with
the same call.

API:
    is_circuit = detect_circuit_bars(prices, asset_class="india")
    # is_circuit: pd.Series[bool] aligned to prices.index

For non-India asset classes the function returns all-False (US/EU don't
have daily circuits in the same way; tokens trade 24x7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_DEFAULT_BAND_TIERS = (0.045, 0.095, 0.195)  # 5%, 10%, 20% bands +/- a tolerance
_RANGE_TOLERANCE = 0.005                     # locked bar has range/open <= 0.5%


def detect_circuit_bars(prices: pd.DataFrame,
                          *, asset_class: str = "india",
                          band_tiers: tuple[float, ...] = _DEFAULT_BAND_TIERS,
                          range_tol: float = _RANGE_TOLERANCE) -> pd.Series:
    """Return a boolean Series flagging circuit-locked bars in `prices`.

    Required columns: open, high, low, close, adj_close (or just close).
    Returns all-False when the asset_class isn't 'india' OR when the
    required columns are missing.
    """
    if asset_class.lower() != "india":
        return pd.Series(False, index=prices.index)

    needed = {"open", "high", "low"}
    have_close = "adj_close" in prices.columns or "close" in prices.columns
    if not needed.issubset(prices.columns) or not have_close:
        return pd.Series(False, index=prices.index)

    close = prices.get("adj_close", prices.get("close")).astype(float)
    prev = close.shift(1)
    daily_ret = (close / prev - 1.0).abs()
    open_ = prices["open"].astype(float).replace(0, np.nan)
    rng = (prices["high"].astype(float) - prices["low"].astype(float)) / open_

    # Match any of the standard circuit bands
    band_match = pd.Series(False, index=prices.index)
    for tier in band_tiers:
        band_match = band_match | (daily_ret >= tier)

    locked = (rng <= range_tol)
    out = (band_match & locked).fillna(False)
    return out


def mask_circuit_labels(out: pd.DataFrame,
                          *,
                          horizons: tuple[int, ...] = (1, 5, 21),
                          asset_class: str = "india") -> pd.DataFrame:
    """Set `fwd_dir_<H>d` and `tb_label_<H>d` to NaN on rows whose source
    bar is a circuit-locked bar. The label distribution becomes cleaner
    because circuit bars are removed from training via the standard
    dropna(subset=[target]) call.

    `out` must contain the OHLC columns plus the target columns. Rows
    where the OHLC isn't available are left untouched (circuit detection
    just returns False).
    """
    is_circ = detect_circuit_bars(out, asset_class=asset_class)
    if not is_circ.any():
        return out
    out = out.copy()
    masked = 0
    for h in horizons:
        for prefix in ("fwd_dir_", "tb_label_"):
            col = f"{prefix}{h}d"
            if col in out.columns:
                # Cast to float so NaN is representable
                if out[col].dtype != float:
                    out[col] = out[col].astype(float)
                out.loc[is_circ, col] = np.nan
                masked += int(is_circ.sum())
    return out
