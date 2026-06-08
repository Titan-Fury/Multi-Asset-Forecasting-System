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
    """Relabel `fwd_ret_<H>d` and `fwd_dir_<H>d` to use the next UNLOCKED
    close as the endpoint instead of dropping circuit-locked rows.

    Phase 24.22 — major change vs. the previous behaviour.

    Old behaviour: NaN the labels on rows whose source bar was a circuit
    lock. Result: training set silently dropped exactly the rows that
    carry the chronic-LC continuation signal, leaving the model blind
    to distressed Indian small-caps (the CIANAGRO failure mode).

    New behaviour: keep every row, but substitute the *next unlocked
    close at or after t+H* as the label endpoint. So a row at t whose
    t+H bar is itself a locked circuit bar gets a label based on the
    first real price-discovery bar after that. The original t-side
    close is kept (even if locked) because clipping the entry by the
    circuit band is the actual fill an operator would experience.

    If no unlocked bar exists within `2 * H` sessions of t+H, the
    label is left as NaN (genuinely no realized signal yet).

    `out` must contain OHLC + `fwd_ret_<H>d` columns. Returns a copy
    with the targets recomputed; non-target columns untouched.
    `tb_label_<H>d` triple-barrier targets are also relabelled to
    NaN on rows where the realized window is fully locked (these
    were not built off close-to-close so the substitution doesn't
    naturally apply).
    """
    if out is None or out.empty:
        return out
    needed = {"open", "high", "low"}
    have_close = "adj_close" in out.columns or "close" in out.columns
    if not needed.issubset(out.columns) or not have_close:
        return out
    if asset_class.lower() != "india":
        return out

    is_circ = detect_circuit_bars(out, asset_class=asset_class)
    if not is_circ.any():
        return out

    out = out.copy()
    ac = out.get("adj_close", out.get("close")).astype(float).to_numpy()
    is_circ_arr = is_circ.to_numpy()
    n = len(out)

    # Pre-compute "next unlocked index >= k" once via a backward pass.
    # next_unlocked[k] = first j >= k with is_circ_arr[j] == False, or n.
    next_unlocked = np.full(n, n, dtype=np.int64)
    last_unlocked = n
    for j in range(n - 1, -1, -1):
        if not is_circ_arr[j]:
            last_unlocked = j
        next_unlocked[j] = last_unlocked

    for h in horizons:
        ret_col = f"fwd_ret_{h}d"
        dir_col = f"fwd_dir_{h}d"
        if ret_col not in out.columns:
            continue
        new_ret = np.full(n, np.nan, dtype=float)
        # Allow searching up to 2*h forward for an unlocked bar; beyond
        # that we accept the row has no clean realised return yet.
        max_offset = max(h * 2, h + 5)
        for i in range(n):
            target_idx = i + h
            if target_idx >= n:
                continue
            j = int(next_unlocked[target_idx])
            if j >= n:
                continue
            if (j - target_idx) > max_offset:
                continue
            start = ac[i]
            end = ac[j]
            if start <= 0 or not np.isfinite(start) or not np.isfinite(end):
                continue
            new_ret[i] = (end / start) - 1.0
        out[ret_col] = new_ret
        if dir_col in out.columns:
            out[dir_col] = np.sign(new_ret)
            # Keep NaN where new_ret is NaN
            mask_nan = np.isnan(new_ret)
            out.loc[mask_nan, dir_col] = np.nan

        # Triple-barrier: if the entire forward window is locked, drop
        # the label since intra-window barrier touches were artificial.
        tb_col = f"tb_label_{h}d"
        if tb_col in out.columns:
            if out[tb_col].dtype != float:
                out[tb_col] = out[tb_col].astype(float)
            for i in range(n):
                end = min(i + h, n - 1)
                if end > i and bool(is_circ.iloc[i:end + 1].all()):
                    out.iat[i, out.columns.get_loc(tb_col)] = np.nan
    return out


def recent_circuit_history(prices: pd.DataFrame,
                             *,
                             asset_class: str = "india",
                             windows: tuple[int, ...] = (5, 20, 60),
                             ) -> dict:
    """Compute circuit-hit history for the LATEST bar of `prices`.

    This is the missing signal that makes the trained `circuit_risk` model
    blind to chronic-LC stocks. Once a stock starts hitting lower circuits:
      - intraday range collapses (high == low) so ATR_pct_14 drops
      - daily returns are clipped at -10% so vol_5/vol_20 are bounded
      - the model's learned "high vol -> high circuit risk" inverts

    By extracting the locked-bar history directly from OHLC at inference
    time, we can apply a deterministic override on top of the model.

    Returns a dict (always populated, never raises) with:
      hits_<W>d         int counts for each window in `windows`
      currently_locked  bool — most recent bar locked at a circuit
      last_lock_direction  'upper'|'lower'|None
      consecutive_lock_days int — current streak of consecutive locked bars
      days_since_last_lock  int — 0 if currently locked, large int if never
      last_lock_band_pct float — magnitude of the last locked move (0.05/0.10/0.20)
    """
    out: dict = {f"hits_{w}d": 0 for w in windows}
    out.update({
        "currently_locked": False,
        "last_lock_direction": None,
        "consecutive_lock_days": 0,
        "days_since_last_lock": 10**6,
        "last_lock_band_pct": 0.0,
        "n_bars": int(0 if prices is None else len(prices)),
    })
    if prices is None or prices.empty:
        return out
    if asset_class.lower() != "india":
        return out

    needed = {"open", "high", "low"}
    have_close = "adj_close" in prices.columns or "close" in prices.columns
    if not needed.issubset(prices.columns) or not have_close:
        return out

    df = prices.sort_values("date").reset_index(drop=True) \
        if "date" in prices.columns else prices.reset_index(drop=True)

    close = df.get("adj_close", df.get("close")).astype(float)
    prev = close.shift(1)
    signed_ret = (close / prev - 1.0)
    is_circ = detect_circuit_bars(df, asset_class="india")

    # Hits per window (most recent N rows)
    for w in windows:
        out[f"hits_{w}d"] = int(is_circ.tail(w).sum())

    # Last bar status
    if len(is_circ) and bool(is_circ.iloc[-1]):
        out["currently_locked"] = True
        out["days_since_last_lock"] = 0
        last_ret = float(signed_ret.iloc[-1]) if pd.notna(signed_ret.iloc[-1]) else 0.0
        out["last_lock_direction"] = "upper" if last_ret > 0 else "lower"
        out["last_lock_band_pct"] = abs(round(last_ret, 4))
    else:
        # Walk backward for the most recent locked bar
        for i in range(len(is_circ) - 1, -1, -1):
            if bool(is_circ.iloc[i]):
                out["days_since_last_lock"] = int(len(is_circ) - 1 - i)
                last_ret = float(signed_ret.iloc[i]) if pd.notna(signed_ret.iloc[i]) else 0.0
                out["last_lock_direction"] = "upper" if last_ret > 0 else "lower"
                out["last_lock_band_pct"] = abs(round(last_ret, 4))
                break

    # Consecutive streak ending at the latest bar
    streak = 0
    for i in range(len(is_circ) - 1, -1, -1):
        if bool(is_circ.iloc[i]):
            streak += 1
        else:
            break
    out["consecutive_lock_days"] = streak

    return out


# ---------------------------------------------------------------- per-row ---

# Cap "days since last lock" at this many bars so the feature stays bounded
# (sentinel-like values blow up tree splits). Anything older counts as "no
# recent lock signal" and is encoded as the cap value.
_DAYS_SINCE_LOCK_CAP = 252


def add_circuit_features(prices_or_features: pd.DataFrame,
                            *,
                            asset_class: str = "india",
                            ) -> pd.DataFrame:
    """Add per-row circuit-history features to a frame with OHLC.

    This is the *training-time* counterpart to `recent_circuit_history()`:
    where that function returns one dict for the latest bar, this one
    produces a parallel column for EVERY bar so the model can learn the
    streak signal natively. Pair this with the trained `circuit_risk`
    model after a retrain.

    Columns produced (all numeric, NaN-safe, ticker-invariant):
      is_circuit_bar               bool -> int {0,1}
      circuit_hits_5d              count of locked bars in the trailing 5
      circuit_hits_20d             count of locked bars in the trailing 20
      circuit_hits_60d             count of locked bars in the trailing 60
      circuit_lower_hits_20d       lower-only locks in last 20
      circuit_upper_hits_20d       upper-only locks in last 20
      consecutive_lock_days        running streak ending at this bar
      days_since_last_lock         clamped to _DAYS_SINCE_LOCK_CAP
      last_lock_direction_signed   +1 (upper) / -1 (lower) / 0 (none in window)

    For non-India asset classes all columns become 0 / cap so the universal
    schema is preserved without injecting bias.
    """
    df = prices_or_features
    if df is None or df.empty:
        return df

    out = df.copy()

    # If the frame has no OHLC, we can't detect circuits — emit zero
    # columns so the downstream model never sees missing keys.
    needed = {"open", "high", "low"}
    have_close = "adj_close" in out.columns or "close" in out.columns
    if not needed.issubset(out.columns) or not have_close \
            or asset_class.lower() != "india":
        out["is_circuit_bar"] = 0
        out["circuit_hits_5d"] = 0
        out["circuit_hits_20d"] = 0
        out["circuit_hits_60d"] = 0
        out["circuit_lower_hits_20d"] = 0
        out["circuit_upper_hits_20d"] = 0
        out["consecutive_lock_days"] = 0
        out["days_since_last_lock"] = _DAYS_SINCE_LOCK_CAP
        out["last_lock_direction_signed"] = 0
        return out

    # Sort & reset for deterministic rolling math; we'll restore order at end.
    sort_order = out["date"].argsort() if "date" in out.columns else None
    if sort_order is not None:
        out = out.iloc[sort_order.values].reset_index(drop=True)

    is_circ = detect_circuit_bars(out, asset_class="india")
    is_circ_int = is_circ.astype(int)
    out["is_circuit_bar"] = is_circ_int.values

    # Rolling counts. min_periods=1 so early rows are populated
    # progressively rather than NaN.
    out["circuit_hits_5d"] = is_circ_int.rolling(5, min_periods=1).sum().astype(int).values
    out["circuit_hits_20d"] = is_circ_int.rolling(20, min_periods=1).sum().astype(int).values
    out["circuit_hits_60d"] = is_circ_int.rolling(60, min_periods=1).sum().astype(int).values

    # Direction-split: did the lock happen on the up side or the down side?
    # `signed_ret` is positive on upper-circuit days and negative on lower.
    close = out.get("adj_close", out.get("close")).astype(float)
    signed_ret = close.pct_change()
    upper_lock = (is_circ & (signed_ret > 0)).astype(int)
    lower_lock = (is_circ & (signed_ret < 0)).astype(int)
    out["circuit_upper_hits_20d"] = upper_lock.rolling(20, min_periods=1).sum().astype(int).values
    out["circuit_lower_hits_20d"] = lower_lock.rolling(20, min_periods=1).sum().astype(int).values

    # Consecutive streak ending at each row. Standard cumcount-on-False trick:
    # group_ids increments every time is_circ flips to False, so cumsum within
    # a group gives the running streak length for True runs.
    group_ids = (~is_circ).cumsum()
    streak = is_circ_int.groupby(group_ids).cumsum()
    out["consecutive_lock_days"] = streak.astype(int).values

    # Days since last lock. Position of last True ffilled forward, then
    # current position minus that.
    positions = pd.Series(np.arange(len(out), dtype=float), index=out.index)
    last_lock_pos = positions.where(is_circ).ffill()
    dsll = (positions - last_lock_pos).fillna(_DAYS_SINCE_LOCK_CAP)
    dsll = dsll.clip(upper=_DAYS_SINCE_LOCK_CAP).astype(int)
    out["days_since_last_lock"] = dsll.values

    # Direction of the most recent lock, ffilled and zeroed beyond the cap.
    sign_at_lock = np.sign(signed_ret).where(is_circ).ffill().fillna(0.0)
    # Zero-out direction when the last lock is older than the cap.
    sign_at_lock = sign_at_lock.where(dsll < _DAYS_SINCE_LOCK_CAP, other=0.0)
    out["last_lock_direction_signed"] = sign_at_lock.astype(float).values

    return out
