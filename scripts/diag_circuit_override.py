"""Synthetic end-to-end test for the circuit-risk override.

Builds two synthetic OHLC frames that mirror the bug pattern the user
reported:

  CIANAGRO-like  : 60 calm days, then 5 lower-circuit days in a row,
                   ending with the latest bar locked at -10%. Pre-fix,
                   the trained XGBoost would underestimate this because
                   ATR/vol features collapse on locked bars. Post-fix,
                   recent_circuit_history forces band='high'.

  FORCE-like     : 60 normal trading days with high vol but ZERO locked
                   bars. The trained model fires "elevated" on the high
                   vol pattern. Post-fix, the dampener pulls it back to
                   the model's raw probability * 0.5 because hits_60d=0.

The script does NOT call the trained model. It directly tests the
helper + the override math, which is the part that just changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from omni.features.circuit import (
    add_circuit_features,
    detect_circuit_bars,
    recent_circuit_history,
)
from omni.features import technical as _tech


def _make_calm_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Calm prices: small daily moves, never circuit. Like FORCE MOTORS."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    rets = rng.normal(loc=0.0008, scale=0.018, size=n)
    # Inject a recent volatile drawdown so the trained model would over-fire.
    rets[-5:] = [-0.03, -0.022, -0.018, -0.025, -0.04]
    close = 1000 * (1 + pd.Series(rets, index=dates)).cumprod()
    high = close * (1 + np.abs(rng.normal(0, 0.012, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.012, size=n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({
        "date": dates, "open": open_.values, "high": high.values,
        "low": low.values, "close": close.values, "adj_close": close.values,
        "volume": rng.integers(50_000, 500_000, size=n),
    })


def _make_lc_streak_frame(n: int = 65, lc_streak: int = 5,
                           seed: int = 1) -> pd.DataFrame:
    """Like CIANAGRO: calm-ish, then `lc_streak` locked -10% bars at the end."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    rets = list(rng.normal(loc=0.0, scale=0.015, size=n - lc_streak))
    rets += [-0.10] * lc_streak  # forced lower-circuit days
    close = 100 * (1 + pd.Series(rets, index=dates)).cumprod()
    # On locked bars, high == low == close (range == 0)
    high = close.copy()
    low = close.copy()
    open_ = close.shift(1).fillna(close.iloc[0])
    # Calm bars get realistic intraday range
    is_lc = np.array([False] * (n - lc_streak) + [True] * lc_streak)
    high.loc[~is_lc] = close.loc[~is_lc] * (1 + np.abs(rng.normal(0, 0.01,
                                              size=(~is_lc).sum())))
    low.loc[~is_lc] = close.loc[~is_lc] * (1 - np.abs(rng.normal(0, 0.01,
                                              size=(~is_lc).sum())))
    # Locked bars: open also at the locked price (no intraday discovery)
    open_.loc[is_lc] = close.loc[is_lc]
    return pd.DataFrame({
        "date": dates, "open": open_.values, "high": high.values,
        "low": low.values, "close": close.values, "adj_close": close.values,
        "volume": rng.integers(50_000, 500_000, size=n),
    })


def _emulate_override(model_p: float, hist: dict) -> tuple[float, str | None]:
    """Mirror the override branch in predict.py for unit testing."""
    base_p = float(model_p)
    adj_p = base_p
    reason = None
    if (hist.get("currently_locked")
            and hist.get("last_lock_direction") == "lower"):
        adj_p = max(adj_p, 0.55)
        reason = "currently locked at lower circuit"
    elif (hist.get("hits_5d", 0) >= 2
            and hist.get("last_lock_direction") == "lower"):
        adj_p = max(adj_p, 0.35)
        reason = f"{hist['hits_5d']} lower-circuit days in last 5"
    elif (hist.get("hits_20d", 0) >= 3
            and hist.get("last_lock_direction") == "lower"
            and hist.get("days_since_last_lock", 10**6) < 5):
        adj_p = max(adj_p, 0.22)
        reason = f"{hist['hits_20d']} lower-circuits in last 20d"
    if (reason is None
            and base_p >= 0.20
            and hist.get("hits_60d", 0) == 0
            and hist.get("n_bars", 0) >= 60):
        adj_p = max(0.05, base_p * 0.50)
        reason = "no circuit hits in last 60d (model false-positive guard)"
    return adj_p, reason


def _band(p: float) -> str:
    if p < 0.10:
        return "low"
    if p < 0.25:
        return "elevated"
    return "high"


def _print_case(name: str, prices: pd.DataFrame, simulated_model_p: float):
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    is_circ = detect_circuit_bars(prices, asset_class="india")
    print(f"  bars={len(prices)}  circuits in frame={int(is_circ.sum())}")
    hist = recent_circuit_history(prices, asset_class="india")
    print(f"  history dict (latest-bar):")
    for k, v in hist.items():
        print(f"    {k}: {v}")
    adj_p, reason = _emulate_override(simulated_model_p, hist)
    print(f"  simulated model p   = {simulated_model_p:.3f}  band={_band(simulated_model_p)}")
    print(f"  post-override p     = {adj_p:.3f}  band={_band(adj_p)}")
    print(f"  override reason     = {reason}")

    # Phase 24.22 verification: per-row circuit features must exist on the
    # latest row with the same numbers as the latest-bar dict.
    cf = add_circuit_features(
        prices[["date", "open", "high", "low", "close", "adj_close"]].copy(),
        asset_class="india",
    )
    last = cf.iloc[-1]
    print(f"  per-row features (latest row, must match dict):")
    for col in ("is_circuit_bar", "circuit_hits_5d", "circuit_hits_20d",
                "circuit_hits_60d", "circuit_lower_hits_20d",
                "circuit_upper_hits_20d", "consecutive_lock_days",
                "days_since_last_lock", "last_lock_direction_signed"):
        print(f"    {col}: {last[col]}")

    # Cross-check: per-row hits/streak/days_since must agree with the
    # latest-bar dict.
    expected = {
        "circuit_hits_5d": hist["hits_5d"],
        "circuit_hits_20d": hist["hits_20d"],
        "circuit_hits_60d": hist["hits_60d"],
        "consecutive_lock_days": hist["consecutive_lock_days"],
        "days_since_last_lock": min(hist["days_since_last_lock"], 252),
    }
    mismatches = []
    for k, v in expected.items():
        if int(last[k]) != int(v):
            mismatches.append(f"{k}: per-row={last[k]} vs dict={v}")
    if mismatches:
        print(f"  MISMATCH: {mismatches}")
    else:
        print(f"  per-row vs dict cross-check: OK")

    # Vol estimators: ATR_pct_14 vs ATR_pct_14_unlocked vs Parkinson vs GK.
    # On the LC-streak case the unlocked ATR should be > the standard ATR
    # (because the locked zeros aren't dragging it down).
    tech = _tech.compute(prices)
    if len(tech) >= 20:
        last_tech = tech.iloc[-1]
        print(f"  vol estimators (latest row):")
        for col in ("atr_pct_14", "atr_pct_14_unlocked",
                    "parkinson_vol_20", "gk_vol_20", "vol_20"):
            v = last_tech.get(col)
            if pd.notna(v):
                print(f"    {col}: {float(v):.5f}")
            else:
                print(f"    {col}: NaN")


def main() -> int:
    # CIANAGRO-like: 5-day lower-circuit streak. Pre-fix the model would
    # see clipped returns + zero range and rate this LOW.
    cia = _make_lc_streak_frame()
    _print_case("CIANAGRO-like (5-day LC streak, currently locked)",
                cia, simulated_model_p=0.06)

    # FORCE-like: high-vol drawdown but zero locked bars.
    force = _make_calm_frame()
    _print_case("FORCE-like (volatile sell-off, zero historical circuits)",
                force, simulated_model_p=0.32)

    # Sanity case: calm name with model already saying "low" — should NOT
    # trigger any override.
    calm = _make_calm_frame(seed=42)
    calm["close"] = 1000 + np.linspace(0, 50, len(calm))
    calm["adj_close"] = calm["close"]
    calm["high"] = calm["close"] * 1.005
    calm["low"] = calm["close"] * 0.995
    _print_case("Calm name (model_p=0.04, no override expected)",
                calm, simulated_model_p=0.04)

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
