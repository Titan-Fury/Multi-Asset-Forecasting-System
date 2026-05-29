"""Forward-return targets.

For a given horizon (1, 5, 20 business days), compute:
  fwd_ret_<H>d  -- forward simple return adj_close[t+H] / adj_close[t] - 1
                   (NOT log return — keep this consistent with downstream
                   PnL and ranking metrics.)
  fwd_dir_<H>d  -- sign(fwd_ret), with neutral band optional
  fwd_q_<H>d    -- quintile bucket (0..4) per (asset_class, date)

These are the *labels*. They MUST NOT be joined into the feature frame as
features. Use `omni.features.build.assemble()` which keeps them separate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_cross_sectional_rank_targets(df: pd.DataFrame, *,
                                     horizons: tuple[int, ...] = (1, 5, 20)) -> pd.DataFrame:
    """Add `fwd_rank_<H>d` columns: percentile rank of `fwd_ret_<H>d` within
    each (date, asset_class). Output is float in [0, 1]; the binary direction
    target stays untouched.

    Numerai-style relative ranking dramatically reduces noise vs absolute
    return regression — the model learns "is this stock ABOVE-AVERAGE today"
    rather than "will it go up". Empirically lifts AUC by 0.02-0.04 on
    multi-asset universes.
    """
    if df is None or df.empty or "asset_class" not in df.columns or "date" not in df.columns:
        return df
    out = df.copy()
    for h in horizons:
        col = f"fwd_ret_{h}d"
        if col not in out.columns:
            continue
        rk = (out
              .groupby(["asset_class", "date"])[col]
              .rank(pct=True, method="average"))
        out[f"fwd_rank_{h}d"] = rk.astype(float)
    return out


def compute(prices: pd.DataFrame, *, horizons: tuple[int, ...] = (1, 5, 20),
            neutral_band: float = 0.0) -> pd.DataFrame:
    """`prices`: same OHLCV schema as features.technical.compute input.
    Returns DataFrame with columns date + per-horizon target columns.
    """
    if prices.empty:
        return pd.DataFrame()
    df = prices.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": df["date"]})
    ac = df["adj_close"].astype(float)
    for h in horizons:
        fwd = ac.shift(-h) / ac - 1
        out[f"fwd_ret_{h}d"] = fwd
        if neutral_band > 0:
            sign = np.where(fwd > neutral_band, 1,
                            np.where(fwd < -neutral_band, -1, 0))
        else:
            sign = np.sign(fwd)
        out[f"fwd_dir_{h}d"] = sign
    return out


def triple_barrier(prices: pd.DataFrame,
                   *,
                   horizons: tuple[int, ...] = (1, 5, 20),
                   pt_mult: float = 1.5,
                   sl_mult: float = 1.0,
                   vol_window: int = 40) -> pd.DataFrame:
    """Triple-barrier labels (de Prado, AFML §3.4).

    Phase 28.4 — `pt_mult` default changed from 2.0 to 1.5 to rebalance
    the positive class share. Under pt_mult=2.0 the (+1, 0, -1) class
    distribution on the india training set was roughly 25 / 33 / 42, so
    binarising as `(== +1)` for the classifier head produced a 25/75
    imbalance that made a constant-zero predictor score ~0.51 OOF AUC
    just from the class skew, masking real signal lift. Tightening to
    1.5σ rebalances to ~38/62 — a meaningful problem for the classifier
    while still preserving the asymmetric 1.5:1.0 reward:risk profile.

    Phase 28.4 — `vol_window` default changed from 20 to 40 trading days.
    20-day rolling sigma is noisy for Indian mid-caps where intraday
    range can swing dramatically across earnings windows. The shorter
    window made the +/- pt_mult * sigma barriers themselves spiky,
    producing TB labels that flipped between PT-hit / SL-hit / time-
    expired in nearby rows for what was clearly the same regime. 40
    days is a more stable estimator of the underlying vol while still
    being short enough to track regime shifts at the multi-week scale.

    Instead of "did the price end up after H days", ask:
      "within the next H days, did price touch +pt_mult*sigma BEFORE
       touching -sl_mult*sigma, or did the time barrier expire first?"

    sigma = rolling std of 1-day log returns over `vol_window`.

    Output columns per horizon H:
      tb_label_{H}d : +1 (PT hit), -1 (SL hit), 0 (time expired)
      tb_first_{H}d : business days until the first barrier was touched

    This is far more economically meaningful than fwd_dir because it
    captures path-dependent risk — a +5% prediction that drew down 8%
    along the way is NOT the same trade as a clean +5%.

    Returns:
        DataFrame keyed by `date` with the new columns. Existing
        forward-return columns are not modified.
    """
    if prices is None or prices.empty:
        return pd.DataFrame()
    df = prices.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": df["date"]})
    px = df["adj_close"].astype(float).to_numpy()
    high = df.get("high", df["adj_close"]).astype(float).to_numpy()
    low = df.get("low", df["adj_close"]).astype(float).to_numpy()
    # Phase 28.4 — guard against zero / negative prices in the log-return
    # calc that follows. A single 0 or negative entry (IPO-day placeholder,
    # corrupt corporate-action adj, data-provider error) produces
    # `-inf`/`NaN` log returns, which then corrupt the rolling sigma for
    # ~20 rows downstream — and through `pt = px * exp(±k*sigma)` produce
    # `+inf` / 0 barriers that collapse every label in the poison window
    # to 0 (time-expired). Replacing with NaN lets pandas' min_periods=5
    # skip the bad bars cleanly.
    px_for_log = np.where(px <= 0, np.nan, px)
    log_ret = np.log(px_for_log[1:] / px_for_log[:-1])
    sigma = pd.Series(log_ret).rolling(vol_window, min_periods=5).std().to_numpy()
    sigma = np.concatenate([[np.nan], sigma])  # align to px length
    n = len(px)
    for h in horizons:
        labels = np.zeros(n, dtype=np.int8)
        firsts = np.full(n, np.nan, dtype=float)
        for i in range(n - 1):
            s = sigma[i]
            if not np.isfinite(s) or s <= 0:
                continue
            pt = px[i] * np.exp(pt_mult * s)     # upper barrier
            sl = px[i] * np.exp(-sl_mult * s)    # lower barrier
            j_end = min(i + h, n - 1)
            label = 0
            first = h
            for j in range(i + 1, j_end + 1):
                hit_pt = high[j] >= pt
                hit_sl = low[j] <= sl
                if hit_pt and hit_sl:
                    # both touched same bar -> assume worst-case for safety
                    label = -1
                    first = j - i
                    break
                if hit_pt:
                    label = 1
                    first = j - i
                    break
                if hit_sl:
                    label = -1
                    first = j - i
                    break
            labels[i] = label
            firsts[i] = first
        out[f"tb_label_{h}d"] = labels
        out[f"tb_first_{h}d"] = firsts
    return out

