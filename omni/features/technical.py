"""Technical features from OHLCV.

All features are computed strictly from past data at each row -- no centered
rolling windows, no forward-looking transforms. Output index aligns with the
input `date` column.

Phase 24.17 — every feature is now ticker-invariant (unitless ratio or
percentile) so a model trained across a multi-ticker / multi-currency pool
doesn't end up making tree splits that separate by currency. Raw price-
level outputs (sma_*, ema_*, bb_upper, bb_lower, atr_14, high_52w,
low_52w, obv) were dropped or replaced with their normalized counterparts.

Features produced:
  ret_1d, ret_5d, ret_20d
  log_ret_1d
  dist_sma_10, dist_sma_20, dist_sma_50, dist_sma_200   (close / sma_n - 1)
  rsi_14
  macd_pct, macd_signal_pct, macd_hist_pct              (MACD divided by close)
  bb_pctb                                                (already %-based)
  atr_pct_14                                             (atr_14 / close)
  atr_pct_14_unlocked                                    (ATR over unlocked bars only)
  parkinson_vol_20                                       (range-based, locked-bar-safe)
  gk_vol_20                                              (Garman-Klass, locked-bar-safe)
  adx_14                                                 (already 0-100)
  obv_z_60d                                              (rolling z-score of OBV)
  vol_5, vol_20, vol_ratio                               (annualized log-ret std)
  dist_from_52w_high                                     (close / max(252) - 1)
  dist_from_52w_low                                      (close / min(252) - 1)
  trend_strength                                         (close / sma_200 - 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def compute(prices: pd.DataFrame) -> pd.DataFrame:
    """`prices` must have columns: date, open, high, low, close, adj_close, volume.
    Returns a DataFrame indexed by date with technical features.
    Uses adj_close for return-style features, raw close for level indicators
    (matches industry convention for charts).
    """
    if prices.empty:
        return pd.DataFrame()
    df = prices.sort_values("date").reset_index(drop=True).copy()
    out = pd.DataFrame({"date": df["date"]})
    c = df["close"].astype(float)
    ac = df["adj_close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)

    # Returns (use adj_close to handle splits/dividends)
    out["ret_1d"] = ac.pct_change(1)
    out["ret_5d"] = ac.pct_change(5)
    out["ret_20d"] = ac.pct_change(20)
    out["log_ret_1d"] = np.log(ac / ac.shift(1))

    # SMAs as DISTANCE from close (unitless). EMAs dropped — MACD captures
    # the spread already, and keeping raw EMA values pollutes cross-ticker
    # training with absolute price levels.
    for n in (10, 20, 50, 200):
        sma_n = c.rolling(n, min_periods=n).mean()
        out[f"dist_sma_{n}"] = _safe_div(c, sma_n) - 1

    # MACD computed in price units, then normalized to % of close so it
    # becomes ticker-invariant.
    ema_12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_pct"] = _safe_div(macd, c)
    out["macd_signal_pct"] = _safe_div(macd_sig, c)
    out["macd_hist_pct"] = _safe_div(macd - macd_sig, c)

    # RSI 14 (already 0-100, ticker-invariant)
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = _safe_div(avg_up, avg_down)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # Bollinger 20, 2sd — keep only %b (position within band, 0-1).
    sma20 = c.rolling(20, min_periods=20).mean()
    sd20 = c.rolling(20, min_periods=20).std()
    bb_upper = sma20 + 2 * sd20
    bb_lower = sma20 - 2 * sd20
    out["bb_pctb"] = _safe_div(c - bb_lower, bb_upper - bb_lower)

    # ATR 14 (Wilder), normalized as % of close
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr_14 = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    out["atr_pct_14"] = _safe_div(atr_14, c)

    # Phase 24.22 — locked-bar-corrected vol estimators.
    # On NSE circuit days the bar high == low (range=0) and the daily return
    # is clipped at the circuit threshold. Wilder ATR averages those zeros
    # in, so a chronic-LC stock looks artificially calm. Two complementary
    # fixes go in here:
    #
    #   1. atr_pct_14_unlocked
    #        ATR-14 EWM that ignores locked bars entirely (NaN-skipping
    #        EWM). For a stock that hits LC every other day, this stays
    #        anchored on the unlocked sessions' true range.
    #
    #   2. parkinson_vol_20  /  gk_vol_20
    #        Range-based vol estimators (Parkinson 1980, Garman-Klass 1980).
    #        Both vanish to 0 on locked bars (ln(H/L)==0), but we substitute
    #        the squared close-to-close return (ln(C/C[-1])^2) which IS the
    #        only price-discovery signal available on a locked bar. This
    #        keeps the rolling variance honest.
    #
    # Range==0 is a strict superset of "circuit-locked" — also catches
    # data-quality flat days. That's the right behaviour: any zero-range
    # bar lacks intraday discovery and should defer to the c-c fallback.
    locked_zero_range = (h - l).abs() < 1e-12
    tr_unlocked = tr.where(~locked_zero_range)
    atr_14_unlocked = tr_unlocked.ewm(
        alpha=1 / 14, min_periods=14, adjust=False, ignore_na=True,
    ).mean()
    out["atr_pct_14_unlocked"] = _safe_div(atr_14_unlocked, c)

    # Parkinson + Garman-Klass with locked-bar fallback.
    # log(H/L) and log(C/O) are both zero when the bar is range-locked;
    # we substitute the squared close-to-close log return to keep the
    # variance signal alive on circuit days.
    h_safe = h.replace(0, np.nan)
    l_safe = l.replace(0, np.nan)
    o = df["open"].astype(float).replace(0, np.nan)
    ln_hl = np.log(h_safe / l_safe).fillna(0.0)
    ln_co = np.log(c / o).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ln_cc = np.log(ac / ac.shift(1))

    pk = (ln_hl ** 2) / (4 * np.log(2))
    gk = 0.5 * (ln_hl ** 2) - (2 * np.log(2) - 1) * (ln_co ** 2)

    fallback = ln_cc ** 2
    pk = pk.where(~locked_zero_range, fallback)
    gk = gk.where(~locked_zero_range, fallback)
    # Negative gk values are possible numerically when ln_co is large vs
    # ln_hl; clip at zero so sqrt is well-defined.
    gk = gk.clip(lower=0.0)

    out["parkinson_vol_20"] = np.sqrt(
        252.0 * pk.rolling(20, min_periods=20).mean()
    )
    out["gk_vol_20"] = np.sqrt(
        252.0 * gk.rolling(20, min_periods=20).mean()
    )

    # ADX 14 (already 0-100, ticker-invariant)
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    plus_di = 100 * _safe_div(plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean(), atr_14)
    minus_di = 100 * _safe_div(minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean(), atr_14)
    dx = 100 * _safe_div((plus_di - minus_di).abs(), (plus_di + minus_di))
    out["adx_14"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # OBV as 60d rolling z-score (unitless flow signal). Raw cumulative OBV
    # is a price/volume-weighted absolute level which doesn't generalize
    # across tickers.
    sign = np.sign(c.diff().fillna(0.0))
    obv = (sign * v).fillna(0.0).cumsum()
    obv_mean = obv.rolling(60, min_periods=60).mean()
    obv_std = obv.rolling(60, min_periods=60).std()
    out["obv_z_60d"] = _safe_div(obv - obv_mean, obv_std)

    # Volatility (annualized; already a unitless rate)
    out["vol_5"] = out["log_ret_1d"].rolling(5, min_periods=5).std() * np.sqrt(252)
    out["vol_20"] = out["log_ret_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)
    out["vol_ratio"] = _safe_div(out["vol_5"], out["vol_20"])

    # 52w extremes (252 trading days) — kept only as % distances.
    high_52w = c.rolling(252, min_periods=60).max()
    low_52w = c.rolling(252, min_periods=60).min()
    out["dist_from_52w_high"] = _safe_div(c - high_52w, high_52w)
    out["dist_from_52w_low"] = _safe_div(c - low_52w, low_52w)

    # Trend strength = close vs SMA-200 (same as dist_sma_200 but kept as
    # an alias for the rubric scorer which reads this name explicitly)
    out["trend_strength"] = out["dist_sma_200"]

    return out
