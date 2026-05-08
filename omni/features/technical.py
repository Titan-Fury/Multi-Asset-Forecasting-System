"""Technical features from OHLCV.

All features are computed strictly from past data at each row -- no centered
rolling windows, no forward-looking transforms. Output index aligns with the
input `date` column.

Features produced:
  ret_1d, ret_5d, ret_20d
  log_ret_1d
  sma_10, sma_20, sma_50, sma_200
  ema_12, ema_26
  rsi_14
  macd, macd_signal, macd_hist
  bb_upper, bb_lower, bb_pctb (Bollinger %b)
  atr_14
  adx_14
  obv (on-balance volume)
  vol_5, vol_20 (realized vol of log returns)
  vol_ratio (vol_5 / vol_20)
  high_52w, low_52w, dist_from_52w_high (in pct)
  trend_strength (close / sma_200 - 1)
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

    # SMAs / EMAs
    for n in (10, 20, 50, 200):
        out[f"sma_{n}"] = c.rolling(n, min_periods=n).mean()
    out["ema_12"] = c.ewm(span=12, adjust=False, min_periods=12).mean()
    out["ema_26"] = c.ewm(span=26, adjust=False, min_periods=26).mean()

    # MACD
    macd = out["ema_12"] - out["ema_26"]
    macd_sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd - macd_sig

    # RSI 14
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = _safe_div(avg_up, avg_down)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # Bollinger 20, 2sd
    sma20 = c.rolling(20, min_periods=20).mean()
    sd20 = c.rolling(20, min_periods=20).std()
    out["bb_upper"] = sma20 + 2 * sd20
    out["bb_lower"] = sma20 - 2 * sd20
    out["bb_pctb"] = _safe_div(c - out["bb_lower"], out["bb_upper"] - out["bb_lower"])

    # ATR 14 (Wilder)
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # ADX 14
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    atr = out["atr_14"]
    plus_di = 100 * _safe_div(plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean(), atr)
    minus_di = 100 * _safe_div(minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean(), atr)
    dx = 100 * _safe_div((plus_di - minus_di).abs(), (plus_di + minus_di))
    out["adx_14"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # OBV
    sign = np.sign(c.diff().fillna(0.0))
    out["obv"] = (sign * v).fillna(0.0).cumsum()

    # Volatility
    out["vol_5"] = out["log_ret_1d"].rolling(5, min_periods=5).std() * np.sqrt(252)
    out["vol_20"] = out["log_ret_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)
    out["vol_ratio"] = _safe_div(out["vol_5"], out["vol_20"])

    # 52w extremes (252 trading days)
    out["high_52w"] = c.rolling(252, min_periods=60).max()
    out["low_52w"] = c.rolling(252, min_periods=60).min()
    out["dist_from_52w_high"] = _safe_div(c - out["high_52w"], out["high_52w"])

    # Trend strength
    out["trend_strength"] = _safe_div(c, out["sma_200"]) - 1

    return out
