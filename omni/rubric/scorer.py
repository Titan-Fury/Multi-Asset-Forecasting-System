"""Rubric scorer.

Loads `config/rubric.yaml` and produces, for each (ticker, date) row:
  technical_score  (0-100)
  fundamental_score
  sentiment_score
  macro_score
  rubric_total     (regime-weighted convex combination)
  rubric_direction (+1 / 0 / -1 — bullish / neutral / bearish)
  rubric_breakdown (dict for explainability)

Pillar scoring (each pillar maps to [0, 100] where 50 == neutral):

  technical:
    +20 if ret_20d > 0
    +15 if close > sma_50 > sma_200 (golden trend), -15 if reverse
    +15 if rsi_14 in (40, 70), -10 if rsi_14 > 80 or < 30
    +10 if macd_hist > 0
    +10 if vol_ratio < 1 (declining vol)
    + signed component from trend_strength (clipped)

  fundamental (skipped for crypto/fx — returns 50 neutral):
    Combines z-scores (when present): earnings_yield_z, return_on_equity_z,
    profit_margin_z, revenue_growth_z, debt_to_equity_z (inverted).
    Mapped to 0-100 via logistic.

  sentiment:
    Logistic of (sent_mean) scaled by sqrt(min(n_articles, 20)/20).
    Adds momentum boost from sent_momentum_7d.

  macro:
    +ve if vix_level low, yield curve positive, dxy declining (for risk assets).
    Inverted for the 'crisis' regime.

Weights per regime come from rubric.yaml. Direction is +1 if total>=70 in
bull/chop, -1 if total<=30 in bear/crisis, else 0 (handled by gate later).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from omni.utils.env import rubric as _load_rubric
from omni.utils.logger import log

_NEUTRAL = 50.0


def _logistic(x: float | np.ndarray, *, k: float = 1.0) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * x))


def _clip01(x: float | np.ndarray) -> float | np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _score_technical(row: pd.Series) -> float:
    s = _NEUTRAL
    if pd.notna(row.get("ret_20d")) and row["ret_20d"] > 0:
        s += 10
    elif pd.notna(row.get("ret_20d")) and row["ret_20d"] < 0:
        s -= 10

    sma50 = row.get("sma_50"); sma200 = row.get("sma_200"); close = row.get("close")
    # 'close' may not be in features frame; use sma_10 vs sma_50 as fallback
    if pd.notna(sma50) and pd.notna(sma200):
        if sma50 > sma200:
            s += 10
        else:
            s -= 10

    rsi = row.get("rsi_14")
    if pd.notna(rsi):
        if 40 <= rsi <= 70:
            s += 10
        elif rsi > 80 or rsi < 30:
            s -= 10

    if pd.notna(row.get("macd_hist")) and row["macd_hist"] > 0:
        s += 5
    if pd.notna(row.get("vol_ratio")) and row["vol_ratio"] < 1.0:
        s += 5

    ts = row.get("trend_strength")
    if pd.notna(ts):
        s += float(np.clip(ts * 100, -10, 10))

    return float(np.clip(s, 0, 100))


def _score_fundamental(row: pd.Series, asset_class: str) -> float:
    if asset_class in ("crypto", "fx"):
        return _NEUTRAL
    parts = []
    # Preferred path: pre-computed z-scores
    for col, sign in [
        ("earnings_yield_z", +1),
        ("return_on_equity_z", +1),
        ("profit_margin_z", +1),
        ("revenue_growth_z", +1),
        ("debt_to_equity_z", -1),
    ]:
        v = row.get(col)
        if pd.notna(v):
            parts.append(sign * float(v))

    # Fallback path: raw fundamentals normalised to roughly z-units. Lets the
    # rubric still differentiate tickers when the feature builder has not
    # produced cross-sectional z-scores yet.
    if not parts:
        # Earnings yield from PE: 1/pe_ratio, centred on long-run S&P avg ~5%
        pe = row.get("pe_ratio") or row.get("trailing_pe")
        if pd.notna(pe) and pe > 0:
            ey = 1.0 / float(pe)
            parts.append(+1.0 * ((ey - 0.05) / 0.03))   # ~unit z near 5% EY
        # ROE: centred at 0.12, scaled by 0.10
        roe = row.get("roe") or row.get("return_on_equity")
        if pd.notna(roe):
            parts.append(+1.0 * ((float(roe) - 0.12) / 0.10))
        # Profit margin: centred at 0.10, scaled by 0.08
        pm = row.get("profit_margin") or row.get("profit_margins")
        if pd.notna(pm):
            parts.append(+1.0 * ((float(pm) - 0.10) / 0.08))
        # Revenue growth: centred at 0.07, scaled by 0.10
        rg = row.get("revenue_growth")
        if pd.notna(rg):
            parts.append(+1.0 * ((float(rg) - 0.07) / 0.10))
        # Debt/equity: centred at 1.0, scaled by 1.0, INVERTED
        de = row.get("debt_to_equity")
        if pd.notna(de):
            parts.append(-1.0 * ((float(de) - 1.0) / 1.0))

    if not parts:
        return _NEUTRAL
    z = float(np.mean(parts))
    return float(_clip01(_logistic(z, k=0.8)) * 100)


def _score_sentiment(row: pd.Series) -> float:
    n = row.get("n_articles")
    sm = row.get("sent_mean")
    if pd.isna(sm) or pd.isna(n) or n <= 0:
        return _NEUTRAL
    confidence = np.sqrt(min(float(n), 20.0) / 20.0)
    base = _logistic(float(sm) * 3.0)  # squash into 0..1
    momentum = row.get("sent_momentum_7d")
    boost = 0.0
    if pd.notna(momentum):
        boost = float(np.clip(momentum * 0.5, -0.15, 0.15))
    score = (_NEUTRAL + (base - 0.5) * 100 * confidence) + boost * 100
    return float(np.clip(score, 0, 100))


def _score_macro(row: pd.Series, regime: str) -> float:
    s = _NEUTRAL
    vix = row.get("vix_level")
    yc = row.get("yield_curve_2_10")
    dxy_chg = row.get("dxy_chg_5d")

    if pd.notna(vix):
        # Lower VIX => higher score for risk assets
        s += float(np.clip((20 - vix) * 1.5, -20, 20))
    if pd.notna(yc):
        s += float(np.clip(yc * 10, -10, 10))
    if pd.notna(dxy_chg):
        s += float(np.clip(-dxy_chg * 200, -10, 10))

    if regime == "crisis":
        # Invert: a low macro environment is itself the signal
        s = 100 - s
    return float(np.clip(s, 0, 100))


@dataclass
class RubricResult:
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    macro_score: float
    rubric_total: float
    rubric_direction: int
    breakdown: dict


def score_row(row: pd.Series, *, regime: str, asset_class: str,
              cfg: dict | None = None) -> RubricResult:
    cfg = cfg or _load_rubric()
    weights = cfg["regimes"].get(regime, cfg["regimes"]["chop"])["weights"]
    gating = cfg.get("gating", {})
    min_conv = float(gating.get("min_conviction", 70.0))

    t = _score_technical(row)
    f = _score_fundamental(row, asset_class)
    sn = _score_sentiment(row)
    m = _score_macro(row, regime)
    total = (weights["technical"] * t
             + weights["fundamental"] * f
             + weights["sentiment"] * sn
             + weights["macro"] * m)

    direction = 0
    if total >= min_conv and regime in ("bull", "chop"):
        direction = +1
    elif total <= (100 - min_conv) and regime in ("bear", "crisis"):
        direction = -1

    return RubricResult(
        technical_score=t, fundamental_score=f, sentiment_score=sn, macro_score=m,
        rubric_total=float(total), rubric_direction=direction,
        breakdown={"weights": weights, "regime": regime, "asset_class": asset_class},
    )


def score_frame(df: pd.DataFrame, *, regimes: pd.Series | None = None,
                cfg: dict | None = None) -> pd.DataFrame:
    """Vectorized-ish scoring across a full feature frame.

    `regimes` aligned with df.index — if None, uses 'chop' for all rows.
    `df` must contain `asset_class`. Returns df augmented with score columns.
    """
    cfg = cfg or _load_rubric()
    if regimes is None:
        regimes = pd.Series(["chop"] * len(df), index=df.index)

    out = df.copy()
    techs, funds, sents, macros, totals, dirs = [], [], [], [], [], []
    for i, row in df.iterrows():
        ac = row.get("asset_class", "us")
        rg = regimes.loc[i] if i in regimes.index else "chop"
        r = score_row(row, regime=rg, asset_class=ac, cfg=cfg)
        techs.append(r.technical_score)
        funds.append(r.fundamental_score)
        sents.append(r.sentiment_score)
        macros.append(r.macro_score)
        totals.append(r.rubric_total)
        dirs.append(r.rubric_direction)
    out["technical_score"] = techs
    out["fundamental_score"] = funds
    out["sentiment_score"] = sents
    out["macro_score"] = macros
    out["rubric_total"] = totals
    out["rubric_direction"] = dirs
    log.info(f"rubric scored rows={len(out)} mean_total={np.mean(totals):.2f}")
    return out
