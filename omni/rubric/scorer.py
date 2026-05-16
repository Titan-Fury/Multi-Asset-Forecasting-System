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
    +15 if dist_sma_50 > dist_sma_200 > 0 (close above 50d above 200d), -15 if reverse
    +15 if rsi_14 in (40, 70), -10 if rsi_14 > 80 or < 30
    +10 if macd_hist_pct > 0
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

    # Phase 24.17 — features.technical now produces dist_sma_* (unitless
    # distance from MA) instead of raw sma_*. Use those: dist_sma_50 vs
    # dist_sma_200 (close above 50d-MA AND above 200d-MA = uptrend).
    dist50 = row.get("dist_sma_50"); dist200 = row.get("dist_sma_200")
    if pd.notna(dist50) and pd.notna(dist200):
        # 50d > 200d distance ≈ "golden cross"-like trend up
        if dist50 > dist200 and dist200 > 0:
            s += 10
        elif dist50 < dist200 and dist200 < 0:
            s -= 10

    rsi = row.get("rsi_14")
    if pd.notna(rsi):
        if 40 <= rsi <= 70:
            s += 10
        elif rsi > 80 or rsi < 30:
            s -= 10

    # macd_hist replaced by macd_hist_pct (% of close) for ticker invariance.
    macd_h = row.get("macd_hist_pct")
    if pd.isna(macd_h):
        macd_h = row.get("macd_hist")  # backwards-compat
    if pd.notna(macd_h) and macd_h > 0:
        s += 5
    if pd.notna(row.get("vol_ratio")) and row["vol_ratio"] < 1.0:
        s += 5

    ts = row.get("trend_strength")
    if pd.notna(ts):
        s += float(np.clip(ts * 100, -10, 10))

    return float(np.clip(s, 0, 100))


def _score_fundamental(row: pd.Series, asset_class: str) -> float:
    """Phase 24.16 — robust fundamentals scoring.

    Three calibration bugs fixed in this iteration:

      1. yfinance returns `debt_to_equity` in PERCENT (e.g. 79.548 means
         79.5% / 0.795 ratio) for many large caps. The earlier code
         treated it as a raw ratio so a single ticker like AAPL
         (D/E=79.5) produced z=-78 and tanked the entire average to 0.
         We now auto-detect: any value >= 5 is interpreted as percent and
         divided by 100. (Real-world D/E ratios > 5 are almost
         exclusively reported as percent in yfinance and Screener.)

      2. yfinance occasionally reports `dividend_yield` as percent
         (0.37 = 0.37%) and sometimes as decimal (0.0037 = 0.37%); same
         heuristic applies.

      3. Per-component z-scores are now CLIPPED to [-3, +3] before
         aggregation, and we use the MEDIAN (not mean) so a single
         extreme reading (e.g. AAPL's post-buyback ROE = 141%) doesn't
         dominate the score.

    For US equities, point-in-time fundamentals can have legitimately
    extreme values (high-growth tech, post-buyback share counts, etc.).
    The clipping keeps the SIGN of the signal but caps its magnitude so
    the rubric stays informative even on outlier names.
    """
    if asset_class in ("crypto", "fx"):
        return _NEUTRAL
    raw_parts: list[float] = []  # uncapped, for diagnostic; capped goes to score

    # Preferred path: pre-computed cross-sectional z-scores
    for col, sign in [
        ("earnings_yield_z", +1),
        ("return_on_equity_z", +1),
        ("profit_margin_z", +1),
        ("revenue_growth_z", +1),
        ("debt_to_equity_z", -1),
    ]:
        v = row.get(col)
        if pd.notna(v):
            raw_parts.append(sign * float(v))

    # Fallback path: derive z-units from raw fundamentals when no
    # cross-sectional pre-computation is available.
    if not raw_parts:
        # ---- valuation ----
        # Earnings yield from PE: 1/pe_ratio, centred on long-run S&P avg ~5%
        pe = row.get("pe_ratio") or row.get("trailing_pe")
        if pd.notna(pe) and pe and pe > 0:
            ey = 1.0 / float(pe)
            raw_parts.append((ey - 0.05) / 0.03)
        # ---- quality ----
        # ROE: centred at 0.12, scaled by 0.10
        roe = row.get("roe") or row.get("return_on_equity")
        if pd.notna(roe):
            roe_v = float(roe)
            # ROE > 5 is almost certainly in percent (yfinance drift); rescale
            if abs(roe_v) > 5:
                roe_v = roe_v / 100.0
            raw_parts.append((roe_v - 0.12) / 0.10)
        # Profit margin: centred at 0.10, scaled by 0.08
        pm = row.get("profit_margin") or row.get("profit_margins")
        if pd.notna(pm):
            pm_v = float(pm)
            if abs(pm_v) > 5:
                pm_v = pm_v / 100.0
            raw_parts.append((pm_v - 0.10) / 0.08)
        # ---- growth ----
        # Revenue growth: centred at 0.07, scaled by 0.10
        rg = row.get("revenue_growth")
        if pd.notna(rg):
            rg_v = float(rg)
            if abs(rg_v) > 5:
                rg_v = rg_v / 100.0
            raw_parts.append((rg_v - 0.07) / 0.10)
        # ---- leverage ----
        # Debt/equity: centred at 1.0, scaled by 1.0, INVERTED.
        # yfinance returns this in PERCENT for large caps (e.g. AAPL=79.5
        # meaning 0.795 ratio). Auto-detect: any value >= 5 is percent.
        de = row.get("debt_to_equity")
        if pd.notna(de):
            de_v = float(de)
            if de_v >= 5:
                de_v = de_v / 100.0   # percent -> ratio
            raw_parts.append(-1.0 * ((de_v - 1.0) / 1.0))

    if not raw_parts:
        return _NEUTRAL

    # ---- robust aggregation -------------------------------------------------
    # Clip each component to +/- 3 z-units and use the MEDIAN.
    # Median is robust to one extreme reading (e.g. ROE of 14 z-units).
    capped = [float(np.clip(p, -3.0, 3.0)) for p in raw_parts]
    z = float(np.median(capped))
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


def _score_microstructure(row: pd.Series) -> float:
    """Phase 24.16 — 5th pillar.

    Combines the intraday-flow signals already computed by
    `omni/features/intraday.py` (and surfaced on every PredictionCard) into
    a 0..100 score. Designed for SME / low-coverage names where the
    fundamental pillar is uninformative (often returns neutral 50).

    Inputs (each contributes 0-15 points around the 50 neutral base):
      gap_open_pct                  +ve gap = up-pressure (capped at +/-2%)
      vwap_dev_bps                  price > VWAP = buyers in control
      opening_range_breakout        +1 / -1 / 0 (breakout / breakdown / inside)
      last_hour_return_pct          closing strength (smart-money window)
      volume_z_today                z-score of today's volume vs 60d
      range_compression             low value = compressed range, breakout setup

    Returns neutral 50 if no inputs are available (graceful degradation).
    """
    s = _NEUTRAL
    parts: list[tuple[str, float, float]] = []

    gap = row.get("gap_open_pct")
    if pd.notna(gap):
        # +/- 2% gap maps to +/- 12 points
        contrib = float(np.clip(float(gap) * 6, -12, 12))
        parts.append(("gap", contrib, 12.0))

    vwd = row.get("vwap_dev_bps")
    if pd.notna(vwd):
        # +/- 50 bps from VWAP = +/- 10 points
        contrib = float(np.clip(float(vwd) / 5.0, -10, 10))
        parts.append(("vwap_dev", contrib, 10.0))

    orb = row.get("opening_range_breakout")
    if pd.notna(orb):
        contrib = float(np.clip(float(orb) * 8, -8, 8))
        parts.append(("orb", contrib, 8.0))

    lhr = row.get("last_hour_return_pct")
    if pd.notna(lhr):
        # Closing strength: +/- 1% in last hour = +/- 8 points
        contrib = float(np.clip(float(lhr) * 8, -8, 8))
        parts.append(("last_hour", contrib, 8.0))

    vz = row.get("volume_z_today")
    if pd.notna(vz):
        # Volume spike: ABSOLUTE bump (high volume confirms direction
        # already encoded in the other components — small +ve only)
        contrib = float(np.clip(abs(float(vz)) * 2, 0, 6))
        parts.append(("vol_z", contrib, 6.0))

    rc = row.get("range_compression")
    if pd.notna(rc) and float(rc) < 1.0:
        # Compressed range = coiled spring; small +ve regardless of direction
        contrib = float(np.clip((1.0 - float(rc)) * 4, 0, 4))
        parts.append(("range_compress", contrib, 4.0))

    if not parts:
        return _NEUTRAL

    # Weighted sum already in points. Total available = ~48 points up or
    # down from neutral 50. Real-world will rarely max out.
    s += sum(c for (_n, c, _w) in parts)
    return float(np.clip(s, 0, 100))


def _score_oi(row: pd.Series) -> float:
    """Phase 24.16 — 6th pillar (F&O open-interest sentiment).

    For NSE F&O-listed names, options positioning is one of the cleanest
    leading indicators of institutional bias. Inputs come either directly
    on the feature row (when the orchestrator's feature builder joins
    OI snapshots) OR are pulled live from `omni.ingest.sources.nse_oi`
    when `asset_class == 'india'` and `ticker` is on the row.

    Signals (sums to +/- 20 points around neutral 50):
      oi_pcr                  PCR > 1.3 = bearish bias (lots of put writing
                              implies put-side hedging by smart money);
                              PCR < 0.7 = bullish bias.
                              Inverted in literature — we treat HIGH PCR as
                              bullish-on-fade because retail buys puts at tops.
                              Default: high PCR = mildly bullish.
      max_pain_dist_pct       price > max-pain = upward pin pressure
                              into expiry; price < max-pain = downward.
      oi_change_pct_5d        rising OI + rising price = strong long
                              build-up; rising OI + falling price = short
                              build-up. We only have OI direction here, so
                              we use abs(change) as a "conviction multiplier"
                              on top of price direction.

    Graceful degradation: returns neutral 50 if no OI data.
    """
    pcr = row.get("oi_pcr")
    mp_dist = row.get("max_pain_dist_pct")
    oi_chg = row.get("oi_change_pct_5d")

    # If none of the OI fields are on the row, skip pulling live (caller
    # may be in a loop where live calls would be too slow). Score 50.
    if all(pd.isna(v) for v in (pcr, mp_dist, oi_chg)):
        return _NEUTRAL

    s = _NEUTRAL
    if pd.notna(pcr) and float(pcr) > 0:
        # PCR ~1.0 = neutral. >1.5 = mildly bullish (puts being written
        # at support); <0.6 = mildly bearish (calls being written at
        # resistance). Cap contribution at +/- 8.
        contrib = float(np.clip((float(pcr) - 1.0) * 8, -8, 8))
        s += contrib

    if pd.notna(mp_dist):
        # Spot vs max-pain: when spot is BELOW max-pain, options writers
        # benefit from the price drifting UP into expiry (bullish pin).
        # When spot is ABOVE max-pain, drift down. We invert the sign so
        # negative dist (below MP) gives + score.
        contrib = float(np.clip(-float(mp_dist) * 1.2, -7, 7))
        s += contrib

    if pd.notna(oi_chg):
        # Rising OI = conviction event. We don't know direction here in
        # isolation, so we add only a small +ve contribution capped at +5
        # for >20% growth. Falling OI = squeeze risk, small -ve.
        contrib = float(np.clip(float(oi_chg) * 0.25, -5, 5))
        s += contrib

    return float(np.clip(s, 0, 100))


def _score_relative_strength(row: pd.Series) -> float:
    """Phase 24.16 — 7th pillar.

    Stock's recent return vs its sector / market benchmark. The cleanest
    equity signal: a stock outperforming its sector is more likely to
    keep doing so (momentum literature: Jegadeesh & Titman 1993; sector-
    relative variant: Asness 2013).

    Inputs (any subset works):
      rs_vs_sector_20d        ticker_ret_20d - sector_ret_20d (decimal,
                              e.g. 0.05 = +5% outperformance over 20d)
      rs_vs_market_60d        long-horizon RS vs broad market (NIFTY/SPY)
      pair_predicted_alpha_pct  expected forward alpha from pair_signal
                              (already on the card_dict for India + US)

    Graceful degradation: returns neutral 50 if no RS field is available.
    Heuristic fallback: if `ret_20d` exists but no sector benchmark, use
    the raw return as a weak proxy (50% weight of normal RS contribution).
    """
    rs20 = row.get("rs_vs_sector_20d")
    rs60 = row.get("rs_vs_market_60d")
    alpha = row.get("pair_predicted_alpha_pct")

    if all(pd.isna(v) for v in (rs20, rs60, alpha)):
        # Weak fallback: raw ret_20d. Treats positive return as RS-positive
        # (assumes market is roughly flat). Half the impact of true RS.
        ret20 = row.get("ret_20d")
        if pd.notna(ret20):
            contrib = float(np.clip(float(ret20) * 100, -10, 10))  # halved scale
            return float(np.clip(_NEUTRAL + contrib, 0, 100))
        return _NEUTRAL

    s = _NEUTRAL
    if pd.notna(rs20):
        # +/- 5% outperformance over 20d -> +/- 18 points
        contrib = float(np.clip(float(rs20) * 360, -18, 18))
        s += contrib
    if pd.notna(rs60):
        # +/- 10% outperformance over 60d -> +/- 12 points (sustained alpha)
        contrib = float(np.clip(float(rs60) * 120, -12, 12))
        s += contrib
    if pd.notna(alpha):
        # Predicted forward alpha (decimal %) — already a forecast, smaller weight
        contrib = float(np.clip(float(alpha) * 4, -8, 8))
        s += contrib

    return float(np.clip(s, 0, 100))


def _score_insider(row: pd.Series) -> float:
    """Phase 24.16 — 8th pillar.

    Insider / institutional activity. Uses recent NSE bulk-deal and
    block-deal data, BSE notices, and (when available) promoter-holding
    deltas to detect smart-money positioning.

    Inputs (any subset works):
      insider_net_z_20d        z-score of net insider buying over 20d
                               (calculated by the feature builder when
                               bulk/block deal data is present)
      promoter_holding_chg     +ve = promoters increasing stake (bullish)
      block_deal_net_value_cr  net block-deal value last 5 sessions, INR cr
      fii_dii_net_z_20d        country-level institutional flow (broadcasted)

    Graceful degradation: returns neutral 50 if no insider field exists.
    For US tickers we additionally check Form 4 metrics if present.
    """
    insider_z = row.get("insider_net_z_20d")
    promoter_chg = row.get("promoter_holding_chg")
    block_net = row.get("block_deal_net_value_cr")
    fii_dii_z = row.get("fii_dii_net_z_20d")

    if all(pd.isna(v) for v in (insider_z, promoter_chg, block_net, fii_dii_z)):
        return _NEUTRAL

    s = _NEUTRAL
    if pd.notna(insider_z):
        # +/- 2 z-units = +/- 14 points
        contrib = float(np.clip(float(insider_z) * 7, -14, 14))
        s += contrib
    if pd.notna(promoter_chg):
        # +/- 1% promoter stake change = +/- 10 points (rare but huge signal)
        contrib = float(np.clip(float(promoter_chg) * 1000, -10, 10))
        s += contrib
    if pd.notna(block_net):
        # Block deals are rare and chunky. INR 100 cr net = +/- 6 points cap.
        contrib = float(np.clip(float(block_net) / 16.0, -6, 6))
        s += contrib
    if pd.notna(fii_dii_z):
        # Country-level flows: smaller weight since same value applies to
        # every ticker. +/- 2 z-units = +/- 6 points.
        contrib = float(np.clip(float(fii_dii_z) * 3, -6, 6))
        s += contrib

    return float(np.clip(s, 0, 100))


def _score_earnings_momentum(row: pd.Series) -> float:
    """Phase 24.16 — 9th pillar.

    Earnings momentum: are earnings ACCELERATING or decelerating? Reads
    revenue + earnings growth fields (yfinance / Screener) and combines
    them into a 0..100 score.

    Quality literature: Asness, Frazzini, Pedersen 2013 — companies with
    accelerating earnings outperform those with decelerating. Most useful
    on equities with quarterly reporting; bypassed for crypto / FX.

    Inputs (any subset works):
      revenue_growth         YoY revenue growth (decimal)
      earnings_growth        YoY earnings growth (decimal)
      eps_revision_3m        analyst consensus 3-month revision (decimal)
      surprise_pct           latest earnings surprise vs consensus (decimal)

    Graceful degradation: returns neutral 50 if no input is available.
    """
    if row.get("asset_class") in ("crypto", "fx"):
        return _NEUTRAL
    rg = row.get("revenue_growth")
    eg = row.get("earnings_growth")
    rev = row.get("eps_revision_3m")
    sur = row.get("surprise_pct")

    if all(pd.isna(v) for v in (rg, eg, rev, sur)):
        return _NEUTRAL

    s = _NEUTRAL
    if pd.notna(rg):
        rg_v = float(rg)
        if abs(rg_v) > 5:        # percent->decimal autodetect
            rg_v /= 100.0
        # +/- 20% YoY growth = +/- 16 points
        contrib = float(np.clip((rg_v - 0.07) * 80, -16, 16))
        s += contrib
    if pd.notna(eg):
        eg_v = float(eg)
        if abs(eg_v) > 5:
            eg_v /= 100.0
        # Accelerating earnings has a higher payoff than revenue
        contrib = float(np.clip((eg_v - 0.10) * 50, -16, 16))
        s += contrib
    if pd.notna(rev):
        # Analyst revision: +/- 5% in 3 months = +/- 8 points
        contrib = float(np.clip(float(rev) * 160, -8, 8))
        s += contrib
    if pd.notna(sur):
        # Earnings surprise: +/- 10% beat = +/- 8 points
        contrib = float(np.clip(float(sur) * 80, -8, 8))
        s += contrib

    return float(np.clip(s, 0, 100))


@dataclass
class RubricResult:
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    macro_score: float
    microstructure_score: float       # Phase 24.16 — 5th pillar
    oi_score: float                   # Phase 24.16 — 6th pillar (F&O OI)
    relative_strength_score: float    # Phase 24.16 — 7th pillar
    insider_score: float              # Phase 24.16 — 8th pillar
    earnings_momentum_score: float    # Phase 24.16 — 9th pillar
    rubric_total: float
    rubric_direction: int
    breakdown: dict


def score_row(row: pd.Series, *, regime: str, asset_class: str,
              cfg: dict | None = None) -> RubricResult:
    cfg = cfg or _load_rubric()
    weights = cfg["regimes"].get(regime, cfg["regimes"]["chop"])["weights"]

    # Phase 25 / Item 5: optional learned rubric weights. When the JSON
    # file `data/models/learned_rubric_weights.json` exists AND the env
    # flag `OMNI_LEARNED_RUBRIC=1` is set, blend the YAML prior with the
    # data-fit posterior 50/50 (or per-env override). This recovers the
    # actually-predictive pillar contributions per regime instead of
    # relying on the hand-tuned weights alone.
    try:
        import os as _os
        if _os.environ.get("OMNI_LEARNED_RUBRIC", "").strip() == "1":
            from omni.rubric.learned import load_learned_weights, blend_weights
            learned = load_learned_weights()
            if learned and regime in learned:
                alpha = float(_os.environ.get("OMNI_LEARNED_RUBRIC_ALPHA", "0.5"))
                weights = blend_weights(weights, learned[regime], alpha=alpha)
    except Exception as _exc:  # noqa: BLE001
        # Never let the learned-rubric path break scoring
        pass

    gating = cfg.get("gating", {})
    min_conv = float(gating.get("min_conviction", 70.0))

    t = _score_technical(row)
    f = _score_fundamental(row, asset_class)
    sn = _score_sentiment(row)
    m = _score_macro(row, regime)
    ms = _score_microstructure(row)
    oi = _score_oi(row)
    rs = _score_relative_strength(row)
    ins = _score_insider(row)
    em = _score_earnings_momentum(row)

    # Backwards-compat: pillars added across phases default to 0 weight
    # if the loaded rubric.yaml predates them. Old 4/5/6-pillar configs
    # keep working unchanged.
    w_micro = float(weights.get("microstructure", 0.0))
    w_oi = float(weights.get("oi", 0.0))
    w_rs = float(weights.get("relative_strength", 0.0))
    w_ins = float(weights.get("insider", 0.0))
    w_em = float(weights.get("earnings_momentum", 0.0))
    total = (weights["technical"] * t
             + weights["fundamental"] * f
             + weights["sentiment"] * sn
             + weights["macro"] * m
             + w_micro * ms
             + w_oi * oi
             + w_rs * rs
             + w_ins * ins
             + w_em * em)

    direction = 0
    if total >= min_conv and regime in ("bull", "chop"):
        direction = +1
    elif total <= (100 - min_conv) and regime in ("bear", "crisis"):
        direction = -1

    return RubricResult(
        technical_score=t, fundamental_score=f, sentiment_score=sn, macro_score=m,
        microstructure_score=ms, oi_score=oi,
        relative_strength_score=rs, insider_score=ins,
        earnings_momentum_score=em,
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
    techs, funds, sents, macros = [], [], [], []
    micros, ois, rss, inss, ems = [], [], [], [], []
    totals, dirs = [], []
    for i, row in df.iterrows():
        ac = row.get("asset_class", "us")
        rg = regimes.loc[i] if i in regimes.index else "chop"
        r = score_row(row, regime=rg, asset_class=ac, cfg=cfg)
        techs.append(r.technical_score)
        funds.append(r.fundamental_score)
        sents.append(r.sentiment_score)
        macros.append(r.macro_score)
        micros.append(r.microstructure_score)
        ois.append(r.oi_score)
        rss.append(r.relative_strength_score)
        inss.append(r.insider_score)
        ems.append(r.earnings_momentum_score)
        totals.append(r.rubric_total)
        dirs.append(r.rubric_direction)
    out["technical_score"] = techs
    out["fundamental_score"] = funds
    out["sentiment_score"] = sents
    out["macro_score"] = macros
    out["microstructure_score"] = micros
    out["oi_score"] = ois
    out["relative_strength_score"] = rss
    out["insider_score"] = inss
    out["earnings_momentum_score"] = ems
    out["rubric_total"] = totals
    out["rubric_direction"] = dirs
    log.info(f"rubric scored rows={len(out)} mean_total={np.mean(totals):.2f}")
    return out
