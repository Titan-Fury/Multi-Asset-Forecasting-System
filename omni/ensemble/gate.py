"""Conviction gate.

Filters blended predictions down to the high-confidence subset that will be
emitted to the user / dashboard / paper-trade tape.

Rules (driven by config/rubric.yaml `gating` block):
  1. conviction (after the regime-direction multiplier) >= min_conviction
  2. direction != 0
  3. If `require_regime_supportive` is true, conviction is *multiplied*
     by `regime_dir_multiplier(regime, direction)` before being compared
     against `min_conviction`. The multiplier is (default):

         direction \\ regime |  bull   chop   bear   crisis
         long  (+1)          |  1.00   0.90   0.50   0.30
         short (-1)          |  0.50   0.90   1.00   0.80

     This replaces the legacy hard binary block (long allowed only in
     bull+chop, short only in bear+crisis), which empirically blocked
     a basket of 7 correctly-bearish calls on 2026-05-08 — every one
     of which was down 1-22% over h+5. A soft multiplier admits the
     high-conviction names whose direction disagrees with the regime
     while still penalising marginal calls into the regime.
  4. (Optional) blocked tickers list — e.g. earnings-day blackout

The output is a frame containing only rows that pass the gate, plus a
`reasons` column explaining which rules triggered (for auditability).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from omni.utils.env import rubric as _load_rubric
from omni.utils.logger import log

_REGIME_LONG_OK = {"bull", "chop"}
_REGIME_SHORT_OK = {"bear", "crisis"}

# Soft regime-direction multiplier applied to conviction BEFORE the
# min_conviction check. Indexed by (regime, direction).
# Default mapping:
#   - aligned (long-bull, short-bear): 1.0  (no penalty)
#   - chop:                            0.9  (mild penalty either side)
#   - counter-trend in trending regime:0.3-0.5 (heavy penalty)
#   - long in crisis:                  0.3  (very heavy penalty)
#   - short in crisis:                 0.8  (volatility regime, shorts work)
_REGIME_DIR_MULT_DEFAULT: dict[tuple[str, int], float] = {
    ("bull",   +1): 1.00, ("bull",   -1): 0.50,
    ("chop",   +1): 0.90, ("chop",   -1): 0.90,
    ("bear",   +1): 0.50, ("bear",   -1): 1.00,
    ("crisis", +1): 0.30, ("crisis", -1): 0.80,
}


def regime_dir_multiplier(regime: str, direction: int) -> float:
    """Look up the soft regime-direction multiplier. Defaults to 1.0 if the
    regime label is unknown (so we never silently zero out a signal)."""
    rg = (regime or "").strip().lower()
    return float(_REGIME_DIR_MULT_DEFAULT.get((rg, int(direction)), 1.0))


@dataclass
class GateConfig:
    min_conviction: float = 70.0
    require_regime_supportive: bool = True
    blocked_tickers: tuple[str, ...] = ()
    # Soft-gate flag (Phase 25 / Item 1). When true, the regime filter
    # multiplies conviction instead of zero-ing it out. When false, the
    # legacy hard-block behaviour is used so backtests against the old
    # logic remain reproducible.
    soft_regime: bool = True
    # Phase 25 / Item 2: per-(asset_class, regime) min_conviction map.
    # Layout: {"india": {"bull": 15.0, ...}, "us": {...}, ...}.
    # When a key is missing for the row's (asset_class, regime), the
    # global `min_conviction` above is used as the fallback.
    min_conviction_overrides: dict | None = None

    @classmethod
    def from_yaml(cls) -> "GateConfig":
        cfg = _load_rubric().get("gating", {})
        return cls(
            min_conviction=float(cfg.get("min_conviction", 70.0)),
            require_regime_supportive=bool(cfg.get("require_regime_supportive", True)),
            soft_regime=bool(cfg.get("soft_regime", True)),
            min_conviction_overrides=cfg.get("min_conviction_overrides", {}) or {},
        )

    def threshold_for(self, asset_class: str | None,
                       regime: str | None) -> float:
        """Resolve the effective min_conviction for one row.

        Lookup order: overrides[asset_class][regime] -> global min_conviction.
        """
        if not self.min_conviction_overrides:
            return float(self.min_conviction)
        ac = (asset_class or "").strip().lower()
        rg = (regime or "").strip().lower()
        ac_map = self.min_conviction_overrides.get(ac)
        if isinstance(ac_map, dict):
            v = ac_map.get(rg)
            if v is not None:
                return float(v)
        return float(self.min_conviction)


def apply(df: pd.DataFrame, *, cfg: GateConfig | None = None,
          regime_col: str = "regime") -> pd.DataFrame:
    """Return df augmented with `passes_gate` (bool) and `reasons` (str).

    Required input columns: conviction, direction, ticker, regime_col.
    Does NOT drop rows -- callers can filter on `passes_gate` themselves so
    explainability stays attached.
    """
    cfg = cfg or GateConfig.from_yaml()
    out = df.copy()
    n = len(out)

    dir_ok = out["direction"] != 0
    ticker_ok = ~out["ticker"].astype(str).isin(set(cfg.blocked_tickers))

    # Compute the per-row regime-direction multiplier (1.0 when filter is off
    # or the regime column is missing).
    if cfg.require_regime_supportive and regime_col in out.columns:
        rg = out[regime_col].astype(str).str.lower()
        if cfg.soft_regime:
            # Phase 25 / Optimization: vectorize the (regime, direction)
            # lookup via pd.Series.map() over a tuple-keyed dict instead
            # of looping with regime_dir_multiplier per row.
            keys = list(zip(rg, out["direction"].astype(int)))
            mult = pd.Series(
                [_REGIME_DIR_MULT_DEFAULT.get(k, 1.0) for k in keys],
                index=out.index, dtype=float,
            )
        else:
            # Legacy hard binary block — kept for backtest reproducibility.
            long_ok = (out["direction"] == 1) & rg.isin(_REGIME_LONG_OK)
            short_ok = (out["direction"] == -1) & rg.isin(_REGIME_SHORT_OK)
            mult = pd.Series(
                np.where(long_ok | short_ok, 1.0, 0.0),
                index=out.index,
            )
    else:
        mult = pd.Series(1.0, index=out.index)

    eff_conv = out["conviction"].astype(float) * mult

    # Per-row min_conviction threshold (Phase 25 / Item 2). Falls back to
    # the global cfg.min_conviction when no asset_class/regime override
    # matches. Vectorized via a flat tuple-keyed map so this is O(n)
    # in C dict lookups instead of O(n) Python method calls.
    if cfg.min_conviction_overrides and "asset_class" in out.columns:
        flat: dict[tuple[str, str], float] = {}
        for ac_k, rg_map in cfg.min_conviction_overrides.items():
            if isinstance(rg_map, dict):
                for rg_k, val in rg_map.items():
                    flat[(str(ac_k).lower(), str(rg_k).lower())] = float(val)
        ac_lower = out["asset_class"].astype(str).str.lower()
        rg_lower = (out[regime_col].astype(str).str.lower()
                    if regime_col in out.columns
                    else pd.Series([""] * len(out), index=out.index))
        keys = list(zip(ac_lower, rg_lower))
        thresholds = pd.Series(
            [flat.get(k, float(cfg.min_conviction)) for k in keys],
            index=out.index, dtype=float,
        )
    else:
        thresholds = pd.Series(float(cfg.min_conviction), index=out.index)
    conv_ok = eff_conv >= thresholds

    passes = conv_ok & dir_ok & ticker_ok

    # Vectorized reasons string. Building per-row Python strings was the
    # backtest hot path; np.where on each rule is O(n) in C and ~50x faster.
    parts: list[pd.Series] = []
    parts.append(pd.Series(
        np.where(~conv_ok.to_numpy(),
                 (eff_conv.round(1).astype(str)
                  + "<" + thresholds.round(1).astype(str) + ";").to_numpy(),
                 ""),
        index=out.index,
    ))
    parts.append(pd.Series(
        np.where(~dir_ok.to_numpy(), "direction=0;", ""),
        index=out.index,
    ))
    parts.append(pd.Series(
        np.where(~ticker_ok.to_numpy(), "blocked_ticker;", ""),
        index=out.index,
    ))
    # Annotate when the multiplier was below 1.0 — pure information,
    # doesn't itself fail the gate.
    parts.append(pd.Series(
        np.where(mult.to_numpy() < 0.999,
                 ("regime_mult=" + mult.round(2).astype(str) + ";").to_numpy(),
                 ""),
        index=out.index,
    ))
    reasons = parts[0].str.cat(parts[1:], sep="").str.rstrip(";")
    reasons = reasons.where(reasons != "", "ok")
    out["passes_gate"] = passes
    out["reasons"] = reasons.values
    out["effective_conviction"] = eff_conv.values

    log.info(f"gate: {int(passes.sum())}/{n} pass "
             f"(min_conv={cfg.min_conviction}, "
             f"regime_filter={cfg.require_regime_supportive}, "
             f"soft={cfg.soft_regime})")
    return out


def emit(df: pd.DataFrame, *, cfg: GateConfig | None = None,
         regime_col: str = "regime") -> pd.DataFrame:
    """Convenience: `apply()` + return only rows where `passes_gate`."""
    g = apply(df, cfg=cfg, regime_col=regime_col)
    return g[g["passes_gate"]].drop(columns=["passes_gate"]).reset_index(drop=True)
