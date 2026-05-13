"""Conviction gate.

Filters blended predictions down to the high-confidence subset that will be
emitted to the user / dashboard / paper-trade tape.

Rules (driven by config/rubric.yaml `gating` block):
  1. conviction >= min_conviction  (default 70)
  2. direction != 0
  3. If `require_regime_supportive` is true:
       - longs (direction=+1) only allowed in regimes 'bull' or 'chop'
       - shorts (direction=-1) only allowed in regimes 'bear' or 'crisis'
  4. (Optional) blocked tickers list — e.g. earnings-day blackout

The output is a frame containing only rows that pass the gate, plus a
`reasons` column explaining which rules triggered (for auditability).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from omni.utils.env import rubric as _load_rubric
from omni.utils.logger import log

_REGIME_LONG_OK = {"bull", "chop"}
_REGIME_SHORT_OK = {"bear", "crisis"}


@dataclass
class GateConfig:
    min_conviction: float = 70.0
    require_regime_supportive: bool = True
    blocked_tickers: tuple[str, ...] = ()

    @classmethod
    def from_yaml(cls) -> "GateConfig":
        cfg = _load_rubric().get("gating", {})
        return cls(
            min_conviction=float(cfg.get("min_conviction", 70.0)),
            require_regime_supportive=bool(cfg.get("require_regime_supportive", True)),
        )


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

    conv_ok = out["conviction"] >= cfg.min_conviction
    dir_ok = out["direction"] != 0
    ticker_ok = ~out["ticker"].astype(str).isin(set(cfg.blocked_tickers))

    if cfg.require_regime_supportive and regime_col in out.columns:
        rg = out[regime_col].astype(str)
        long_ok = (out["direction"] == 1) & rg.isin(_REGIME_LONG_OK)
        short_ok = (out["direction"] == -1) & rg.isin(_REGIME_SHORT_OK)
        regime_ok = long_ok | short_ok
    else:
        regime_ok = pd.Series(True, index=out.index)

    passes = conv_ok & dir_ok & ticker_ok & regime_ok

    reasons = []
    for i in out.index:
        rs = []
        if not conv_ok.loc[i]:
            rs.append(f"conviction<{cfg.min_conviction}")
        if not dir_ok.loc[i]:
            rs.append("direction=0")
        if not ticker_ok.loc[i]:
            rs.append("blocked_ticker")
        if not regime_ok.loc[i]:
            rs.append("regime_not_supportive")
        reasons.append(",".join(rs) if rs else "ok")
    out["passes_gate"] = passes
    out["reasons"] = reasons

    log.info(f"gate: {int(passes.sum())}/{n} pass "
             f"(min_conv={cfg.min_conviction}, regime_filter={cfg.require_regime_supportive})")
    return out


def emit(df: pd.DataFrame, *, cfg: GateConfig | None = None,
         regime_col: str = "regime") -> pd.DataFrame:
    """Convenience: `apply()` + return only rows where `passes_gate`."""
    g = apply(df, cfg=cfg, regime_col=regime_col)
    return g[g["passes_gate"]].drop(columns=["passes_gate"]).reset_index(drop=True)
