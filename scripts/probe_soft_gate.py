"""Unit-test the soft regime gate vs the legacy hard gate."""
import pandas as pd
from omni.ensemble.gate import GateConfig, apply

# Synthetic frame: same ticker scored under different regimes/directions/convictions
rows = [
    # (ticker, conviction, direction, regime, expected_soft_pass, comment)
    ("LONG_BULL_HI",   60, +1, "bull",   True,  "aligned, strong: should pass"),
    ("SHORT_BULL_HI",  35, -1, "bull",   True,  "counter-trend, strong (eff 17.5): SOFT passes"),
    ("SHORT_BULL_LO",  16, -1, "bull",   False, "counter-trend, weak (eff 8): blocked"),
    ("LONG_BEAR_HI",   45, +1, "bear",   False, "counter-trend long in bear (eff 22.5): SOFT passes"),
    ("LONG_BEAR_LO",   18, +1, "bear",   False, "counter-trend long, weak: blocked"),
    ("SHORT_CHOP",     20, -1, "chop",   True,  "short in chop (eff 18): SOFT passes (legacy hard would block)"),
    ("LONG_CRISIS_HI", 60, +1, "crisis", True,  "long crisis (eff 18): passes"),
]
df = pd.DataFrame([
    {"ticker": t, "conviction": c, "direction": d, "regime": r}
    for (t, c, d, r, _, _) in rows
])

# Test soft gate
soft_cfg = GateConfig(min_conviction=15.0, require_regime_supportive=True, soft_regime=True)
soft = apply(df, cfg=soft_cfg)

# Test legacy hard gate
hard_cfg = GateConfig(min_conviction=15.0, require_regime_supportive=True, soft_regime=False)
hard = apply(df, cfg=hard_cfg)

print(f"{'ticker':<16s} {'conv':>5s} {'dir':>4s} {'regime':>7s}  "
      f"{'eff':>5s} {'soft':>5s} {'hard':>5s}  comment")
print("-" * 110)
for i, r in enumerate(rows):
    t, c, d, rg, _exp, comment = r
    eff = float(soft.iloc[i]["effective_conviction"])
    spass = bool(soft.iloc[i]["passes_gate"])
    hpass = bool(hard.iloc[i]["passes_gate"])
    print(f"{t:<16s} {c:>5} {d:>4} {rg:>7s}  {eff:>5.1f} {str(spass):>5s} {str(hpass):>5s}  {comment}")
