"""Verify per-(asset_class, regime) min_conviction overrides."""
import pandas as pd
from omni.ensemble.gate import GateConfig, apply

cfg = GateConfig.from_yaml()
print("loaded overrides:")
for ac, rgmap in (cfg.min_conviction_overrides or {}).items():
    print(f"  {ac}: {rgmap}")
print()

# Build a synthetic frame with one row per (ac, regime) so we can read
# the resolved threshold for each combination.
rows = []
for ac in ("india", "us", "crypto", "fx"):
    for rg in ("bull", "chop", "bear", "crisis"):
        rows.append({"ticker": f"{ac.upper()}_{rg.upper()}",
                     "asset_class": ac, "regime": rg,
                     "direction": +1, "conviction": 50.0})
df = pd.DataFrame(rows)
g = apply(df, cfg=cfg)

print(f"{'ticker':<24s} {'thresh':>7s}  result")
for _, r in g.iterrows():
    # eff conviction printed via reasons when the gate fires; here all
    # rows have conv=50 long-aligned across regimes (long-bull=50, long-chop=45,
    # long-bear=25, long-crisis=15 effective)
    th = cfg.threshold_for(r["asset_class"], r["regime"])
    print(f"{r['ticker']:<24s} {th:>7.1f}  passes={bool(r['passes_gate'])}  reasons={r['reasons']}")
