"""Standalone HMM regime refit entry point.

Runs the live regime fitter (`omni.ondemand.regime_live.fit_or_load`)
which:
  - pulls SPX (^GSPC) + macro_live (VIX, etc.) over the lookback window,
  - fits the 4-state Gaussian HMM,
  - writes the per-row regime labels to `data/regime/latest.parquet`.

Used by:
  - dev-tab `Refit regime (HMM)` button
  - cron/orchestrator if the daily run wants a forced regime refresh
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pandas as pd

    from omni.ondemand.regime_live import fit_or_load
    from omni.utils.logger import log

    res = fit_or_load()
    if res is None:
        log.error("regime fit failed (insufficient data or fetch error)")
        return 1
    feats, out = res

    # Persist the per-row labels so the rest of the pipeline (notably the
    # learned-rubric trainer + regime-stack trainer) can read them via
    # data/regime/latest.parquet.
    target = Path("data") / "regime" / "latest.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    if "regime" not in feats.columns:
        log.error("regime fit returned no 'regime' column")
        return 1
    persisted = feats[["date", "regime"]].copy()
    persisted["date"] = pd.to_datetime(persisted["date"])
    persisted.to_parquet(target, index=False)
    log.info(f"regime labels saved -> {target} (n={len(persisted)}, "
              f"today={out.label_today}, asof={out.asof})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
