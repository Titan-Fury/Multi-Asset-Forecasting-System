"""Retrain everything: universal models + stack ensemble + India specialist.

Usage:
    python scripts/retrain_all.py              # all horizons
    python scripts/retrain_all.py --horizons 5 # single horizon
    python scripts/retrain_all.py --skip-stack # quick path

Order matters:
    1. Universal XGBoost (base learner of the stack) - per horizon
    2. Stack ensemble (XGB + HistGB + logistic meta) - per horizon
    3. India specialist - reads the universal+stack outputs

Each step emits OOF AUC so a regression in the model is visible
immediately. Saves to data/models/ overwriting the previous bundles.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omni.utils.logger import log  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", default="1,5,21",
                          help="comma-separated list of horizons in days")
    parser.add_argument("--skip-stack", action="store_true",
                          help="skip stack retrain (faster, only updates "
                                "universal models)")
    parser.add_argument("--skip-india", action="store_true",
                          help="skip India specialist retrain")
    args = parser.parse_args()
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    t0 = time.time()
    log.info(f"=== retrain_all start  horizons={horizons} ===")

    # 1. Universal models
    log.info("[1/3] Universal XGBoost (per-horizon walk-forward)")
    import subprocess
    cmd = [sys.executable, "scripts/train_universal_model.py",
            "--horizons", ",".join(str(h) for h in horizons)]
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        log.error(f"universal train failed (rc={res.returncode})")
        sys.exit(res.returncode)

    # 2. Stack ensemble (if available)
    if not args.skip_stack:
        log.info("[2/3] Stack ensemble (XGB + HistGB + logistic meta)")
        try:
            from omni.models.stack import retrain_all_horizons  # type: ignore
            retrain_all_horizons(horizons=horizons)
        except (ImportError, AttributeError):
            log.warning(
                "omni.models.stack does not expose retrain_all_horizons(). "
                "Skipping. The existing stack bundles will be used as base "
                "learners but will not be re-fit. Add a CLI to "
                "omni/models/stack.py to enable full stack retrain."
            )

    # 3. India specialist
    if not args.skip_india:
        log.info("[3/3] India specialist")
        try:
            from omni.specialists.india import retrain  # type: ignore
            retrain()
        except (ImportError, AttributeError):
            log.warning(
                "omni.specialists.india does not expose retrain(). "
                "Skipping. To enable, add a top-level retrain() function "
                "to omni/specialists/india.py."
            )

    elapsed = time.time() - t0
    log.info(f"=== retrain_all done in {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()
