"""CLI: backfill the Omni_Stocks data lake.

Usage:
  python scripts/backfill_universe.py --mode full --asset-classes us,india --limit 30 --lookback-years 5
  python scripts/backfill_universe.py --mode incremental
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from omni.lake.backfill import backfill_universe  # noqa: E402
from omni.lake.io import lake_summary  # noqa: E402
from omni.utils.logger import log  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill the Omni_Stocks data lake")
    p.add_argument("--mode", choices=["full", "incremental"], default="incremental",
                    help="full = fresh N-year pull; incremental = only since last date in lake")
    p.add_argument("--asset-classes", default="us,india,crypto,fx",
                    help="comma-separated subset of asset classes to backfill")
    p.add_argument("--lookback-years", type=int, default=15,
                    help="how many years of history for full mode")
    p.add_argument("--limit", type=int, default=None,
                    help="cap the number of tickers (useful for smoke tests)")
    p.add_argument("--no-fundamentals", action="store_true",
                    help="skip fundamentals snapshot")
    p.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to sleep between tickers (avoid rate limits)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    classes = [c.strip() for c in args.asset_classes.split(",") if c.strip()]
    log.info(f"=== backfill_universe START === mode={args.mode} "
              f"classes={classes} limit={args.limit} lookback={args.lookback_years}y")
    t0 = time.time()
    summary = backfill_universe(
        asset_classes=classes,
        limit=args.limit,
        lookback_years=args.lookback_years,
        mode=args.mode,
        do_fundamentals=not args.no_fundamentals,
        sleep_between=args.sleep,
    )
    elapsed = time.time() - t0
    print("\n=== backfill summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nelapsed: {elapsed:.1f}s")

    print("\n=== lake summary ===")
    print(json.dumps(lake_summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
