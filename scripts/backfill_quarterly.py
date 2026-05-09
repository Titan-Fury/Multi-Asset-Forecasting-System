"""Backfill quarterly fundamentals from yfinance for the lake universe.

Run once after backfill_universe.py. Only fetches equities (skips crypto/fx).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from omni.lake.backfill import universe_pairs   # noqa: E402
from omni.lake.quarterly import backfill_quarterly  # noqa: E402
from omni.utils.logger import log                # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill quarterly fundamentals")
    p.add_argument("--asset-classes", default="us,india")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    classes = [c.strip() for c in args.asset_classes.split(",") if c.strip()]
    pairs = universe_pairs(classes, limit=args.limit)
    log.info(f"backfill_quarterly: {len(pairs)} tickers")
    t0 = time.time()
    stats = backfill_quarterly(pairs)
    elapsed = time.time() - t0
    print(f"\nbackfill_quarterly: {stats}")
    print(f"elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
