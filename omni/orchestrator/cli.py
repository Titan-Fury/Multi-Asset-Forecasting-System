"""CLI entry point for the daily orchestrator.

Usage:
    python -m omni.orchestrator.cli                 # full daily run
    python -m omni.orchestrator.cli --skip-backtest # faster daily, no walk-forward fold metrics
    python -m omni.orchestrator.cli --skip-mount    # already-mounted Drive (dev)
    python -m omni.orchestrator.cli --run-date 2026-05-07
"""
from __future__ import annotations

import argparse
import json
import sys

from omni.orchestrator.daily import run_daily


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="omni.orchestrator",
                                description="Omni_Stocks daily run")
    p.add_argument("--run-date", default=None,
                   help="ISO date YYYY-MM-DD; defaults to today UTC")
    p.add_argument("--skip-backtest", action="store_true",
                   help="Skip walk-forward backtest (use last-known summary)")
    p.add_argument("--skip-mount", action="store_true",
                   help="Skip rclone mount/unmount (dev / already-mounted)")
    p.add_argument("--manifest-only", action="store_true",
                   help="Print run manifest as JSON to stdout instead of pretty log")
    args = p.parse_args(argv)

    rep = run_daily(
        run_date=args.run_date,
        skip_backtest=args.skip_backtest,
        skip_mount=args.skip_mount,
    )

    if args.manifest_only:
        print(json.dumps(rep.manifest(), default=str, indent=2))
    else:
        print()
        print(f"run_date     : {rep.run_date}")
        print(f"overall      : {'OK' if rep.ok else 'FAILED'}")
        for s in rep.stages:
            mark = "ok " if s.ok else "FAIL"
            err = f" -- {s.error}" if s.error else ""
            print(f"  [{mark}] {s.name:18s} {s.seconds:6.1f}s{err}")
        if rep.persist:
            print(f"\npersist      : {rep.persist.n_signals} signals "
                  f"({rep.persist.n_emitted} emitted), "
                  f"{rep.persist.n_predictions} predictions")
            for k, v in rep.persist.paths.items():
                print(f"  {k:20s} {v}")

    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
