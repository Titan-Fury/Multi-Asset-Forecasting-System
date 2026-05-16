"""Daily orchestrator entry point.

Thin wrapper around `python -m omni.orchestrator.production_runner`
so the dev-tab `Run daily orchestrator now` button + the cron/Task
Scheduler invocations can use a stable script path.

The Windows .bat (`scripts/run_daily.bat`) and the Linux/macOS cron
example in the dev tab both reference this entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    # Ensure the project root is on sys.path so `omni.*` imports resolve
    # when this script is run from anywhere.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    # Hand off to the production runner. argparse args pass through
    # transparently because we re-invoke its main() directly.
    from omni.orchestrator.production_runner import main as _runner_main
    try:
        _runner_main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
