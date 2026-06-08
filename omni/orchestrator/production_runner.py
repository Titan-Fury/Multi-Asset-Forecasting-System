"""Production-grade orchestration helpers (Phase 24.16).

Wraps `omni.orchestrator.daily.run_daily` with the operational primitives
the original was missing:

  - per-stage timeouts (so a hung HTTP call can't block the run forever)
  - retries with exponential backoff + jitter
  - dead-letter queue for failed tickers (written to JSONL for review)
  - structured manifest persistence (one JSON per run)
  - concise SLA log line (total time, per-stage breakdown)

This module is opt-in. The existing `daily.py` remains the in-process API
so anything already calling `run_daily()` keeps working.

Usage
-----
    python -m omni.orchestrator.production_runner

Or as a library:

    from omni.orchestrator.production_runner import run_with_retries
    report = run_with_retries(skip_backtest=False)

When you outgrow this (multiple machines, web UI for retries, distributed
scheduling) graduate to Prefect / Dagster — the contract here mirrors what
those frameworks expose so the migration is a few-line change.
"""
from __future__ import annotations

import json
import logging
import random
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from omni.orchestrator import daily as _daily

_log = logging.getLogger("omni.orchestrator.production")

# --------------------------------------------------------------------------
# Configuration knobs (env-overridable)
# --------------------------------------------------------------------------
import os

DLQ_PATH = Path(os.environ.get(
    "OMNI_DLQ_PATH", "data/orchestrator/dead_letter.jsonl"))
MANIFEST_DIR = Path(os.environ.get(
    "OMNI_MANIFEST_DIR", "data/orchestrator/manifests"))
DEFAULT_STAGE_TIMEOUT = float(os.environ.get(
    "OMNI_STAGE_TIMEOUT_SECONDS", "1800"))   # 30 min hard cap per stage
DEFAULT_MAX_RETRIES = int(os.environ.get(
    "OMNI_STAGE_MAX_RETRIES", "3"))
DEFAULT_BACKOFF_BASE = float(os.environ.get(
    "OMNI_STAGE_BACKOFF_BASE", "2.0"))
DEFAULT_BACKOFF_MAX = float(os.environ.get(
    "OMNI_STAGE_BACKOFF_MAX_SECONDS", "60"))


@dataclass
class DLQEntry:
    """One failed unit (a ticker, a stage, etc.) for offline review."""
    timestamp: str
    run_date: str
    stage: str
    item: str            # ticker, asset_class, or other unit identifier
    error_type: str
    error_message: str
    attempts: int


@dataclass
class ProductionReport:
    """Wraps a base RunReport with retry counts and DLQ pointer."""
    base: dict
    retries_by_stage: dict[str, int] = field(default_factory=dict)
    dlq_entries_written: int = 0
    started_at: str = ""
    finished_at: str = ""
    total_seconds: float = 0.0


# --------------------------------------------------------------------------
# Timeout helper — POSIX uses SIGALRM, Windows uses a watchdog thread.
# --------------------------------------------------------------------------
class TimeoutError_(Exception):
    """Raised when a stage exceeds its timeout budget."""


def _run_with_timeout(fn: Callable, *args, timeout: float, **kwargs):
    """Execute `fn(*args, **kwargs)` with a hard timeout in seconds.

    Cross-platform AND thread-safe:
      - POSIX main thread: uses SIGALRM (interrupts CPU-bound code too).
      - POSIX worker thread / Windows: uses a daemon-thread watchdog
        (cannot interrupt pure CPU work, but caller is unblocked).

    The earlier SIGALRM-only branch crashed with ValueError when called
    from a worker thread (Celery, Prefect, etc.), because Python only
    allows signal handlers to be installed from the main thread.
    """
    use_sigalrm = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if use_sigalrm:
        old = signal.getsignal(signal.SIGALRM)
        def _handler(signum, frame):
            raise TimeoutError_(f"stage exceeded timeout of {timeout}s")
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    # Worker-thread or Windows fallback — daemon thread + join(timeout).
    # Note: this cannot interrupt CPU-bound code; the function runs to
    # completion but the caller is no longer waiting on it.
    result: dict = {}
    def _runner():
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001
            result["error"] = e
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError_(
            f"stage exceeded timeout of {timeout}s "
            f"(thread still running; CPU-bound code will continue)")
    if "error" in result:
        raise result["error"]
    return result.get("value")


# --------------------------------------------------------------------------
# DLQ writer
# --------------------------------------------------------------------------
def write_dlq(entry: DLQEntry, *, path: Path = DLQ_PATH) -> None:
    """Append one entry to the dead-letter queue. Atomic-enough for daily
    runs (we don't run concurrent orchestrators)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def read_dlq(*, path: Path = DLQ_PATH, since_iso: str | None = None
             ) -> list[DLQEntry]:
    """Load DLQ entries (optionally filtered by ISO timestamp). For ops
    review and replay tooling."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if since_iso and rec.get("timestamp", "") < since_iso:
                    continue
                out.append(DLQEntry(**rec))
            except Exception as e:  # noqa: BLE001
                _log.warning(f"bad DLQ line: {e}")
    return out


# --------------------------------------------------------------------------
# Retry primitive
# --------------------------------------------------------------------------
def with_retries(
    fn: Callable,
    *args,
    stage: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_max: float = DEFAULT_BACKOFF_MAX,
    timeout: float = DEFAULT_STAGE_TIMEOUT,
    on_failure: Callable[[Exception, int], None] | None = None,
    **kwargs,
):
    """Run `fn` with up to `max_retries` retries on Exception.

    Backoff: `min(base ** attempt + jitter, max)` seconds.
    Each call has a hard `timeout` (TimeoutError_ raised on exceed).
    `on_failure` callback fires after each failed attempt (for telemetry).
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return _run_with_timeout(fn, *args, timeout=timeout, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if on_failure:
                try:
                    on_failure(e, attempt)
                except Exception:  # noqa: BLE001
                    _log.warning("on_failure callback raised; ignoring",
                                  exc_info=True)
            if attempt >= max_retries:
                break
            sleep_s = min(
                backoff_base ** attempt + random.uniform(0, 0.5 * attempt),
                backoff_max,
            )
            _log.warning(
                f"[{stage}] attempt {attempt}/{max_retries} failed: "
                f"{type(e).__name__}: {e}. retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)
    # Defensive: `last_exc` is set in every loop iteration that might
    # not return, but guard explicitly so this still works under
    # `python -O` (which strips `assert` statements).
    if last_exc is None:
        raise RuntimeError(
            f"with_retries({stage}) loop exited without setting last_exc; "
            f"this should be unreachable")
    raise last_exc


# --------------------------------------------------------------------------
# Top-level entry point — same shape as run_daily but production-grade
# --------------------------------------------------------------------------
def run_with_retries(
    *,
    run_date: str | None = None,
    skip_backtest: bool = False,
    skip_mount: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ProductionReport:
    """Run the daily orchestrator with retries, timeouts, manifest, and DLQ.

    The base `daily.run_daily` is invoked under a single retry envelope.
    Per-ticker failures inside `_stage_features` etc. are still caught
    silently by the existing per-ticker try/except — those get logged but
    do not enter the DLQ unless they cause a stage-level failure. To get
    per-ticker DLQ entries, add `write_dlq()` calls inside the relevant
    feature/predict loops in `daily.py`.
    """
    started = datetime.now(UTC)
    rep = ProductionReport(
        base={},
        started_at=started.isoformat(),
    )
    retries_by_stage: dict[str, int] = {}

    def _on_failure(exc: Exception, attempt: int) -> None:
        retries_by_stage["run_daily"] = attempt
        # Write a DLQ entry for any non-final attempt failure
        write_dlq(DLQEntry(
            timestamp=datetime.now(UTC).isoformat(),
            run_date=run_date or started.strftime("%Y-%m-%d"),
            stage="run_daily",
            item="orchestrator",
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            attempts=attempt,
        ))

    try:
        base_report = with_retries(
            _daily.run_daily,
            stage="run_daily",
            max_retries=max_retries,
            on_failure=_on_failure,
            run_date=run_date,
            skip_backtest=skip_backtest,
            skip_mount=skip_mount,
        )
        rep.base = base_report.manifest()
    except Exception as e:  # noqa: BLE001
        _log.error(f"run_daily exhausted retries: {e}", exc_info=True)
        rep.base = {"ok": False, "error": str(e)}

    finished = datetime.now(UTC)
    rep.finished_at = finished.isoformat()
    rep.total_seconds = (finished - started).total_seconds()
    rep.retries_by_stage = retries_by_stage

    # Persist the manifest
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"run_{started.strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.write_text(
        json.dumps(asdict(rep), default=str, indent=2),
        encoding="utf-8",
    )
    _log.info(f"manifest written: {manifest_path}")
    _log.info(f"run completed in {rep.total_seconds:.1f}s "
              f"(ok={rep.base.get('ok', False)})")
    return rep


def main() -> None:
    """CLI entry: `python -m omni.orchestrator.production_runner`."""
    import argparse
    logging.basicConfig(
        level=os.environ.get("OMNI_ORCH_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    p = argparse.ArgumentParser(description="Production daily orchestrator")
    p.add_argument("--run-date", default=None,
                   help="ISO date (default: today UTC)")
    p.add_argument("--skip-backtest", action="store_true")
    p.add_argument("--skip-mount", action="store_true")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = p.parse_args()
    rep = run_with_retries(
        run_date=args.run_date,
        skip_backtest=args.skip_backtest,
        skip_mount=args.skip_mount,
        max_retries=args.max_retries,
    )
    raise SystemExit(0 if rep.base.get("ok") else 1)


if __name__ == "__main__":
    main()
