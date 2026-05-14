"""Windows Task Scheduler integration for the daily orchestrator.

Wraps the `schtasks` CLI so the dev-tab UI can install / uninstall / run /
query the scheduled task without shelling-out scattered commands. All
mutating ops (Create, Delete) require the host process to run as
administrator; read-only ops (Query) work without elevation.

Public API:
    is_admin()          -> bool
    is_installed(name)  -> bool
    status(name)        -> dict        {installed, last_run, last_result, next_run, state}
    install(name, ...)  -> (ok, msg)   creates SYSTEM-level daily trigger
    uninstall(name)     -> (ok, msg)   removes the task
    run_now(name)       -> (ok, msg)   triggers an immediate run

    install_weekly_retrain(...)        creates the Sunday 02:00 retrain task
    uninstall_weekly_retrain()
    weekly_retrain_status()

Convention:
    - Daily task    : "Omni_Stocks_Daily"          runs scripts/run_daily.bat at 18:30
    - Weekly retrain: "Omni_Stocks_Weekly_Retrain" runs scripts/run_retrain.bat Sun 02:00
"""
from __future__ import annotations

import csv
import io
import os
import subprocess
from pathlib import Path
from typing import Optional

DEFAULT_TASK_NAME = "Omni_Stocks_Daily"
DEFAULT_HOUR = 18
DEFAULT_MINUTE = 30

# Phase 24.18 — weekly retrain task
WEEKLY_TASK_NAME = "Omni_Stocks_Weekly_Retrain"
WEEKLY_DAY = "SUN"
WEEKLY_HOUR = 2
WEEKLY_MINUTE = 0


# ----------------------------------------------------------------------- helpers

def is_admin() -> bool:
    """Return True iff the current Python process has admin privileges.
    Always False on non-Windows."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _bat_path() -> Path:
    """Absolute path to the daily runner .bat. Resolved from this file's
    location so it works regardless of cwd."""
    return Path(__file__).resolve().parents[2] / "scripts" / "run_daily.bat"


def _retrain_bat_path() -> Path:
    """Absolute path to the weekly retrain .bat (Phase 24.18)."""
    return Path(__file__).resolve().parents[2] / "scripts" / "run_retrain.bat"


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run a subprocess; return (rc, stdout, stderr). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, shell=False)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError as e:
        return 127, "", str(e)
    except subprocess.TimeoutExpired:
        return 124, "", "schtasks timed out"
    except Exception as e:  # noqa: BLE001
        return 1, "", str(e)


# ----------------------------------------------------------------------- queries

def is_installed(name: str = DEFAULT_TASK_NAME) -> bool:
    """True iff a task with the given name exists."""
    if os.name != "nt":
        return False
    rc, _, _ = _run(["schtasks", "/Query", "/TN", name])
    return rc == 0


def status(name: str = DEFAULT_TASK_NAME) -> dict:
    """Structured status of the scheduled task.

    Returns dict with keys:
        installed     bool
        last_run      str|None    e.g. "5/13/2026 6:30:00 PM"
        last_result   str|None    e.g. "0" (success) or hex code
        next_run      str|None
        state         str|None    e.g. "Ready", "Running", "Disabled"
    """
    out = {"installed": False, "last_run": None, "last_result": None,
           "next_run": None, "state": None}
    if os.name != "nt":
        out["state"] = "non-windows"
        return out

    rc, stdout, stderr = _run(
        ["schtasks", "/Query", "/TN", name, "/FO", "CSV", "/V"],
    )
    if rc != 0:
        return out  # not installed

    out["installed"] = True
    try:
        reader = csv.reader(io.StringIO(stdout))
        rows = list(reader)
        if len(rows) < 2:
            return out
        header = [h.strip().strip('"') for h in rows[0]]
        # Verbose CSV may contain multiple data rows (one per trigger);
        # take the first one as canonical.
        first = [c.strip().strip('"') for c in rows[1]]
        m = dict(zip(header, first))
        out["last_run"] = m.get("Last Run Time") or None
        out["last_result"] = m.get("Last Result") or None
        out["next_run"] = m.get("Next Run Time") or None
        out["state"] = m.get("Status") or m.get("Scheduled Task State") or None
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------- mutations

def install(
    name: str = DEFAULT_TASK_NAME,
    *,
    hour: int = DEFAULT_HOUR,
    minute: int = DEFAULT_MINUTE,
    bat_path: Optional[Path] = None,
) -> tuple[bool, str]:
    """Install a daily Task Scheduler entry. Requires admin.

    The trigger fires every day at `hour:minute` local time, runs as the
    current user (so it has access to the venv + .env + Drive mount), and
    reuses the run_daily.bat wrapper.
    """
    if os.name != "nt":
        return False, "Task Scheduler is Windows-only"
    if not is_admin():
        return False, "admin rights required — re-launch as administrator"

    bat = bat_path or _bat_path()
    if not bat.exists():
        return False, f"runner not found: {bat}"

    # /SC DAILY /ST HH:MM /F (force overwrite if exists)
    # Use /RL HIGHEST so the task can write to data\lake without UAC prompts
    cmd = [
        "schtasks", "/Create", "/TN", name,
        "/TR", f'"{bat}"',
        "/SC", "DAILY",
        "/ST", f"{hour:02d}:{minute:02d}",
        "/RL", "HIGHEST",
        "/F",
    ]
    rc, stdout, stderr = _run(cmd, timeout=20)
    if rc == 0:
        return True, stdout.strip() or f"installed {name} @ {hour:02d}:{minute:02d}"
    return False, (stderr or stdout).strip() or f"schtasks failed (rc={rc})"


def uninstall(name: str = DEFAULT_TASK_NAME) -> tuple[bool, str]:
    """Remove the scheduled task. Requires admin."""
    if os.name != "nt":
        return False, "Task Scheduler is Windows-only"
    if not is_admin():
        return False, "admin rights required"
    rc, stdout, stderr = _run(["schtasks", "/Delete", "/TN", name, "/F"])
    if rc == 0:
        return True, stdout.strip() or f"uninstalled {name}"
    return False, (stderr or stdout).strip() or f"schtasks failed (rc={rc})"


def run_now(name: str = DEFAULT_TASK_NAME) -> tuple[bool, str]:
    """Trigger an immediate run of the task. Does NOT require admin —
    any user who can read the task can /Run it."""
    if os.name != "nt":
        return False, "Task Scheduler is Windows-only"
    rc, stdout, stderr = _run(["schtasks", "/Run", "/TN", name])
    if rc == 0:
        return True, stdout.strip() or f"triggered {name}"
    return False, (stderr or stdout).strip() or f"schtasks failed (rc={rc})"


# ============================================================================
# Phase 24.18 — Weekly retrain task management.
# Mirrors the daily-task helpers but with a WEEKLY trigger on Sundays at 02:00
# local time, running scripts/run_retrain.bat which calls
# `python scripts/retrain_all.py --horizons 1,5,21`.
# ============================================================================

def weekly_retrain_status() -> dict:
    """Status of the weekly retrain task. Same shape as status()."""
    return status(WEEKLY_TASK_NAME)


def is_weekly_retrain_installed() -> bool:
    return is_installed(WEEKLY_TASK_NAME)


def install_weekly_retrain(
    *,
    day: str = WEEKLY_DAY,
    hour: int = WEEKLY_HOUR,
    minute: int = WEEKLY_MINUTE,
    bat_path: Optional[Path] = None,
) -> tuple[bool, str]:
    """Install a weekly Task Scheduler entry for retraining all horizons.

    Default trigger: every Sunday at 02:00 local time. Runs as the current
    user so the venv + .env + Drive mount are accessible.
    """
    if os.name != "nt":
        return False, "Task Scheduler is Windows-only"
    if not is_admin():
        return False, "admin rights required — re-launch as administrator"

    bat = bat_path or _retrain_bat_path()
    if not bat.exists():
        return False, (
            f"runner not found: {bat}\n"
            "Create scripts/run_retrain.bat first (one-line wrapper around "
            "`.venv\\Scripts\\python.exe scripts\\retrain_all.py`)."
        )

    cmd = [
        "schtasks", "/Create", "/TN", WEEKLY_TASK_NAME,
        "/TR", f'"{bat}"',
        "/SC", "WEEKLY",
        "/D", day.upper(),
        "/ST", f"{hour:02d}:{minute:02d}",
        "/RL", "HIGHEST",
        "/F",
    ]
    rc, stdout, stderr = _run(cmd, timeout=20)
    if rc == 0:
        return True, stdout.strip() or (
            f"installed {WEEKLY_TASK_NAME} @ {day.upper()} "
            f"{hour:02d}:{minute:02d}"
        )
    return False, (stderr or stdout).strip() or f"schtasks failed (rc={rc})"


def uninstall_weekly_retrain() -> tuple[bool, str]:
    """Remove the weekly retrain task. Requires admin."""
    return uninstall(WEEKLY_TASK_NAME)


def run_weekly_retrain_now() -> tuple[bool, str]:
    """Trigger an immediate run of the weekly retrain task."""
    return run_now(WEEKLY_TASK_NAME)
