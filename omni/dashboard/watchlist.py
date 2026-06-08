"""User watchlist (favorites) — persisted to a small JSON file on disk.

Survives Streamlit reruns AND restarts (unlike st.session_state which
only lives in the browser session). One JSON file at:

    data/watchlist/favorites.json

Schema: {"items": [{"ticker": "RELIANCE", "asset_class": "india",
                     "display_name": "Reliance", "added_at": "..."}]}
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

WATCHLIST_PATH = Path("data/watchlist/favorites.json")

# Process-local lock guards the read-modify-write critical section so two
# Streamlit interactions in the same process can't clobber each other. The
# `.lock` sentinel guards CROSS-process races (best-effort, OS file lock
# would be ideal but is platform-specific; this is the same approach used
# by predict_log.py).
_RMW_LOCK = threading.Lock()
_LOCK_PATH = WATCHLIST_PATH.with_suffix(".lock")


def _acquire_file_lock(timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    while time.monotonic() < deadline:
        try:
            fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.05)
    return False


def _release_file_lock() -> None:
    try:
        _LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def _read() -> list[dict]:
    if not WATCHLIST_PATH.exists():
        return []
    try:
        with WATCHLIST_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", []) if isinstance(data, dict) else data
        return [i for i in items if isinstance(i, dict) and "ticker" in i]
    except Exception:  # noqa: BLE001
        return []


def _write(items: Iterable[dict]) -> None:
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"items": list(items)}
    tmp = WATCHLIST_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(WATCHLIST_PATH)


def list_favorites() -> list[dict]:
    return _read()


def add_favorite(ticker: str, asset_class: str, display_name: str = "") -> bool:
    """Returns True if added, False if already present."""
    ticker = ticker.upper().strip()
    with _RMW_LOCK:
        _acquire_file_lock()
        try:
            items = _read()
            if any(i["ticker"].upper() == ticker for i in items):
                return False
            items.append({
                "ticker": ticker,
                "asset_class": asset_class,
                "display_name": display_name or ticker,
                "added_at": datetime.now(UTC).isoformat(),
            })
            _write(items)
            return True
        finally:
            _release_file_lock()


def remove_favorite(ticker: str) -> bool:
    """Returns True if removed, False if not present."""
    ticker = ticker.upper().strip()
    with _RMW_LOCK:
        _acquire_file_lock()
        try:
            items = _read()
            new_items = [i for i in items if i["ticker"].upper() != ticker]
            if len(new_items) == len(items):
                return False
            _write(new_items)
            return True
        finally:
            _release_file_lock()


def is_favorite(ticker: str) -> bool:
    ticker = ticker.upper().strip()
    return any(i["ticker"].upper() == ticker for i in _read())


def clear_all() -> int:
    n = len(_read())
    _write([])
    return n
