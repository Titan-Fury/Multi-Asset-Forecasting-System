"""Persistent search history (Google-style: time-stamped, recent + frequent).

Survives both Streamlit reruns AND restarts. Written to:

    data/watchlist/search_history.json

Schema:
    {
        "items": [
            {
                "ticker": "RELIANCE",
                "asset_class": "india",
                "display_name": "Reliance Industries",
                "first_searched_at": "2026-05-09T14:32:00Z",
                "last_searched_at":  "2026-05-09T14:55:00Z",
                "hit_count": 7,
                "last_proba_up": 0.547,
                "last_direction": 0,
                "last_conviction": 1.6,
                "last_regime": "bull",
                "exchange": "NSE",
            },
            ...
        ]
    }

The file is bounded to 200 entries and rotated by `last_searched_at`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HISTORY_PATH = Path("data/watchlist/search_history.json")
MAX_ENTRIES = 200


def _read() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", []) if isinstance(data, dict) else data
        return [i for i in items if isinstance(i, dict) and "ticker" in i]
    except Exception:  # noqa: BLE001
        return []


def _write(items) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"items": list(items)}
    tmp = HISTORY_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(HISTORY_PATH)


def record_search(ticker: str, asset_class: str, *,
                    display_name: str = "", exchange: str = "",
                    proba_up: float | None = None,
                    direction: int | None = None,
                    conviction: float | None = None,
                    regime: str | None = None) -> None:
    """Insert or upsert a search-history entry. Bumps hit_count + timestamp."""
    if not ticker:
        return
    ticker = ticker.upper().strip()
    now = datetime.now(timezone.utc).isoformat()
    items = _read()
    existing = next((i for i in items if i.get("ticker", "").upper() == ticker
                       and i.get("asset_class") == asset_class), None)
    if existing is None:
        existing = {
            "ticker": ticker,
            "asset_class": asset_class,
            "display_name": display_name or ticker,
            "exchange": exchange,
            "first_searched_at": now,
            "last_searched_at": now,
            "hit_count": 1,
        }
        items.append(existing)
    else:
        existing["last_searched_at"] = now
        existing["hit_count"] = int(existing.get("hit_count", 0)) + 1
        if display_name:
            existing["display_name"] = display_name
        if exchange:
            existing["exchange"] = exchange
    if proba_up is not None:
        existing["last_proba_up"] = round(float(proba_up), 4)
    if direction is not None:
        existing["last_direction"] = int(direction)
    if conviction is not None:
        existing["last_conviction"] = round(float(conviction), 2)
    if regime:
        existing["last_regime"] = str(regime)

    # Sort by recency, cap to MAX_ENTRIES
    items.sort(key=lambda x: x.get("last_searched_at", ""), reverse=True)
    _write(items[:MAX_ENTRIES])


def list_recent(limit: int = 20) -> list[dict]:
    items = _read()
    items.sort(key=lambda x: x.get("last_searched_at", ""), reverse=True)
    return items[:limit]


def list_frequent(limit: int = 10) -> list[dict]:
    items = _read()
    items.sort(key=lambda x: (-int(x.get("hit_count", 0)),
                                 x.get("last_searched_at", "")))
    return items[:limit]


def remove(ticker: str, asset_class: str | None = None) -> int:
    """Drop entries matching ticker (and asset_class if given). Returns count."""
    if not ticker:
        return 0
    ticker = ticker.upper().strip()
    items = _read()
    keep = []
    n_dropped = 0
    for it in items:
        if (it.get("ticker", "").upper() == ticker
                and (asset_class is None or it.get("asset_class") == asset_class)):
            n_dropped += 1
            continue
        keep.append(it)
    if n_dropped:
        _write(keep)
    return n_dropped


def clear_all() -> int:
    n = len(_read())
    _write([])
    return n


def stats() -> dict:
    items = _read()
    if not items:
        return {"total_searches": 0, "unique_tickers": 0, "most_searched": None}
    total = sum(int(i.get("hit_count", 0)) for i in items)
    most = max(items, key=lambda x: int(x.get("hit_count", 0)))
    return {
        "total_searches": total,
        "unique_tickers": len(items),
        "most_searched": most.get("ticker"),
        "most_searched_count": int(most.get("hit_count", 0)),
    }
