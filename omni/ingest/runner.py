"""Daily ingest runner.

Coordinates the per-source ingest jobs (prices, fundamentals, news, macro).
Each source is wrapped in try/except so a single failure doesn't stop the
whole run; results are returned as a dict for the orchestrator to record.

This is intentionally thin: the actual fetching logic lives in the per-source
modules under omni.ingest.*. Network calls are gated behind env-var checks
and source toggles in config/sources.yaml.
"""
from __future__ import annotations

from typing import Any

from omni.utils.env import sources as _sources
from omni.utils.logger import log


def run_daily_ingest(*, run_date: str | None = None,
                     only: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Run the configured ingestors for `run_date`.

    Returns a dict {source_name: {"ok": bool, "rows": int, "error": str|None}}.
    """
    cfg = _sources()
    enabled = cfg.get("enabled", {})

    out: dict[str, Any] = {}

    sources_to_run = []
    if enabled.get("prices", True) and (only is None or "prices" in only):
        sources_to_run.append(("prices", "omni.ingest.prices", "ingest"))
    if enabled.get("fundamentals", True) and (only is None or "fundamentals" in only):
        sources_to_run.append(("fundamentals", "omni.ingest.fundamentals", "ingest"))
    if enabled.get("news", False) and (only is None or "news" in only):
        sources_to_run.append(("news", "omni.ingest.news", "ingest"))
    if enabled.get("macro", True) and (only is None or "macro" in only):
        sources_to_run.append(("macro", "omni.ingest.macro", "ingest"))

    for name, module_path, fn_name in sources_to_run:
        try:
            mod = __import__(module_path, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if fn is None:
                out[name] = {"ok": False, "rows": 0, "error": f"{module_path}.{fn_name} not found"}
                continue
            rows = fn(run_date=run_date) if run_date else fn()
            n = int(rows) if isinstance(rows, int) else (
                len(rows) if hasattr(rows, "__len__") else -1)
            out[name] = {"ok": True, "rows": n, "error": None}
            log.info(f"ingest {name}: rows={n}")
        except Exception as e:
            out[name] = {"ok": False, "rows": 0, "error": f"{type(e).__name__}: {e}"}
            log.warning(f"ingest {name} failed: {e}")

    return out
