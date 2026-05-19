"""Daily ingest runner.

Coordinates the per-source ingest jobs (prices, fundamentals, news, macro).
Each source is wrapped in try/except so a single failure doesn't stop the
whole run; results are returned as a dict for the orchestrator to record.

This is intentionally thin: the actual fetching logic lives in the per-source
modules under omni.ingest.*. Network calls are gated behind env-var checks
and source toggles in config/sources.yaml.
"""
from __future__ import annotations

import inspect
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

    # (name, module, primary_fn, fallback_fn) -- runner accepts the first
    # function that exists, so future renames don't silently disable a source.
    sources_to_run: list[tuple[str, str, tuple[str, ...]]] = []
    if enabled.get("prices", True) and (only is None or "prices" in only):
        sources_to_run.append(("prices", "omni.ingest.prices", ("ingest_universe", "ingest")))
    if enabled.get("fundamentals", True) and (only is None or "fundamentals" in only):
        sources_to_run.append(("fundamentals", "omni.ingest.fundamentals", ("ingest_universe", "ingest")))
    if enabled.get("news", False) and (only is None or "news" in only):
        sources_to_run.append(("news", "omni.ingest.news", ("ingest",)))
    if enabled.get("macro", True) and (only is None or "macro" in only):
        sources_to_run.append(("macro", "omni.ingest.macro", ("ingest_fred", "ingest")))

    for name, module_path, candidates in sources_to_run:
        try:
            mod = __import__(module_path, fromlist=list(candidates))
            fn = None
            for cand in candidates:
                fn = getattr(mod, cand, None)
                if callable(fn):
                    break
            if not callable(fn):
                out[name] = {"ok": False, "rows": 0,
                             "error": f"{module_path}.{{{','.join(candidates)}}} not found"}
                continue
            # Only pass run_date if the function actually accepts it.
            sig = inspect.signature(fn)
            kwargs: dict[str, Any] = {}
            if run_date and "run_date" in sig.parameters:
                kwargs["run_date"] = run_date
            rows = fn(**kwargs)
            if isinstance(rows, dict):
                n = sum(int(v) for v in rows.values() if isinstance(v, (int, float)))
            elif isinstance(rows, int):
                n = rows
            elif hasattr(rows, "__len__"):
                n = len(rows)
            else:
                n = -1
            out[name] = {"ok": True, "rows": n, "error": None}
            log.info(f"ingest {name}: rows={n}")
        except Exception as e:
            out[name] = {"ok": False, "rows": 0, "error": f"{type(e).__name__}: {e}"}
            log.warning(f"ingest {name} failed: {e}")

    return out
