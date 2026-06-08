"""Degradation monitor for the RAG layer (Phase 24.16).

The RAG pipeline has many "fail soft" paths: when the embedder API returns
nothing, when the reranker is missing, when Qdrant is unreachable, the code
silently falls back to deterministic explanations or zero vectors. That
graceful degradation hides real outages — operators only learn about it
when users complain.

This module gives every RAG component a uniform way to record degradation
events that are:

  - written to a structured JSONL log (for offline analysis)
  - counted in-process (for /metrics)
  - emitted to a webhook (Slack/Discord/PagerDuty) when configured
  - rate-limited so a sustained outage doesn't spam the channel

Usage in any RAG component:

    from omni.llm.rag.degradation import monitor

    if not vec.any():
        monitor.record(
            component="embedder",
            severity="warn",
            event="zero_vector",
            details={"text_len": len(text), "endpoint": "openrouter"},
        )

Environment toggles:
    OMNI_DEGRADE_LOG_PATH   (default: data/rag/degradation.jsonl)
    OMNI_DEGRADE_WEBHOOK    (optional Slack/Discord webhook URL)
    OMNI_DEGRADE_RATE_LIMIT_SECONDS  (default: 300; min seconds between
                                     duplicate webhook posts per (component,event))
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger("omni.llm.rag.degradation")


@dataclass
class DegradationEvent:
    timestamp: str
    component: str
    severity: str         # "info" | "warn" | "error"
    event: str            # short slug, e.g. "zero_vector", "reranker_missing"
    details: dict = field(default_factory=dict)


class DegradationMonitor:
    """Process-wide singleton; thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[tuple[str, str, str], int] = defaultdict(int)
        self._last_webhook: dict[tuple[str, str], float] = {}
        self._log_path = Path(os.environ.get(
            "OMNI_DEGRADE_LOG_PATH", "data/rag/degradation.jsonl"))
        self._webhook_url = os.environ.get("OMNI_DEGRADE_WEBHOOK")
        self._rate_limit = float(os.environ.get(
            "OMNI_DEGRADE_RATE_LIMIT_SECONDS", "300"))

    # ---- public API ----------------------------------------------------
    def record(self, *, component: str, severity: str, event: str,
               details: dict[str, Any] | None = None) -> None:
        """Record a degradation event. Safe to call from any thread."""
        ev = DegradationEvent(
            timestamp=datetime.now(UTC).isoformat(),
            component=component,
            severity=severity,
            event=event,
            details=details or {},
        )
        with self._lock:
            self._counts[(component, severity, event)] += 1
        # Logging (always) + JSONL (always) + webhook (rate-limited)
        log_msg = (f"[degraded:{severity}] {component}/{event} "
                   f"{json.dumps(ev.details)}")
        if severity == "error":
            _log.error(log_msg)
        elif severity == "warn":
            _log.warning(log_msg)
        else:
            _log.info(log_msg)
        self._append_jsonl(ev)
        if severity in ("warn", "error"):
            self._maybe_webhook(ev)

    def counts(self) -> dict[str, int]:
        """Return a flat snapshot keyed by 'component.event'. For /metrics."""
        with self._lock:
            return {f"{c}.{e}": n for (c, _s, e), n in self._counts.items()}

    def reset(self) -> None:
        """Clear in-process counts. JSONL log is kept."""
        with self._lock:
            self._counts.clear()
            self._last_webhook.clear()

    # ---- private -------------------------------------------------------
    def _append_jsonl(self, ev: DegradationEvent) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            # Never let logging failures crash the caller
            _log.warning(f"degradation jsonl append failed: {e}")

    def _maybe_webhook(self, ev: DegradationEvent) -> None:
        if not self._webhook_url:
            return
        key = (ev.component, ev.event)
        now = time.time()
        with self._lock:
            last = self._last_webhook.get(key, 0)
            if now - last < self._rate_limit:
                return
            self._last_webhook[key] = now
        try:
            import requests  # transitive dep via yfinance
            payload = {
                "text": (
                    f":warning: *RAG degradation* — `{ev.component}` / "
                    f"`{ev.event}` ({ev.severity})\n"
                    f"```{json.dumps(ev.details, indent=2)[:1500]}```"
                ),
                "username": "omni-stocks-rag",
            }
            requests.post(self._webhook_url, json=payload, timeout=4.0)
        except Exception as e:  # noqa: BLE001
            _log.warning(f"degradation webhook failed: {e}")


# Module-level singleton — import as `from ... import monitor`.
monitor = DegradationMonitor()
