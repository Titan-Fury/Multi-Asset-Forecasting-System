"""Production observability for Omni_Stocks (Phase 24.16).

Single import point for metrics, tracing, and structured events. Uses
prometheus_client when available, falls back to in-process counters
otherwise — so importing this module never breaks a deploy that hasn't
installed the full observability stack yet.

The metrics defined here answer the questions the auditor flagged you
can't answer today:

  - Prediction latency p50/p99
  - Cache hit/miss rates per tier
  - LLM backend success/failure rates per provider
  - RAG grounding score distributions
  - Feature drift (data_quality_score) over time
  - Model prediction confidence (conviction) distributions
  - Data freshness (when did each artifact last update)

Usage:

    from omni.observability.metrics import metrics

    with metrics.timer("predict_latency_seconds", labels={"path": "live"}):
        card = predict_ticker(...)

    metrics.counter("predictions_total",
                    labels={"asset_class": "us", "direction": "long"}).inc()

    metrics.histogram("data_quality_score", labels={"asset_class": "us"})\\
           .observe(card.data_quality_score)

Scraping:
    Mount `/metrics` on the FastAPI app (already done in app.py:metrics()).
    Configure Prometheus to scrape it.
    Build Grafana dashboards from these metric names.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import defaultdict
from typing import Any

_log = logging.getLogger("omni.observability")


# --------------------------------------------------------------------------
# Prometheus client (optional). Without it, we keep in-process counters so
# nothing ever crashes; you just don't get the /metrics endpoint payload.
# --------------------------------------------------------------------------
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        generate_latest,
    )
    from prometheus_client import (  # type: ignore
        Counter as _PromCounter,
    )
    from prometheus_client import (
        Gauge as _PromGauge,
    )
    from prometheus_client import (
        Histogram as _PromHistogram,
    )
    _HAS_PROMETHEUS = True
    _registry = CollectorRegistry()
except ImportError:
    _HAS_PROMETHEUS = False
    _registry = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


# Default histogram buckets — tuned for prediction latency (sub-second to
# minutes) and rates 0..1 like grounding scores / confidences.
_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
    1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
)
_RATE_BUCKETS = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)
_SCORE_BUCKETS = (
    0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
)


# --------------------------------------------------------------------------
# Fallback in-process counters (used when prometheus_client is missing)
# --------------------------------------------------------------------------
class _FallbackCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[tuple, float] = defaultdict(float)

    def inc(self, *, labels: dict[str, str] | None = None,
            amount: float = 1.0) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] += amount

    def snapshot(self) -> dict[tuple, float]:
        with self._lock:
            return dict(self._values)


class _FallbackHistogram:
    def __init__(self, buckets: tuple[float, ...]) -> None:
        self._lock = threading.Lock()
        self._buckets = buckets
        self._counts: dict[tuple, list[int]] = defaultdict(
            lambda: [0] * (len(buckets) + 1))
        self._sums: dict[tuple, float] = defaultdict(float)
        self._n: dict[tuple, int] = defaultdict(int)

    def observe(self, value: float, *,
                labels: dict[str, str] | None = None) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._n[key] += 1
            self._sums[key] += value
            # Prometheus histograms are CUMULATIVE: every bucket whose upper
            # bound is >= the observation gets incremented. The previous
            # implementation only incremented the first matching bucket,
            # which made le="..." counts wrong. The +Inf bucket (last slot)
            # is also always incremented.
            buckets = self._counts[key]
            for i, b in enumerate(self._buckets):
                if value <= b:
                    buckets[i] += 1
            buckets[-1] += 1


# --------------------------------------------------------------------------
# Public API — Metrics facade
# --------------------------------------------------------------------------
class Metrics:
    """Process-wide singleton. Define metrics lazily on first access so
    import order doesn't matter."""

    def __init__(self) -> None:
        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}
        self._lock = threading.Lock()

        # Define the canonical metric set up-front so /metrics shows them
        # even before the first observation. This makes Grafana dashboards
        # easier (panels don't break on empty series).
        self._declare()

    # ------------------------------------------------------------------ declare
    def _declare(self) -> None:
        # Predictions
        self.counter("predictions_total",
                     "Total predictions served",
                     labels=("asset_class", "direction", "path"))
        self.histogram("prediction_latency_seconds",
                       "Wall time of a prediction request, seconds",
                       buckets=_LATENCY_BUCKETS,
                       labels=("path",))
        self.histogram("prediction_conviction",
                       "Conviction score of emitted predictions, 0..100",
                       buckets=_SCORE_BUCKETS,
                       labels=("asset_class",))
        self.histogram("data_quality_score",
                       "Per-prediction data_quality_score, 0..100",
                       buckets=_SCORE_BUCKETS,
                       labels=("asset_class",))

        # Cache
        self.counter("cache_total",
                     "Cache lookups",
                     labels=("tier", "outcome"))  # outcome: hit|miss

        # LLM backends
        self.counter("llm_calls_total",
                     "LLM backend calls",
                     labels=("provider", "outcome"))  # ok|error|timeout
        self.histogram("llm_latency_seconds",
                       "LLM backend call latency",
                       buckets=_LATENCY_BUCKETS,
                       labels=("provider",))

        # RAG
        self.histogram("rag_grounding_score",
                       "Grounding score per RAG response, 0..1",
                       buckets=_RATE_BUCKETS, labels=())
        self.counter("rag_path_total",
                     "RAG response path",
                     labels=("path",))   # rag|rag_low_grounding|deterministic

        # Regime
        self.counter("regime_label_flips_total",
                     "Number of times the daily regime label changed")
        self.gauge("regime_current",
                   "Current regime label as numeric (0=bull,1=chop,2=bear,3=crisis)")

        # Models
        self.gauge("model_age_seconds",
                   "Wall-clock age of the universal model bundle since training")

        # Data freshness
        self.gauge("artifact_freshness_seconds",
                   "Wall-clock age of a data artifact",
                   labels=("artifact",))   # predictions|signals|regime|features

    # ------------------------------------------------------------------ counter
    def counter(self, name: str, description: str = "",
                labels: tuple[str, ...] = ()) -> Any:
        with self._lock:
            if name in self._counters:
                return self._counters[name]
            if _HAS_PROMETHEUS:
                c = _PromCounter(name, description, labels,
                                 registry=_registry)
            else:
                c = _FallbackCounter()
            self._counters[name] = c
            return c

    # ------------------------------------------------------------------ histo
    def histogram(self, name: str, description: str = "",
                  buckets: tuple[float, ...] = _LATENCY_BUCKETS,
                  labels: tuple[str, ...] = ()) -> Any:
        with self._lock:
            if name in self._histograms:
                return self._histograms[name]
            if _HAS_PROMETHEUS:
                h = _PromHistogram(name, description, labels,
                                   buckets=buckets, registry=_registry)
            else:
                h = _FallbackHistogram(buckets)
            self._histograms[name] = h
            return h

    # ------------------------------------------------------------------ gauge
    def gauge(self, name: str, description: str = "",
              labels: tuple[str, ...] = ()) -> Any:
        with self._lock:
            if name in self._gauges:
                return self._gauges[name]
            if _HAS_PROMETHEUS:
                g = _PromGauge(name, description, labels,
                               registry=_registry)
            else:
                g = _FallbackCounter()  # gauges fall back to counter
            self._gauges[name] = g
            return g

    # ------------------------------------------------------------------ timer
    @contextlib.contextmanager
    def timer(self, histogram_name: str, *,
              labels: dict[str, str] | None = None):
        """Context manager that observes elapsed seconds into a histogram."""
        h = self._histograms.get(histogram_name)
        if h is None:
            # Auto-declare if a caller picks a name we haven't pre-declared
            h = self.histogram(histogram_name, "auto-declared timer",
                                buckets=_LATENCY_BUCKETS,
                                labels=tuple((labels or {}).keys()))
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            try:
                if _HAS_PROMETHEUS and labels:
                    h.labels(**labels).observe(dt)
                elif _HAS_PROMETHEUS:
                    h.observe(dt)
                else:
                    h.observe(dt, labels=labels)
            except Exception as e:  # noqa: BLE001
                _log.warning(f"metrics timer observe failed for "
                              f"{histogram_name}: {e}")

    # ------------------------------------------------------------------ inc
    def inc(self, counter_name: str, *,
            labels: dict[str, str] | None = None,
            amount: float = 1.0) -> None:
        c = self._counters.get(counter_name)
        if c is None:
            c = self.counter(counter_name, "auto-declared counter",
                              labels=tuple((labels or {}).keys()))
        try:
            if _HAS_PROMETHEUS and labels:
                c.labels(**labels).inc(amount)
            elif _HAS_PROMETHEUS:
                c.inc(amount)
            else:
                c.inc(labels=labels, amount=amount)
        except Exception as e:  # noqa: BLE001
            _log.warning(f"metrics inc failed for {counter_name}: {e}")

    # ------------------------------------------------------------------ observe
    def observe(self, histogram_name: str, value: float, *,
                labels: dict[str, str] | None = None) -> None:
        h = self._histograms.get(histogram_name)
        if h is None:
            return
        try:
            if _HAS_PROMETHEUS and labels:
                h.labels(**labels).observe(value)
            elif _HAS_PROMETHEUS:
                h.observe(value)
            else:
                h.observe(value, labels=labels)
        except Exception as e:  # noqa: BLE001
            _log.warning(f"metrics observe failed for {histogram_name}: {e}")

    # ------------------------------------------------------------------ set
    def set_gauge(self, gauge_name: str, value: float, *,
                  labels: dict[str, str] | None = None) -> None:
        g = self._gauges.get(gauge_name)
        if g is None:
            return
        try:
            if _HAS_PROMETHEUS and labels:
                g.labels(**labels).set(value)
            elif _HAS_PROMETHEUS:
                g.set(value)
            else:
                # FallbackCounter doesn't have set; emulate with inc.
                pass
        except Exception as e:  # noqa: BLE001
            _log.warning(f"metrics set failed for {gauge_name}: {e}")

    # ------------------------------------------------------------------ render
    def prometheus_payload(self) -> tuple[bytes, str]:
        """Return (body, content_type) for a /metrics endpoint."""
        if _HAS_PROMETHEUS:
            return generate_latest(_registry), CONTENT_TYPE_LATEST
        # Fallback: emit a minimal text-format snapshot of in-process counters
        lines = []
        for name, c in self._counters.items():
            if isinstance(c, _FallbackCounter):
                for labels, v in c.snapshot().items():
                    lbl = ",".join(f'{k}="{val}"' for k, val in labels)
                    lines.append(f"{name}{{{lbl}}} {v}")
        return ("\n".join(lines) + "\n").encode("utf-8"), CONTENT_TYPE_LATEST


# Module-level singleton
metrics = Metrics()
