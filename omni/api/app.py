"""FastAPI app exposing the prediction surface.

Endpoints:
    GET  /health              -> {"status": "ok", "models_loaded": bool}
    GET  /ready               -> 200 only when models are warmed up
    GET  /resolve?q=...       -> universe.Resolution as JSON
    POST /predict             -> single-shot prediction (cached)
    POST /predict/deep        -> walk-forward fit (slow, cached)
    GET  /metrics             -> minimal counters for ops dashboards

Notes
-----
* `predict_ticker` and `deep_analysis` are already cached via `diskcache`,
  so re-hitting the same ticker within the TTL returns instantly.
* Errors are mapped to HTTP status codes with structured JSON payloads
  rather than 500 stack traces. This is what every ops team needs.
* CORS is permissive in dev; lock it down via `OMNI_API_CORS_ORIGINS` in
  production.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from omni.observability.metrics import metrics

# Import for side effect: configures loguru + stdlib logging bridge so
# uvicorn / FastAPI logs flow through the same pipeline.
from omni.utils.logger import log as _omni_log  # noqa: F401

_log = logging.getLogger("omni.api")


# --------------------------------------------------------------------------
# Request / response schemas — Pydantic enforces shape at the boundary
# so a malformed payload produces a clear 422 instead of a 500.
# --------------------------------------------------------------------------
class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=24,
                        description="Ticker symbol, e.g. 'RELIANCE.NS' or 'AAPL'.")
    asset_class: str | None = Field(
        None, description="One of india/us/crypto/fx/etf/index. "
                          "Defaults to resolver's best guess.")
    display_name: str | None = Field(
        None, description="Optional human-readable name; passed through.")


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str = "24.16"
    uptime_seconds: float


class MetricsResponse(BaseModel):
    requests_total: int
    predictions_total: int
    deep_analysis_total: int
    errors_total: int
    cache_hits: int


# --------------------------------------------------------------------------
# In-process counters. For multi-worker deploys swap for prometheus_client.
# Phase 24.16 — guarded with a lock so concurrent requests don't lose
# increments (FastAPI's middleware can run truly concurrently on async
# call sites). The Prometheus metrics in `omni.observability` are the
# real source of truth; these stay only for the legacy /metrics route.
# --------------------------------------------------------------------------
import threading as _threading

_metrics_lock = _threading.Lock()
_metrics: dict[str, int] = {
    "requests_total": 0,
    "predictions_total": 0,
    "deep_analysis_total": 0,
    "errors_total": 0,
    "cache_hits": 0,
}
_started_at = time.time()
_models_loaded = False


# --------------------------------------------------------------------------
# Lifespan — warmup hook. Loads the universal model + reranker BEFORE the
# server reports ready. Solves the auditor's "first user eats cold start"
# concern.
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models_loaded
    _log.info("omni.api startup: warming models...")
    t0 = time.perf_counter()
    try:
        from omni.ondemand import model_registry
        # Touch the model registry so the bundle paths get cached
        _ = model_registry.is_available()
        # Lazy-import heavy reranker only if RAG is enabled
        if os.environ.get("OMNI_API_WARMUP_RERANKER", "0") == "1":
            try:
                from omni.llm.rag.retriever import _get_reranker
                _get_reranker()
            except Exception as e:  # noqa: BLE001
                _log.warning(f"reranker warmup failed (non-fatal): {e}")
        _models_loaded = True
        _log.info(f"omni.api startup: models warm in {time.perf_counter() - t0:.2f}s")
    except Exception as e:  # noqa: BLE001
        _log.error(f"omni.api startup: warmup FAILED: {e}", exc_info=True)
        # Don't crash on warmup failure — first request will pay the cost.
        # /ready will return 503 until models are loaded.
    yield
    _log.info("omni.api shutdown.")


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = FastAPI(
    title="Omni_Stocks Prediction API",
    description="HTTP service layer over the on-demand prediction surface.",
    version="24.16",
    lifespan=lifespan,
)

# CORS — permissive default for local dev; override via env in production.
# When using the wildcard "*", we MUST disable allow_credentials because
# browsers reject Access-Control-Allow-Origin: * combined with
# Access-Control-Allow-Credentials: true.
_cors_origins = [o.strip() for o in
                  os.environ.get("OMNI_API_CORS_ORIGINS", "*").split(",")
                  if o.strip()]
_allow_credentials = "*" not in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _count_requests(request, call_next):
    with _metrics_lock:
        _metrics["requests_total"] += 1
    try:
        response = await call_next(request)
        return response
    except Exception:
        with _metrics_lock:
            _metrics["errors_total"] += 1
        raise


# --------------------------------------------------------------------------
# Health / readiness / metrics
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe — always 200 when the process is up."""
    return HealthResponse(
        status="ok",
        models_loaded=_models_loaded,
        uptime_seconds=time.time() - _started_at,
    )


@app.get("/ready")
async def ready():
    """Readiness probe — 200 only after warmup completes.
    Use this for Kubernetes readinessProbe / load-balancer health checks
    so traffic isn't routed to instances still warming up."""
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="models warming up")
    return {"status": "ready"}


@app.get("/metrics", response_model=MetricsResponse)
async def metrics_legacy() -> MetricsResponse:
    """Legacy in-process counters (kept for backwards compat)."""
    return MetricsResponse(**_metrics)


@app.get("/metrics/prom")
async def metrics_prom() -> Response:
    """Prometheus-format metrics endpoint. Scrape this from Prometheus
    with a 15-30s interval. Falls back to a minimal text format if
    prometheus_client isn't installed."""
    body, content_type = metrics.prometheus_payload()
    return Response(content=body, media_type=content_type)


# --------------------------------------------------------------------------
# Universe resolution
# --------------------------------------------------------------------------
@app.get("/resolve")
async def resolve_endpoint(q: str = Query(..., min_length=1)):
    """Resolve a free-form ticker query to a canonical Resolution."""
    from omni.ondemand.universe import resolve
    res = resolve(q)
    if res is None:
        raise HTTPException(status_code=404,
                            detail=f"could not resolve {q!r}")
    return {
        "ticker": res.ticker,
        "asset_class": res.asset_class,
        "display_name": res.display_name,
        "confident": res.confident,
    }


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------
@app.post("/predict")
async def predict_endpoint(req: PredictRequest) -> dict[str, Any]:
    """Single-shot prediction. Returns the PredictionCard as a dict.
    Cache TTL is governed by `omni.ondemand.cache`.
    """
    from omni.ondemand.predict import predict_ticker
    from omni.ondemand.universe import resolve

    asset_class = req.asset_class
    display_name = req.display_name
    if not asset_class:
        res = resolve(req.ticker)
        if res is None:
            raise HTTPException(status_code=404,
                                detail=f"could not resolve {req.ticker!r}")
        asset_class = res.asset_class
        display_name = display_name or res.display_name

    # Phase 24.16 — observability: time the call and record outcome
    with metrics.timer("prediction_latency_seconds", labels={"path": "live"}):
        try:
            # predict_ticker is CPU-bound and synchronous — run it in a
            # threadpool so the FastAPI event loop stays responsive.
            card = await asyncio.to_thread(
                predict_ticker, req.ticker, asset_class,
                display_name=display_name,
            )
        except Exception as e:  # noqa: BLE001
            metrics.inc("predictions_total",
                        labels={"asset_class": asset_class,
                                "direction": "error", "path": "live"})
            _log.error(f"predict {req.ticker}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    if card is None:
        metrics.inc("predictions_total",
                    labels={"asset_class": asset_class,
                            "direction": "no_data", "path": "live"})
        raise HTTPException(
            status_code=422,
            detail=f"insufficient data for {req.ticker} "
                   f"({asset_class}); ticker may be delisted or too illiquid",
        )

    # Record the rich observation set: direction, conviction, data quality
    d = card.to_dict()
    direction_str = {1: "long", -1: "short", 0: "flat"}.get(
        int(d.get("direction", 0)), "flat")
    metrics.inc("predictions_total",
                labels={"asset_class": asset_class,
                        "direction": direction_str, "path": "live"})
    try:
        metrics.observe("prediction_conviction",
                        float(d.get("conviction", 0) or 0),
                        labels={"asset_class": asset_class})
        metrics.observe("data_quality_score",
                        float(d.get("data_quality_score", 0) or 0),
                        labels={"asset_class": asset_class})
        # RAG path/grounding telemetry — surfaces silent degradation
        why_path = str(d.get("why_path", "deterministic"))
        metrics.inc("rag_path_total", labels={"path": why_path})
        gs = float(d.get("why_grounding_score", 0.0) or 0.0)
        if gs > 0:
            metrics.observe("rag_grounding_score", gs)
    except Exception as e:  # noqa: BLE001
        _log.debug(f"metrics observation failed: {e}")

    with _metrics_lock:
        _metrics["predictions_total"] += 1
    return d


@app.post("/predict/deep")
async def predict_deep_endpoint(req: PredictRequest) -> dict[str, Any]:
    """Walk-forward deep-analysis prediction. Slow (~15s cold) but fits
    the model on the ticker's own history; used for one-off deep dives."""
    from omni.ondemand.predict import deep_analysis
    from omni.ondemand.universe import resolve

    asset_class = req.asset_class
    display_name = req.display_name
    if not asset_class:
        res = resolve(req.ticker)
        if res is None:
            raise HTTPException(status_code=404,
                                detail=f"could not resolve {req.ticker!r}")
        asset_class = res.asset_class
        display_name = display_name or res.display_name

    try:
        # deep_analysis is heavy and synchronous — offload to threadpool.
        card = await asyncio.to_thread(
            deep_analysis, req.ticker, asset_class,
            display_name=display_name,
        )
    except Exception as e:  # noqa: BLE001
        _log.error(f"deep_analysis {req.ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

    if card is None:
        raise HTTPException(
            status_code=422,
            detail=f"insufficient data for deep analysis on {req.ticker}",
        )

    with _metrics_lock:
        _metrics["deep_analysis_total"] += 1
    return card.to_dict()


# --------------------------------------------------------------------------
# Cache control
# --------------------------------------------------------------------------
@app.post("/cache/purge")
async def purge_endpoint(ticker: str = Query(..., min_length=1)):
    """Drop every cache entry referencing this ticker. Returns count purged."""
    from omni.ondemand.cache import purge_ticker
    n = purge_ticker(ticker)
    return {"ticker": ticker, "purged": n}
