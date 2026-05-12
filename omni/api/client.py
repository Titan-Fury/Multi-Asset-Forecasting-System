"""HTTP client for the Omni_Stocks prediction API.

This is the bridge between the Streamlit UI and the FastAPI service.
The dashboard calls `predict_via_api(...)` instead of importing
`predict_ticker` directly, which means:

  - the UI process stays light (no model loading)
  - the API process owns the heavy lifting and can be scaled separately
  - if the API is down, we transparently fall back to in-process calls
    so local dev keeps working without spinning up a server

Phase 24.16 optimizations:
  - reuse a single `requests.Session` so TCP / TLS connections pool
  - cache the /health probe result for 5s so we don't pay an extra
    round-trip on every predict call
  - graceful fallback timing — short timeout on the probe, longer on
    the real call

Environment toggle:
    OMNI_API_URL              (default: http://127.0.0.1:8000)
    OMNI_API_ENABLED          (default: 1; set to 0 to force in-process)
    OMNI_API_TIMEOUT_SECONDS  (default: 30)
    OMNI_API_PROBE_TTL_SECONDS  (default: 5)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

_log = logging.getLogger("omni.api.client")

_API_URL = os.environ.get("OMNI_API_URL", "http://127.0.0.1:8000").rstrip("/")
_API_ENABLED = os.environ.get("OMNI_API_ENABLED", "1") == "1"
_TIMEOUT = float(os.environ.get("OMNI_API_TIMEOUT_SECONDS", "30"))
_PROBE_TTL = float(os.environ.get("OMNI_API_PROBE_TTL_SECONDS", "5"))

# requests is already a transitive dep via yfinance; no new install needed.
try:
    import requests
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except ImportError:
        Retry = None  # type: ignore
    _HAS_REQUESTS = True
except ImportError:  # pragma: no cover
    _HAS_REQUESTS = False


# --------------------------------------------------------------------------
# Sessions — one per process. We keep TWO: one with retries for real
# predict POSTs (transient 5xx → backoff), and one WITHOUT retries for
# the /health probe (we don't want to spend 6+ seconds retrying a probe
# of a server that's clearly down).
# --------------------------------------------------------------------------
_session: "requests.Session | None" = None
_probe_session: "requests.Session | None" = None
_session_lock = threading.Lock()


def _get_session() -> "requests.Session":
    """Predict-call session: pooled + retries on transient 5xx."""
    global _session
    if _session is not None:
        return _session
    with _session_lock:
        if _session is not None:
            return _session
        s = requests.Session()
        if Retry is not None:
            retry = Retry(
                total=2,
                backoff_factor=0.3,
                status_forcelist=(502, 503, 504),
                allowed_methods=frozenset(["GET", "POST"]),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry,
                                   pool_connections=8,
                                   pool_maxsize=16)
            s.mount("http://", adapter)
            s.mount("https://", adapter)
        _session = s
        return _session


def _get_probe_session() -> "requests.Session":
    """Health-probe session: pooled but NO retries. Probes fail fast so
    the dashboard doesn't spend 6+ seconds discovering the API is down."""
    global _probe_session
    if _probe_session is not None:
        return _probe_session
    with _session_lock:
        if _probe_session is not None:
            return _probe_session
        s = requests.Session()
        # Plain adapter with pooling but no retry policy
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _probe_session = s
        return _probe_session


# --------------------------------------------------------------------------
# Health probe — cached so back-to-back predicts don't each hit /health.
# Phase 24.16 fix: timestamp is recorded AFTER the probe completes (not
# at the start), so a slow probe doesn't get its result thrown away
# immediately by the next caller.
# --------------------------------------------------------------------------
_probe_lock = threading.Lock()
_probe_cache: dict[str, float | bool] = {"checked_at": 0.0, "available": False}


def _api_available() -> bool:
    """TTL-cached probe. Returns True if /health is reachable now (or was
    in the last `_PROBE_TTL` seconds)."""
    if not (_API_ENABLED and _HAS_REQUESTS):
        return False
    with _probe_lock:
        age = time.monotonic() - float(_probe_cache["checked_at"])
        if age < _PROBE_TTL:
            return bool(_probe_cache["available"])
    # Outside lock: actually probe (fail-fast, no retries)
    try:
        r = _get_probe_session().get(f"{_API_URL}/health", timeout=1.0)
        ok = r.status_code == 200
    except Exception:  # noqa: BLE001
        # Connection refused, DNS, timeout — all fail-fast at this layer.
        # The retry-enabled session is reserved for real predict calls.
        ok = False
    with _probe_lock:
        # Record the timestamp AFTER the probe completes, so a slow
        # network round-trip doesn't immediately invalidate its own
        # cached result.
        _probe_cache["checked_at"] = time.monotonic()
        _probe_cache["available"] = ok
    return ok


def predict_via_api(ticker: str, asset_class: str,
                    *, display_name: str | None = None,
                    deep: bool = False) -> dict[str, Any] | None:
    """Call /predict (or /predict/deep) and return the card dict.

    Falls back to in-process `predict_ticker` / `deep_analysis` if the
    API isn't reachable, so local dev keeps working with `streamlit run`
    alone (no separate uvicorn process required).
    """
    if _api_available():
        endpoint = "/predict/deep" if deep else "/predict"
        try:
            r = _get_session().post(
                f"{_API_URL}{endpoint}",
                json={"ticker": ticker, "asset_class": asset_class,
                      "display_name": display_name},
                timeout=_TIMEOUT,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code == 422:
                # Insufficient data — same as None in the in-process API
                _log.info(f"API 422 for {ticker}: {r.json().get('detail', '')}")
                return None
            _log.warning(f"API {r.status_code} for {ticker}: "
                          f"{r.text[:200]}; falling back in-process")
            # Treat 5xx as a probe failure so we don't keep retrying it
            with _probe_lock:
                _probe_cache["available"] = False
                _probe_cache["checked_at"] = time.monotonic()
        except Exception as e:  # noqa: BLE001
            _log.warning(f"API call failed ({e}); falling back in-process")
            with _probe_lock:
                _probe_cache["available"] = False
                _probe_cache["checked_at"] = time.monotonic()
    # ---- in-process fallback ------------------------------------------------
    if deep:
        from omni.ondemand.predict import deep_analysis
        card = deep_analysis(ticker, asset_class, display_name=display_name)
    else:
        from omni.ondemand.predict import predict_ticker
        card = predict_ticker(ticker, asset_class, display_name=display_name)
    return card.to_dict() if card else None


def health() -> dict[str, Any] | None:
    """Return the /health response or None if API is down."""
    if not _api_available():
        return None
    try:
        return _get_session().get(f"{_API_URL}/health", timeout=2.0).json()
    except Exception:  # noqa: BLE001
        return None
