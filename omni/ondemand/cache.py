"""Diskcache wrapper with moderate-TTL tiers for on-demand prediction.

TTL tiers (per user choice on architectural pivot):
    prices         4h
    fundamentals   1d
    news           30min
    macro          24h   (shared across all tickers)
    regime         24h   (shared across all tickers)
    universal_model file-mtime-keyed (in-memory)

Usage:
    from omni.ondemand.cache import cached, TTL

    @cached(ttl=TTL.prices, key="prices:{asset_class}:{ticker}")
    def fetch_prices(asset_class, ticker): ...

The decorator KEY pattern uses str.format with the function kwargs
(positional args mapped by name via inspect). Returns the cached value
on hit; runs and stores on miss. Exceptions are NEVER cached.
"""
from __future__ import annotations

import functools
import inspect
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import diskcache  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError("diskcache missing; pip install diskcache") from e

from omni.utils.paths import hot_dir


@dataclass(frozen=True)
class _TTL:
    prices: int = 4 * 3600
    fundamentals: int = 24 * 3600
    news: int = 30 * 60
    macro: int = 24 * 3600
    regime: int = 24 * 3600
    # Prediction cards expire fast - news is the most volatile input
    # (30min TTL matches the news cache). Without this, an old cached
    # prediction can mask intraday price + sentiment moves for a full
    # day and the dashboard looks frozen.
    deep_analysis: int = 30 * 60


TTL = _TTL()


def _cache_root() -> Path:
    p = hot_dir().parent / "cache" / "ondemand"
    p.mkdir(parents=True, exist_ok=True)
    return p


_CACHE: diskcache.Cache | None = None


def get_cache() -> diskcache.Cache:
    global _CACHE
    if _CACHE is None:
        _CACHE = diskcache.Cache(str(_cache_root()), size_limit=int(2e9))
    return _CACHE


def cached(*, ttl: int, key: str) -> Callable:
    """Decorator: cache function results in diskcache with TTL seconds.

    `key` is a str.format template referencing the function's bound
    arguments by name. e.g. key="prices:{asset_class}:{ticker}".

    Phase 24.16 — adds in-process singleflight: if two threads call this
    decorated function concurrently with the same key, only one runs the
    underlying function; the other waits for and reuses the result.
    Eliminates duplicate yfinance fetches when the portfolio tab fans out
    or when two users hit the same ticker simultaneously.
    """
    def deco(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        # Per-decorated-function inflight registry. We key by the cache
        # key string so unrelated callers never block each other.
        _inflight: dict[str, "_InflightSlot"] = {}
        _inflight_lock = threading.Lock()

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            try:
                k = key.format(**bound.arguments)
            except KeyError:
                k = f"{fn.__module__}.{fn.__name__}:{args}:{sorted(kwargs.items())}"
            cache = get_cache()
            hit = cache.get(k, default=_MISS)
            if hit is not _MISS:
                return hit
            # ---- singleflight: claim or wait on an inflight slot --------
            # Ownership is a thread-local decision, NOT shared state on the
            # slot. The previous code mutated `slot.owner` from the second
            # thread, which could flip the first thread's ownership flag in
            # a race and cause neither thread to compute.
            with _inflight_lock:
                slot = _inflight.get(k)
                if slot is None:
                    slot = _InflightSlot()
                    _inflight[k] = slot
                    is_owner = True
                else:
                    is_owner = False
            if not is_owner:
                # Another thread is already running fn(...) for this key.
                # Wait for it; on completion either re-read the cache or
                # use the result the owner stashed on the slot.
                slot.event.wait(timeout=120)
                if slot.error is not None:
                    raise slot.error
                if slot.value is not _MISS:
                    return slot.value
                hit = cache.get(k, default=_MISS)
                if hit is not _MISS:
                    return hit
                # Owner finished without storing — fall through to compute
            # ---- this thread is the owner: actually compute -------------
            try:
                out = fn(*args, **kwargs)
                slot.value = out
                try:
                    cache.set(k, out, expire=ttl)
                except Exception:
                    pass  # cache write failures are non-fatal
                return out
            except Exception as e:  # noqa: BLE001
                slot.error = e
                raise
            finally:
                slot.event.set()
                with _inflight_lock:
                    _inflight.pop(k, None)

        wrapper.cache_key_template = key  # type: ignore[attr-defined]
        return wrapper
    return deco


class _InflightSlot:
    __slots__ = ("event", "value", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.value: Any = _MISS
        self.error: BaseException | None = None


_MISS = object()


def invalidate(prefix: str) -> int:
    """Invalidate all keys starting with `prefix`. Returns count cleared."""
    cache = get_cache()
    n = 0
    for k in list(cache):
        if isinstance(k, str) and k.startswith(prefix):
            try:
                del cache[k]
                n += 1
            except KeyError:
                pass
    return n


def purge_ticker(ticker: str) -> int:
    """Drop every cache entry that mentions a ticker. Used by the
    'Force refresh' button on the Predict tab so the next call rebuilds
    everything (prices + news + fundamentals + prediction card) from
    live APIs instead of returning the stale cached prediction.
    """
    if not ticker:
        return 0
    needle = f":{ticker.upper()}"
    needle_lower = f":{ticker.lower()}"
    cache = get_cache()
    n = 0
    for k in list(cache):
        if not isinstance(k, str):
            continue
        # Match either ":TICKER" segment (most keys) or ":TICKER:" mid-key
        if (needle in k or needle_lower in k
                or k.endswith(f":{ticker.upper()}")
                or k.endswith(f":{ticker.lower()}")):
            try:
                del cache[k]
                n += 1
            except KeyError:
                pass
    return n
