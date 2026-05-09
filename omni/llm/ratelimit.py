"""Token-bucket rate limiters keyed by backend name.

Per `llm_routes.yaml`, some backends declare:
    rate_limit:
      rpm: 18
      rpd: 180

Two buckets are enforced (whichever is tighter wins). Thread-safe.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    capacity: float
    refill_per_sec: float
    tokens: float
    last_refill: float

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
        self.last_refill = now

    def try_acquire(self, n: float = 1.0) -> bool:
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def time_to_available(self, n: float = 1.0) -> float:
        self._refill()
        if self.tokens >= n:
            return 0.0
        return (n - self.tokens) / self.refill_per_sec


class RateLimiter:
    """Composite rate limiter: enforces both rpm and rpd if configured."""

    def __init__(self, rpm: int | None = None, rpd: int | None = None) -> None:
        self._lock = threading.Lock()
        now = time.monotonic()
        self._buckets: list[_Bucket] = []
        if rpm:
            self._buckets.append(_Bucket(rpm, rpm / 60.0, rpm, now))
        if rpd:
            self._buckets.append(_Bucket(rpd, rpd / 86_400.0, rpd, now))

    def acquire(self, *, block: bool = True, timeout: float | None = None) -> bool:
        """Try to take one token from each bucket. If block=True, wait until available."""
        if not self._buckets:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                if all(b.try_acquire() for b in self._buckets):
                    return True
                wait = max(b.time_to_available() for b in self._buckets)
            if not block:
                return False
            if deadline is not None and time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 5.0))


# Module-level registry: one limiter per backend name
_REGISTRY: dict[str, RateLimiter] = {}
_REG_LOCK = threading.Lock()


def get_limiter(name: str, *, rpm: int | None = None, rpd: int | None = None) -> RateLimiter:
    with _REG_LOCK:
        if name not in _REGISTRY:
            _REGISTRY[name] = RateLimiter(rpm=rpm, rpd=rpd)
        return _REGISTRY[name]
