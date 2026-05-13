"""Smoke test for the cache singleflight added in Phase 24.16.

Spawns N concurrent threads that all call the same cached function. The
underlying function should be invoked exactly ONCE; the other N-1 callers
should receive the same return value via the inflight slot.

Run:
    python scripts/test_singleflight.py
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from omni.ondemand.cache import cached, invalidate


def main() -> int:
    # Fresh cache slot for the test
    invalidate("singleflight_test:")

    call_count = 0
    count_lock = threading.Lock()

    @cached(ttl=60, key="singleflight_test:{x}")
    def slow_compute(x: int) -> int:
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.5)  # simulate slow work
        return x * 100

    # 8 concurrent threads, all asking for the same x. Should produce
    # exactly 1 underlying call.
    N = 8
    with ThreadPoolExecutor(max_workers=N) as ex:
        futures = [ex.submit(slow_compute, 7) for _ in range(N)]
        results = [f.result() for f in futures]

    print(f"call_count: {call_count} (expected 1)")
    print(f"results: {results}")
    print(f"all match: {all(r == 700 for r in results)}")

    if call_count != 1:
        print("FAIL: singleflight did not coalesce duplicate calls")
        return 1
    if not all(r == 700 for r in results):
        print("FAIL: not all callers got the same result")
        return 1

    # Now verify cached path still works (subsequent calls = 0 new computes)
    invalidate("singleflight_test:")
    call_count = 0
    slow_compute(99)  # populate
    slow_compute(99)  # cache hit
    slow_compute(99)  # cache hit
    print(f"after cache populate: call_count={call_count} (expected 1)")
    if call_count != 1:
        print("FAIL: cache hits not honoured")
        return 1

    # Different keys should NOT block each other
    call_count = 0
    invalidate("singleflight_test:")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(slow_compute, i) for i in range(4)]
        results = [f.result() for f in futures]
    print(f"4 different keys: call_count={call_count} (expected 4)")
    if call_count != 4:
        print("FAIL: different keys were incorrectly coalesced")
        return 1

    invalidate("singleflight_test:")
    print("OK: singleflight, cache hits, and per-key isolation all work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
