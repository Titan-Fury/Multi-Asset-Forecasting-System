"""Embedder wrapper with disk + LRU cache around Nemotron 1B.

Nemotron via OpenRouter free tier has tight rate limits (~20 rpm / 200 rpd),
so caching is critical. We cache at two levels:

    1. In-process LRU (last 4096 texts)         — sub-ms hits during ingest
    2. Persistent diskcache via LLMCache        — survives restarts

Failed embeddings (zero vectors) are NEVER cached, so transient failures
don't poison the cache.
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Iterable

import numpy as np

from omni.llm.cache import LLMCache
from omni.llm.embed_nemotron import EMBED_DIM, embed as _nemotron_embed, is_zero

_log = logging.getLogger("omni.llm.rag.embedder")

_BATCH_SIZE = 96  # Nemotron limit
_CACHE_BACKEND = "rag_embed"
# Phase 24.15 — bumped to v3 because Nemotron model output dim changed
# from 1024 to 2048; old cached vectors are no longer valid.
_CACHE_TASK = "nemotron_v3"
_CACHE_TTL_DAYS = 90  # embeddings rarely go stale


def _text_key(text: str) -> str:
    """Stable cache key. We hash both the model id and the text so changing
    the embedder version won't return stale vectors."""
    h = hashlib.sha1((_CACHE_TASK + "::" + text).encode("utf-8")).hexdigest()
    return h


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

class Embedder:
    """Cached embedder with in-process + disk persistence."""

    def __init__(self) -> None:
        self._cache = LLMCache.get()
        # Phase 24.21 — was a (dict + list) pair where eviction did
        # `self._mem_order.pop(0)`, an O(n) shift over up to 4096 keys on
        # every insert past the cap. OrderedDict.popitem(last=False) is O(1)
        # FIFO eviction and removes the parallel-list-of-keys hazard.
        self._mem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._mem_cap = 4096

    def _mem_get(self, key: str) -> np.ndarray | None:
        return self._mem.get(key)

    def _mem_put(self, key: str, vec: np.ndarray) -> None:
        if key in self._mem:
            return
        self._mem[key] = vec
        while len(self._mem) > self._mem_cap:
            self._mem.popitem(last=False)  # O(1) FIFO eviction

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns (1024,) float32. Uses cache."""
        if not text or not text.strip():
            return np.zeros(EMBED_DIM, dtype=np.float32)
        return self.embed_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        """Embed a batch. Returns (N, 1024) float32.

        Hits in-process LRU first, then disk cache, then makes a single
        Nemotron API call for the remaining miss set (chunked to 96/batch).
        Successful embeddings populate both caches.
        """
        items = [t if t is not None else "" for t in texts]
        n = len(items)
        if n == 0:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        out = np.zeros((n, EMBED_DIM), dtype=np.float32)
        miss_idx: list[int] = []
        miss_keys: list[str] = []
        keys: list[str] = [_text_key(t) for t in items]

        # Stage 1: in-memory hits
        for i, k in enumerate(keys):
            v = self._mem_get(k)
            if v is not None:
                out[i] = v

        # Stage 2: disk-cache hits
        for i, k in enumerate(keys):
            if not is_zero(out[i]):
                continue
            disk = self._cache.lookup(_CACHE_BACKEND, _CACHE_TASK, k, None)
            if disk is not None:
                try:
                    # Phase 24.21 — prefer raw-bytes payload (5x smaller on
                    # disk + much faster ser/de) but fall back to legacy
                    # tolist() payload for caches written before this change.
                    v: np.ndarray | None = None
                    if "vec_bytes" in disk:
                        buf = disk["vec_bytes"]
                        if isinstance(buf, (bytes, bytearray, memoryview)):
                            v = np.frombuffer(bytes(buf), dtype=np.float32)
                    elif "vec" in disk:
                        v = np.array(disk["vec"], dtype=np.float32)
                    if v is not None and v.shape == (EMBED_DIM,) and not is_zero(v):
                        out[i] = v
                        self._mem_put(k, v)
                        continue
                except Exception:  # noqa: BLE001
                    # Disk cache hit shape/format mismatch — recompute below.
                    # Common after embedder version bump.
                    pass
            miss_idx.append(i)
            miss_keys.append(k)

        if not miss_idx:
            return out

        miss_texts = [items[i] for i in miss_idx]
        # Stage 3: API call (auto-batched)
        for batch_start in range(0, len(miss_texts), _BATCH_SIZE):
            batch = miss_texts[batch_start : batch_start + _BATCH_SIZE]
            try:
                vecs = _nemotron_embed(batch)  # (B, 1024) normalised
            except Exception as e:
                _log.warning("nemotron embed call failed: %s", e)
                # Phase 24.16 — surface the silent failure to the
                # degradation monitor so ops can see embedder outages
                # instead of just observing worse explanations downstream.
                try:
                    from omni.llm.rag.degradation import monitor
                    monitor.record(
                        component="embedder",
                        severity="error",
                        event="api_call_failed",
                        details={
                            "error_type": type(e).__name__,
                            "error_message": str(e)[:300],
                            "batch_size": len(batch),
                            "endpoint": "nemotron",
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
                continue
            if vecs.shape[0] != len(batch):
                _log.warning("nemotron returned %d vecs for %d inputs", vecs.shape[0], len(batch))
                try:
                    from omni.llm.rag.degradation import monitor
                    monitor.record(
                        component="embedder",
                        severity="warn",
                        event="batch_size_mismatch",
                        details={"requested": len(batch),
                                 "received": int(vecs.shape[0])},
                    )
                except Exception:  # noqa: BLE001
                    pass
                continue
            for j, vec in enumerate(vecs):
                global_i = miss_idx[batch_start + j]
                global_k = miss_keys[batch_start + j]
                if is_zero(vec):
                    continue  # never cache failures
                out[global_i] = vec
                self._mem_put(global_k, vec)
                # Disk-cache the success
                try:
                    # Phase 24.21 — store raw float32 bytes instead of a
                    # 2048-element JSON list. ~5x smaller on disk and removes
                    # both ser/deser CPU cost on cache hits/misses.
                    self._cache.store(
                        _CACHE_BACKEND, _CACHE_TASK, global_k,
                        {"vec_bytes": vec.astype(np.float32).tobytes()},
                        ttl_days=_CACHE_TTL_DAYS,
                        params=None,
                    )
                except Exception as e:
                    _log.debug("disk cache store failed: %s", e)
        return out


_DEFAULT: Embedder | None = None


def get_embedder() -> Embedder:
    """Process-singleton accessor."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Embedder()
    return _DEFAULT
