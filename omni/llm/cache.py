"""Disk-backed LLM response cache (diskcache).

Cache key = sha256(backend_name || task || prompt || params_json).
Stored value = LLMResult.to_dict(). TTL configurable per task.
Hard size cap from llm_routes.yaml -> cache.size_limit_gb.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from omni.utils.env import llm_routes
from omni.utils.logger import log
from omni.utils.paths import llm_cache_dir


def _make_key(backend: str, task: str, prompt: str, params: dict[str, Any] | None) -> str:
    payload = json.dumps(
        {"b": backend, "t": task, "p": prompt, "k": params or {}},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCache:
    """Lazy-init diskcache wrapper. Single instance per process."""

    _instance: LLMCache | None = None
    _lock = __import__("threading").Lock()

    def __init__(self) -> None:
        import diskcache
        cfg = llm_routes().get("cache", {})
        size_gb = float(cfg.get("size_limit_gb", 5))
        path = llm_cache_dir()
        path.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(
            str(path), size_limit=int(size_gb * 1024**3), eviction_policy="least-recently-used"
        )
        log.debug(f"LLMCache initialized at {path} (limit {size_gb} GB)")

    @classmethod
    def get(cls) -> LLMCache:
        # Double-checked locking: avoid lock cost on the common hot path
        # while still guaranteeing only one instance is constructed even
        # under concurrent dashboard requests.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = LLMCache()
        return cls._instance

    def lookup(self, backend: str, task: str, prompt: str, params: dict | None = None) -> dict | None:
        return self._cache.get(_make_key(backend, task, prompt, params))

    def store(self, backend: str, task: str, prompt: str, value: dict,
              *, ttl_days: int | None = None, params: dict | None = None) -> None:
        ttl = None if ttl_days is None else ttl_days * 86_400
        self._cache.set(_make_key(backend, task, prompt, params), value, expire=ttl)

    def stats(self) -> dict:
        return {
            "size_bytes": self._cache.volume(),
            "count": len(self._cache),
        }

    def clear(self) -> None:
        self._cache.clear()
