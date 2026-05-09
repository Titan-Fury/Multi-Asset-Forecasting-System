"""Multi-backend LLM router.

Reads `config/llm_routes.yaml` and dispatches each task to its
primary -> fallback chain. Adds:
  - per-backend token-bucket rate limiting
  - diskcache lookups keyed by (backend, task, prompt, params)
  - structured LLMResult return values

Public API: `route(task, prompt, **params) -> LLMResult`.
"""
from __future__ import annotations

from typing import Any

from omni.llm.backends.base import Backend, BackendError, LLMResult
from omni.llm.cache import LLMCache
from omni.llm.ratelimit import get_limiter
from omni.utils.env import llm_routes
from omni.utils.logger import log

_BACKEND_TYPES: dict[str, type[Backend]] = {}


def _backend_factory():
    """Lazy-import backend classes (avoids torch import unless FinBERT used)."""
    if all(k in _BACKEND_TYPES for k in ("ollama", "openrouter", "cortex", "groq", "gemini")):
        return _BACKEND_TYPES
    from omni.llm.backends.cortex import CortexBackend
    from omni.llm.backends.gemini import GeminiBackend
    from omni.llm.backends.groq import GroqBackend
    from omni.llm.backends.ollama import OllamaBackend
    from omni.llm.backends.openrouter import OpenRouterBackend
    _BACKEND_TYPES.setdefault("ollama", OllamaBackend)
    _BACKEND_TYPES.setdefault("openrouter", OpenRouterBackend)
    _BACKEND_TYPES.setdefault("cortex", CortexBackend)
    _BACKEND_TYPES.setdefault("groq", GroqBackend)
    _BACKEND_TYPES.setdefault("gemini", GeminiBackend)
    # FinBERT only loaded on demand (heavy)
    return _BACKEND_TYPES


def _get_finbert():
    if "finbert" not in _BACKEND_TYPES:
        from omni.llm.backends.finbert import FinBERTBackend
        _BACKEND_TYPES["finbert"] = FinBERTBackend
    return _BACKEND_TYPES["finbert"]


class Router:
    _instance: "Router | None" = None

    def __init__(self) -> None:
        self.cfg = llm_routes()
        self._backends: dict[str, Backend] = {}
        self._cache = LLMCache.get()

    @classmethod
    def get(cls) -> "Router":
        if cls._instance is None:
            cls._instance = Router()
        return cls._instance

    def _instantiate(self, name: str) -> Backend:
        if name in self._backends:
            return self._backends[name]
        spec = self.cfg["backends"].get(name)
        if not spec:
            raise KeyError(f"backend '{name}' not in llm_routes.yaml")
        if not spec.get("enabled", True):
            raise BackendError(f"backend '{name}' disabled")
        btype = spec["type"]
        factory = _backend_factory()
        if btype == "finbert":
            cls = _get_finbert()
        else:
            if btype not in factory:
                raise KeyError(f"unknown backend type: {btype}")
            cls = factory[btype]
        be = cls(name, spec)
        self._backends[name] = be
        return be

    def _limiter_for(self, name: str):
        spec = self.cfg["backends"][name]
        rl = spec.get("rate_limit") or {}
        return get_limiter(name, rpm=rl.get("rpm"), rpd=rl.get("rpd"))

    def route(self, task: str, prompt: str, *, use_cache: bool = True, **params: Any) -> LLMResult:
        if task not in self.cfg["tasks"]:
            raise KeyError(f"task '{task}' not configured in llm_routes.yaml")
        chain = list(self.cfg["tasks"][task]["primary"]) + list(self.cfg["tasks"][task].get("fallback") or [])
        ttl_days = self.cfg["tasks"][task].get("cache_ttl_days")
        last_err: Exception | None = None

        for backend_name in chain:
            # Cache lookup (per backend so different backends cache independently)
            if use_cache:
                hit = self._cache.lookup(backend_name, task, prompt, params)
                if hit is not None:
                    log.debug(f"[{task}] cache hit on {backend_name}")
                    return LLMResult.from_dict(hit)
            try:
                be = self._instantiate(backend_name)
                self._limiter_for(backend_name).acquire(block=True)
                result = be.complete(prompt, **params)
                if use_cache and ttl_days:
                    self._cache.store(backend_name, task, prompt, result.to_dict(),
                                      ttl_days=ttl_days, params=params)
                log.debug(f"[{task}] succeeded on {backend_name} ({result.latency_ms:.0f}ms)")
                return result
            except (BackendError, KeyError) as e:
                log.warning(f"[{task}] backend '{backend_name}' failed: {e}; trying next")
                last_err = e
                continue

        raise BackendError(f"all backends exhausted for task '{task}'; last error: {last_err}")


def route(task: str, prompt: str, **kwargs: Any) -> LLMResult:
    """Module-level convenience wrapper."""
    return Router.get().route(task, prompt, **kwargs)
