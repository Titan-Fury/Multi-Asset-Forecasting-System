"""Backend abstract base + shared types.

All backends implement `Backend.complete(prompt, **kw) -> LLMResult`.
The router calls them through this uniform interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


# Phase 24.21 — shared module-level requests.Session with connection pooling.
# Was a bare `requests.post()` per call across 5 backends (openrouter, groq,
# gemini, ollama, embed_nemotron) which paid a fresh TCP+TLS handshake every
# time, ~100-300ms of pure overhead per LLM call. One pooled Session saves
# that on every retry/fallback in the router chain. Lazy so importing
# `Backend` doesn't pull in urllib3's adapter machinery if no HTTP backend
# is ever used (e.g. FinBERT-only runs).
_http_session = None


def http_session():
    """Return a process-wide pooled requests.Session.

    Backends share one session so the underlying urllib3 connection pool is
    reused across them — the typical hot path is a chain of openrouter ->
    groq -> gemini calls all going to different hosts but with similar
    handshake costs amortised by keep-alive on each.
    """
    global _http_session
    if _http_session is not None:
        return _http_session
    import requests
    from requests.adapters import HTTPAdapter
    sess = requests.Session()
    adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=0)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    _http_session = sess
    return sess


@dataclass
class LLMResult:
    backend: str
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LLMResult":
        return cls(**d)


class BackendError(RuntimeError):
    """Raised by a backend when the primary call fails. Triggers fallback."""


class Backend(ABC):
    name: str

    def __init__(self, name: str, config: dict) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResult:
        ...

    def health(self) -> bool:
        """Cheap probe — override in subclasses where applicable."""
        return True
