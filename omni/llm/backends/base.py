"""Backend abstract base + shared types.

All backends implement `Backend.complete(prompt, **kw) -> LLMResult`.
The router calls them through this uniform interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


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
