"""OpenRouter backend (OpenAI-compatible /chat/completions).

Free models are heavily rate-limited; the router pairs this with the
token-bucket limiter declared in llm_routes.yaml.
"""
from __future__ import annotations

import time
from typing import Any

import requests

from omni.llm.backends.base import Backend, BackendError, LLMResult, http_session
from omni.utils.env import get_env

_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterBackend(Backend):
    def _api_key(self) -> str:
        env_name = self.config.get("api_key_env", "OPENROUTER_API_KEY")
        key = get_env(env_name)
        if not key:
            raise BackendError(f"{env_name} not set")
        return key

    def complete(self, prompt: str, **kwargs: Any) -> LLMResult:
        model = self.config["model"]
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(kwargs.get("temperature", 0.2)),
            "max_tokens": int(kwargs.get("max_tokens", 1024)),
        }
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/omni-stocks",
            "X-Title": "Omni_Stocks",
        }
        t0 = time.monotonic()
        try:
            # Phase 24.21 — use pooled session to skip per-call TLS handshake.
            r = http_session().post(_BASE_URL, json=body, headers=headers, timeout=60)
        except Exception as e:
            raise BackendError(f"openrouter request failed: {e}") from e
        if r.status_code == 429:
            raise BackendError(f"openrouter rate-limited: {r.text[:200]}")
        if r.status_code != 200:
            raise BackendError(f"openrouter {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise BackendError(f"openrouter malformed response: {data}") from e
        usage = data.get("usage", {}) or {}
        return LLMResult(
            backend=self.name,
            text=text,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.monotonic() - t0) * 1000,
            extra={"model": model, "id": data.get("id")},
        )
