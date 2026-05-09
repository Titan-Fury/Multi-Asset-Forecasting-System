"""Groq backend (OpenAI-compatible /chat/completions).

Groq's free tier offers very fast Llama / Mixtral inference. Same shape as
OpenRouter so the router can hot-swap them. Rate limits per Groq's free
tier (currently 30 rpm / 14400 rpd for most free models) should be declared
in llm_routes.yaml under `rate_limit`.
"""
from __future__ import annotations

import time
from typing import Any

import requests

from omni.llm.backends.base import Backend, BackendError, LLMResult
from omni.utils.env import get_env

_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqBackend(Backend):
    def _api_key(self) -> str:
        env_name = self.config.get("api_key_env", "GROQ_API_KEY")
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
        }
        t0 = time.monotonic()
        try:
            r = requests.post(_BASE_URL, json=body, headers=headers,
                              timeout=int(self.config.get("timeout_s", 60)))
        except Exception as e:
            raise BackendError(f"groq request failed: {e}") from e
        if r.status_code == 429:
            raise BackendError(f"groq rate-limited: {r.text[:200]}")
        if r.status_code != 200:
            raise BackendError(f"groq {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise BackendError(f"groq malformed response: {data}") from e
        usage = data.get("usage", {}) or {}
        return LLMResult(
            backend=self.name,
            text=text,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.monotonic() - t0) * 1000,
            extra={"model": model, "id": data.get("id")},
        )
