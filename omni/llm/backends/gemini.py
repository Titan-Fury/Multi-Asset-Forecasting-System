"""Google Gemini backend (Generative Language API).

Uses the `v1beta/models/<model>:generateContent` endpoint with an API key
in the query string (Google's preferred auth for the public API). Free tier
allows generous quotas on gemini-1.5-flash and gemini-2.0-flash.

Free tier limits (subject to change): ~15 rpm / 1500 rpd on flash models.
Configure `rate_limit` in llm_routes.yaml accordingly.
"""
from __future__ import annotations

import time
from typing import Any

from omni.llm.backends.base import Backend, BackendError, LLMResult, http_session
from omni.utils.env import get_env

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiBackend(Backend):
    def _api_key(self) -> str:
        env_name = self.config.get("api_key_env", "GEMINI_API_KEY")
        key = get_env(env_name)
        if not key:
            raise BackendError(f"{env_name} not set")
        return key

    def complete(self, prompt: str, **kwargs: Any) -> LLMResult:
        model = self.config["model"]
        url = f"{_BASE_URL}/{model}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(kwargs.get("temperature", 0.2)),
                "maxOutputTokens": int(kwargs.get("max_tokens", 1024)),
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key(),
        }
        t0 = time.monotonic()
        try:
            # Phase 24.21 — pooled session.
            r = http_session().post(url, json=body, headers=headers,
                              timeout=int(self.config.get("timeout_s", 60)))
        except Exception as e:
            raise BackendError(f"gemini request failed: {e}") from e
        if r.status_code == 429:
            raise BackendError(f"gemini rate-limited: {r.text[:200]}")
        if r.status_code != 200:
            raise BackendError(f"gemini {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            cands = data.get("candidates") or []
            if not cands:
                raise BackendError(f"gemini no candidates: {data}")
            parts = cands[0].get("content", {}).get("parts", []) or []
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"gemini malformed response: {data}") from e
        usage = data.get("usageMetadata", {}) or {}
        return LLMResult(
            backend=self.name,
            text=text,
            tokens_in=int(usage.get("promptTokenCount", 0)),
            tokens_out=int(usage.get("candidatesTokenCount", 0)),
            latency_ms=(time.monotonic() - t0) * 1000,
            extra={"model": model,
                   "finish_reason": cands[0].get("finishReason")},
        )
