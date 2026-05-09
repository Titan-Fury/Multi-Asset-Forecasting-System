"""Local Ollama backend (HTTP /api/generate)."""
from __future__ import annotations

import time
from typing import Any

import requests

from omni.llm.backends.base import Backend, BackendError, LLMResult


class OllamaBackend(Backend):
    def _url(self) -> str:
        base = self.config.get("base_url") or "http://localhost:11434"
        return f"{base.rstrip('/')}/api/generate"

    def health(self) -> bool:
        try:
            base = self.config.get("base_url") or "http://localhost:11434"
            r = requests.get(f"{base.rstrip('/')}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def complete(self, prompt: str, **kwargs: Any) -> LLMResult:
        timeout = float(self.config.get("timeout_s", 120))
        model = self.config["model"]
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(kwargs.get("temperature", 0.2)),
                "num_predict": int(kwargs.get("max_tokens", 1024)),
            },
        }
        t0 = time.monotonic()
        try:
            r = requests.post(self._url(), json=body, timeout=timeout)
        except Exception as e:
            raise BackendError(f"ollama request failed: {e}") from e
        if r.status_code != 200:
            raise BackendError(f"ollama {r.status_code}: {r.text[:300]}")
        data = r.json()
        return LLMResult(
            backend=self.name,
            text=data.get("response", ""),
            tokens_in=int(data.get("prompt_eval_count", 0)),
            tokens_out=int(data.get("eval_count", 0)),
            latency_ms=(time.monotonic() - t0) * 1000,
            extra={"model": model, "done_reason": data.get("done_reason")},
        )
