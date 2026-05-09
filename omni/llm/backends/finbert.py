"""FinBERT sentiment backend (ProsusAI/finbert).

Returns a sentiment label (positive/negative/neutral) and a signed score.
Used for per-article sentiment in the news pipeline.
"""
from __future__ import annotations

import time
from typing import Any

from omni.llm.backends.base import Backend, BackendError, LLMResult
from omni.utils.logger import log


class FinBERTBackend(Backend):
    _pipe = None  # class-level cache to avoid reloading

    def _load(self) -> None:
        if FinBERTBackend._pipe is not None:
            return
        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise BackendError(f"transformers/torch not installed: {e}") from e
        device_cfg = self.config.get("device", "auto")
        if device_cfg == "auto":
            device = 0 if torch.cuda.is_available() else -1
        elif device_cfg == "cuda":
            device = 0
        else:
            device = -1
        model = self.config.get("model", "ProsusAI/finbert")
        log.info(f"loading FinBERT '{model}' on device={device}")
        FinBERTBackend._pipe = pipeline(
            "sentiment-analysis", model=model, device=device, top_k=None,
        )

    def complete(self, prompt: str, **kwargs: Any) -> LLMResult:
        self._load()
        t0 = time.monotonic()
        try:
            out = FinBERTBackend._pipe(prompt[:512])
        except Exception as e:
            raise BackendError(f"finbert inference failed: {e}") from e
        # `out` is list of [{"label","score"},...] for top_k=None, single-input
        if isinstance(out, list) and out and isinstance(out[0], list):
            scores = {x["label"].lower(): x["score"] for x in out[0]}
        elif isinstance(out, list):
            scores = {x["label"].lower(): x["score"] for x in out}
        else:
            raise BackendError(f"unexpected finbert output: {out!r}")
        # Signed score in [-1, 1]: positive - negative
        signed = float(scores.get("positive", 0.0) - scores.get("negative", 0.0))
        label = max(scores, key=scores.get)
        return LLMResult(
            backend=self.name,
            text=label,
            latency_ms=(time.monotonic() - t0) * 1000,
            extra={"scores": scores, "signed_score": signed},
        )
