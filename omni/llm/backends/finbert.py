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
        # Phase 24.18 — silence transformers auto-discovery noise (torchvision
        # missing for `prompt_depth_anything` etc.) before we touch the lib.
        import os
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            import torch
            import transformers.utils.logging as _hf_log
            _hf_log.set_verbosity_error()
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

    # Phase 24.21 — native batch path. The HF sentiment-analysis pipeline
    # supports list inputs and dispatches them as a single tensor through the
    # model. On GPU this is a 10-50x speedup over per-item dispatch; even on
    # CPU it kills the Python-loop overhead. The sentiment-feature builder
    # was previously calling `route("sentiment_per_article", text)` inside
    # `df.iterrows()` — 200 articles = 200 sequential model calls. With
    # `complete_batch` the caller can score the whole frame in one shot.
    def complete_batch(self, prompts: list[str], **kwargs: Any) -> list[LLMResult]:
        self._load()
        if not prompts:
            return []
        truncated = [(p or "")[:512] for p in prompts]
        t0 = time.monotonic()
        try:
            outs = FinBERTBackend._pipe(truncated, batch_size=32)
        except Exception as e:
            raise BackendError(f"finbert batch inference failed: {e}") from e
        elapsed = (time.monotonic() - t0) * 1000
        results: list[LLMResult] = []
        for raw in outs:
            # raw is list[{"label","score"},...] when top_k=None.
            if isinstance(raw, list):
                scores = {x["label"].lower(): x["score"] for x in raw}
            elif isinstance(raw, dict):
                scores = {raw["label"].lower(): raw["score"]}
            else:
                scores = {"neutral": 1.0}
            signed = float(scores.get("positive", 0.0) - scores.get("negative", 0.0))
            label = max(scores, key=scores.get) if scores else "neutral"
            results.append(LLMResult(
                backend=self.name,
                text=label,
                latency_ms=elapsed / max(len(prompts), 1),  # amortized per item
                extra={"scores": scores, "signed_score": signed},
            ))
        return results
