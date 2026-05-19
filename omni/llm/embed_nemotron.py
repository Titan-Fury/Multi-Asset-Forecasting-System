"""Free Nvidia Llama-Nemotron-Embed-VL 1B (OpenRouter) text embedder.

OpenRouter exposes the model at the OpenAI-compatible /embeddings
endpoint with `model="nvidia/llama-nemotron-embed-vl-1b-v2:free"`.

Output: 2048-dim float vector per input string (see EMBED_DIM).

Use cases:
    - S3 news embeddings (replaces FinBERT 768-d for cross-asset
      semantic search; FinBERT remains the sentiment scorer).
    - Future RAG over filings.

This module deliberately keeps zero hard dependency on the rest of the
LLM router so it can be imported during training without pulling the
full backend stack. Falls back to an EMBED_DIM zero vector if the API
key is missing — callers should check `is_zero(vec)` before using.

Free tier limits (rough): ~20 rpm / ~200 rpd. Always batch (up to 96
inputs per request) and cache aggressively.
"""
from __future__ import annotations

import os
import time
from typing import Iterable

import numpy as np

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
# Phase 24.15 — Nemotron VL 1B v2 returns 2048-d as of mid-2026 (model
# update; was 1024 earlier). Verified empirically via /v1/embeddings.
EMBED_DIM = 2048


def is_zero(vec: np.ndarray) -> bool:
    return float(np.linalg.norm(vec)) < 1e-9


def _zero_batch(n: int) -> np.ndarray:
    return np.zeros((n, EMBED_DIM), dtype=np.float32)


def embed(texts: Iterable[str],
          *,
          api_key: str | None = None,
          timeout_s: float = 30.0,
          retries: int = 2) -> np.ndarray:
    """Embed a list of texts. Returns (N, EMBED_DIM) float32 array.

    Returns zeros (and logs nothing — caller decides) on any failure.
    """
    items = [t if t is not None else "" for t in texts]
    if not items:
        return _zero_batch(0)
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or requests is None:
        return _zero_batch(len(items))

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "input": items}

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(OPENROUTER_URL, json=payload, headers=headers,
                              timeout=timeout_s)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json().get("data", [])
            if len(data) != len(items):
                return _zero_batch(len(items))
            arr = np.array([d.get("embedding", [0.0] * EMBED_DIM) for d in data],
                           dtype=np.float32)
            # L2-normalise (cosine-similarity ready)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1 + attempt)
    _ = last_err  # (could log)
    return _zero_batch(len(items))
