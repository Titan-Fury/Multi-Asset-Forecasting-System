"""Hybrid retriever: BM25 (keyword) + Qdrant (dense) + bge-reranker-base.

Pipeline:
    1. Build per-query BM25 index over the ticker+date filtered subset
       (typically <1k chunks — fast)
    2. Dense search via Qdrant on the same filtered subset
    3. Reciprocal Rank Fusion combines both ranked lists
    4. Reranker re-scores the top fused candidates as (query, doc) pairs
    5. Return top-K most relevant chunks

The reranker (BAAI/bge-reranker-base) loads on CUDA when available
(checked once, cached). Falls back to CPU automatically.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from omni.llm.rag.embedder import get_embedder
from omni.llm.rag.store import RAGStore

_log = logging.getLogger("omni.llm.rag.retriever")

# Reranker model id and lazy singleton
_RERANK_MODEL = "BAAI/bge-reranker-base"
_RERANK_MAX_LENGTH = 512
_rerank_singleton = None
_rerank_lock = threading.Lock()


def _get_reranker():
    """Lazy-load the cross-encoder reranker. Tries CUDA, falls back to CPU."""
    global _rerank_singleton
    if _rerank_singleton is not None:
        return _rerank_singleton
    with _rerank_lock:
        if _rerank_singleton is not None:
            return _rerank_singleton
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            import torch  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _log.info("loading bge-reranker-base on device=%s", device)
            _rerank_singleton = CrossEncoder(
                _RERANK_MODEL, device=device,
                max_length=_RERANK_MAX_LENGTH,
            )
        except Exception as e:
            _log.warning("reranker unavailable, will skip rerank stage: %s", e)
            _rerank_singleton = False  # sentinel: unavailable
            # Phase 24.16 — emit degradation event so ops see this.
            try:
                from omni.llm.rag.degradation import monitor
                monitor.record(
                    component="reranker",
                    severity="error",
                    event="load_failed",
                    details={
                        "error_type": type(e).__name__,
                        "error_message": str(e)[:300],
                        "model": _RERANK_MODEL,
                    },
                )
            except Exception:  # noqa: BLE001
                pass
    return _rerank_singleton


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    text: str
    ticker: str
    asset_class: str
    source: str
    doc_type: str
    date: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, p: dict, score: float) -> "RetrievedChunk":
        return cls(
            text=str(p.get("text", "")),
            ticker=str(p.get("ticker", "")),
            asset_class=str(p.get("asset_class", "")),
            source=str(p.get("source", "")),
            doc_type=str(p.get("doc_type", "")),
            date=str(p.get("date", "")),
            score=float(score),
            payload={k: v for k, v in p.items()
                     if k not in ("text", "ticker", "asset_class",
                                  "source", "doc_type", "date")},
        )


# ---------------------------------------------------------------------------
# BM25 helper
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+\.?\d*%?|\$?\d[\d,]*\.?\d*")


def _tokenize(text: str) -> list[str]:
    """Cheap tokeniser that preserves dollar amounts and percentages."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _bm25_rank(query: str, candidates: list[dict], top_k: int) -> list[tuple[int, float]]:
    """Build a one-shot BM25 index over `candidates` and return top_k
    `(index, score)` pairs sorted by score desc."""
    if not candidates:
        return []
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError:
        _log.warning("rank_bm25 not installed; skipping sparse retrieval")
        return []
    corpus = [_tokenize(c.get("text", "")) for c in candidates]
    if not any(corpus):
        return []
    bm = BM25Okapi(corpus)
    q_tok = _tokenize(query)
    scores = bm.get_scores(q_tok)
    order = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in order if float(scores[i]) > 0]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf(*rankings: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
    """Combine multiple ranked lists via Reciprocal Rank Fusion.

    Score = sum(1 / (k + rank_in_list)) across all lists where the doc appears.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Combines BM25, Qdrant dense search, and a cross-encoder reranker.

    Typical use:
        ret = HybridRetriever()
        chunks = ret.retrieve("Why did NVDA gap up?",
                              ticker="NVDA", asset_class="us",
                              doc_types=["news"], date_window_days=14,
                              top_k=5)
    """

    def __init__(self, store: RAGStore | None = None,
                  embedder=None, *, default_pool: int = 50,
                  rrf_pool: int = 30) -> None:
        self.store = store or RAGStore()
        self.embedder = embedder or get_embedder()
        self.default_pool = default_pool
        self.rrf_pool = rrf_pool

    def retrieve(self, query: str, *,
                  ticker: str,
                  asset_class: str,
                  doc_types: list[str],
                  date_window_days: int = 14,
                  top_k: int = 5,
                  ) -> list[RetrievedChunk]:
        """Run the full hybrid pipeline and return top_k RetrievedChunks."""
        if not query or not query.strip():
            return []
        if not doc_types:
            doc_types = ["news"]

        date_to = datetime.utcnow().date().isoformat()
        date_from = (datetime.utcnow().date()
                      - timedelta(days=int(date_window_days))).isoformat()

        # Step 1: dense search via Qdrant (across requested doc_types)
        qvec = self.embedder.embed_one(query)
        dense_pool: list[dict] = []
        for dt in doc_types:
            dense_pool.extend(self.store.search(
                asset_class=asset_class, doc_type=dt,
                query_vec=qvec, ticker=ticker.upper(),
                date_from=date_from, date_to=date_to,
                top_k=self.default_pool,
            ))
        # Dedup by qdrant id
        seen: set = set()
        candidates: list[dict] = []
        for hit in dense_pool:
            qid = hit.get("_qdrant_id")
            if qid in seen:
                continue
            seen.add(qid)
            candidates.append(hit)

        if not candidates:
            return []

        # Step 2: BM25 over the candidate pool (already filtered by ticker)
        bm25_top = _bm25_rank(query, candidates, top_k=self.default_pool)
        # Translate dense's already-ranked-by-cosine list to (idx, score)
        dense_top = [(i, float(c.get("score", 0.0)))
                     for i, c in enumerate(candidates)]
        # Sort dense_top by score desc
        dense_top.sort(key=lambda kv: kv[1], reverse=True)

        # Step 3: RRF fusion
        fused = _rrf(dense_top, bm25_top)[: self.rrf_pool]

        # Step 4: reranker (cross-encoder)
        reranker = _get_reranker()
        if reranker and reranker is not False:
            try:
                pairs = [(query, candidates[i].get("text", "")[:_RERANK_MAX_LENGTH * 4])
                         for i, _ in fused]
                rerank_scores = reranker.predict(pairs, batch_size=32,
                                                   show_progress_bar=False)
                # bge-reranker outputs higher = more relevant
                ordered = sorted(
                    zip(fused, rerank_scores),
                    key=lambda kv: float(kv[1]), reverse=True,
                )
                final = [(idx_score[0], float(score))
                         for (idx_score, score) in ordered][:top_k]
            except Exception as e:
                _log.warning("rerank failed: %s; falling back to RRF order", e)
                final = [(idx, sc) for idx, sc in fused[:top_k]]
        else:
            # No reranker available — use RRF order
            final = [(idx, sc) for idx, sc in fused[:top_k]]

        out: list[RetrievedChunk] = []
        for idx, score in final:
            cand = candidates[idx]
            out.append(RetrievedChunk.from_payload(cand, score))
        return out
