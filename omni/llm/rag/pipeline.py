"""End-to-end RAG pipeline: question -> retrieve -> generate -> validate.

Single entry point used by the prediction pipeline. Composes all five
sub-modules:

    chunker      via callers (this module doesn't ingest)
    store        via HybridRetriever
    embedder     via HybridRetriever
    retriever    HybridRetriever.retrieve()
    prompts      build_signal_explanation_prompt(), build_qa_prompt()
    validators   validate()

Failure modes are explicit. Every fallback path is logged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from omni.llm.backends.base import BackendError
from omni.llm.rag.chunker import chunk_news_dataframe
from omni.llm.rag.embedder import get_embedder
from omni.llm.rag.prompts import (
    build_qa_prompt, build_signal_explanation_prompt,
)
from omni.llm.rag.retriever import HybridRetriever, RetrievedChunk
from omni.llm.rag.store import RAGStore
from omni.llm.rag.validators import ValidationResult, validate
from omni.llm.router import route as _llm_route

_log = logging.getLogger("omni.llm.rag.pipeline")


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    text: str
    citations: list[dict] = field(default_factory=list)  # one per cited chunk
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    grounding_score: float = 0.0
    is_safe: bool = False
    fallback_used: str = ""    # "" | "no_news" | "low_grounding" | "llm_error" | "no_chunks"
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Singleton-friendly pipeline. Construct once, reuse across requests."""

    def __init__(self, store: RAGStore | None = None,
                  retriever: HybridRetriever | None = None) -> None:
        self.store = store or RAGStore()
        self.retriever = retriever or HybridRetriever(store=self.store)

    # ---------- ingestion (idempotent upsert) ----------

    def ingest_news_for_ticker(self, df: pd.DataFrame, *,
                                 ticker: str, asset_class: str) -> int:
        """One-shot ingest of a per-ticker news DataFrame.

        Idempotent: re-ingesting the same articles produces the same chunk
        ids so Qdrant just overwrites in place. Safe to call from
        `predict_ticker()` every prediction.
        """
        if df is None or df.empty:
            return 0
        chunks = chunk_news_dataframe(df, ticker=ticker.upper(),
                                        asset_class=asset_class)
        if not chunks:
            return 0
        # Embed all chunks (cached — cheap on repeats)
        embedder = get_embedder()
        vecs = embedder.embed_many([c.text for c in chunks])
        # Failed embeddings (zero vectors) are skipped by store.upsert()
        try:
            n = self.store.upsert(chunks, vecs)
        except Exception as e:
            _log.warning("rag ingest upsert failed for %s: %s", ticker, e)
            return 0
        return n

    # ---------- generation paths ----------

    def explain_signal(self, *,
                        ticker: str,
                        display_name: str,
                        asset_class: str,
                        regime: str,
                        pillars: dict,
                        proba_up: float,
                        direction: int,
                        conviction: float,
                        date_window_days: int = 14,
                        top_k: int = 5,
                        task_name: str = "signal_explanation_rag",
                        grounding_min: float = 0.55,
                        ) -> RAGResponse:
        """Generate a grounded why-text for a stock signal.

        Workflow:
          1. Retrieve top-k news chunks for the ticker
          2. Build prompt with citation contract
          3. Route through llm_routes.yaml task chain
          4. Validate output (citations, numeric grounding, sentence score)
          5. On `is_safe=False`: try once more with conservative settings;
             if still unsafe, return `fallback_used="low_grounding"` so the
             caller can substitute its deterministic explanation
        """
        # Build the retrieval query — combine ticker, regime, and direction
        # as keywords so BM25 can find recent ticker-relevant news
        direction_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(int(direction), "FLAT")
        query = (
            f"{display_name} {ticker} {regime} regime "
            f"{direction_label} signal recent news catalysts"
        )

        chunks = self.retriever.retrieve(
            query, ticker=ticker, asset_class=asset_class,
            doc_types=["news"],
            date_window_days=date_window_days, top_k=top_k,
        )

        if not chunks:
            return RAGResponse(
                text="", retrieved=[], grounding_score=0.0,
                is_safe=False, fallback_used="no_chunks",
                issues=["no_chunks_retrieved"],
            )

        prompt = build_signal_explanation_prompt(
            ticker=ticker, display_name=display_name,
            asset_class=asset_class, regime=regime, pillars=pillars,
            proba_up=proba_up, direction=direction, conviction=conviction,
            chunks=chunks,
        )

        # First attempt: default temperature
        text, llm_err = self._generate(prompt, task=task_name, temperature=0.4)
        if llm_err:
            return RAGResponse(
                text="", retrieved=chunks, grounding_score=0.0,
                is_safe=False, fallback_used="llm_error",
                issues=[f"llm_error:{llm_err}"],
            )

        result: ValidationResult = validate(
            text, chunks,
            grounding_min=grounding_min,
            require_citations=True, strip_unsupported=True,
        )

        # Retry once with stricter temperature if unsafe
        if not result.is_safe:
            _log.info("rag explain unsafe (%s); retrying with temp=0.1",
                       result.issues)
            retry_text, retry_err = self._generate(
                prompt, task=task_name, temperature=0.1,
            )
            if not retry_err:
                retry_result = validate(
                    retry_text, chunks,
                    grounding_min=grounding_min,
                    require_citations=True, strip_unsupported=True,
                )
                if retry_result.is_safe:
                    return self._make_response(
                        retry_result, chunks, fallback_used="",
                    )
                # Both attempts failed — return the better-grounded one
                # but mark it as fallback so caller can override
                better = (retry_result if retry_result.grounding_score
                           > result.grounding_score else result)
                return self._make_response(
                    better, chunks, fallback_used="low_grounding",
                )

        return self._make_response(result, chunks, fallback_used="")

    # ---------- internals ----------

    def _generate(self, prompt: str, *, task: str,
                   temperature: float) -> tuple[str, str | None]:
        """Route prompt through llm_routes.yaml task chain. Returns
        (text, error). On success error is None."""
        try:
            res = _llm_route(task, prompt, temperature=temperature,
                              max_tokens=300)
            text = (res.text if hasattr(res, "text") else str(res)).strip()
            return text, None
        except (BackendError, KeyError) as e:
            return "", str(e)
        except Exception as e:  # noqa: BLE001
            _log.warning("rag _generate unexpected error: %s", e)
            return "", str(e)

    def _make_response(self, result: ValidationResult,
                        chunks: list[RetrievedChunk],
                        *, fallback_used: str) -> RAGResponse:
        """Convert internal ValidationResult into the public RAGResponse,
        materialising citation metadata for the UI to render."""
        # Build citation list — one entry per chunk, regardless of which
        # were actually cited in the answer (caller can filter on demand)
        citations = []
        for i, ch in enumerate(chunks, start=1):
            citations.append({
                "n": i,
                "ticker": ch.ticker,
                "source": ch.source,
                "date": ch.date,
                "url": ch.payload.get("url", ""),
                "title": ch.payload.get("title", ""),
                "score": ch.score,
            })
        return RAGResponse(
            text=result.sanitized_text,
            citations=citations,
            retrieved=chunks,
            grounding_score=result.grounding_score,
            is_safe=result.is_safe,
            fallback_used=fallback_used,
            issues=result.issues,
        )

    # ---------- general Q&A (Phase D placeholder) ----------

    def answer_question(self, question: str, *,
                          ticker: str | None = None,
                          asset_class: str = "us",
                          doc_types: list[str] | None = None,
                          date_window_days: int = 30,
                          top_k: int = 6,
                          task_name: str = "signal_explanation_rag",
                          grounding_min: float = 0.55,
                          ) -> RAGResponse:
        """Open-ended Q&A. Used by the future "Ask Omni" tab (Phase D)."""
        if not ticker:
            return RAGResponse(
                text="Question requires a ticker scope for now.",
                fallback_used="no_ticker", is_safe=False,
            )
        chunks = self.retriever.retrieve(
            question, ticker=ticker, asset_class=asset_class,
            doc_types=doc_types or ["news"],
            date_window_days=date_window_days, top_k=top_k,
        )
        if not chunks:
            return RAGResponse(
                text="I don't have enough information to answer.",
                fallback_used="no_chunks", is_safe=False,
            )
        prompt = build_qa_prompt(question=question, chunks=chunks)
        text, llm_err = self._generate(prompt, task=task_name, temperature=0.3)
        if llm_err:
            return RAGResponse(
                text="", retrieved=chunks, fallback_used="llm_error",
                issues=[f"llm_error:{llm_err}"],
            )
        result = validate(
            text, chunks,
            grounding_min=grounding_min,
            require_citations=True, strip_unsupported=True,
        )
        return self._make_response(
            result, chunks,
            fallback_used="" if result.is_safe else "low_grounding",
        )


# ---------------------------------------------------------------------------
# Module-level singleton (cheap to share across requests)
# ---------------------------------------------------------------------------

_DEFAULT: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Process-singleton accessor."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RAGPipeline()
    return _DEFAULT
