"""Finance-tuned prompt builders with strict citation contract.

Every prompt enforces:
    1. Answer ONLY from provided context (no parametric knowledge)
    2. Cite each claim with [N] referring to the numbered source
    3. Refuse if context is insufficient
    4. Never invent numbers — quote them from context

The prompts are written for instruction-tuned chat models (Cortex Claude,
Llama 70B Instruct, Gemini Flash). They work fine with smaller open
models too but quality drops.
"""
from __future__ import annotations

from typing import Iterable

from omni.llm.rag.retriever import RetrievedChunk


_SYSTEM_SIGNAL_EXPLAIN = """You are a financial signal explainer. You will be \
given a stock signal (LONG/SHORT/FLAT with conviction) and recent news context. \
Your job: write 3-4 sentences (max 80 words) explaining WHY the signal \
direction makes sense, grounded in the news context.

STRICT RULES:
1. Use ONLY the provided NEWS CONTEXT. Do not draw on outside knowledge.
2. Cite every factual claim with [N] referring to the numbered source.
3. If multiple news items support the same point, cite them all e.g. [1][3].
4. Never invent numbers, dates, or quotes — quote exactly from context.
5. If the news context is empty or doesn't justify the signal, say so plainly: \
"Insufficient news context to attribute this signal." Don't make stuff up.
6. No financial advice disclaimers, no "this is not financial advice" boilerplate.
7. Plain-language English. No jargon dump."""


def _format_context_block(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""
    if not chunks:
        return "(no recent news context retrieved)"
    lines = []
    for i, ch in enumerate(chunks, start=1):
        # Truncate the chunk text so a 5-chunk context fits comfortably
        # in a 2k-token window. Each chunk is at most ~400 chars rendered.
        text = ch.text.strip()
        if len(text) > 400:
            text = text[:400].rstrip() + "..."
        meta_bits = [ch.ticker, ch.date, ch.source]
        meta = ", ".join([m for m in meta_bits if m])
        lines.append(f"[{i}] ({meta}) {text}")
    return "\n".join(lines)


def build_signal_explanation_prompt(*,
                                       ticker: str,
                                       display_name: str,
                                       asset_class: str,
                                       regime: str,
                                       pillars: dict,
                                       proba_up: float,
                                       direction: int,
                                       conviction: float,
                                       chunks: list[RetrievedChunk],
                                       ) -> str:
    """Build the full prompt sent to the LLM router for `signal_explanation_rag`.

    The shape is (system instruction + context + question), concatenated as
    a single string because some backends in the router are completion-style
    rather than chat-style (Ollama qwen, certain OpenRouter models). For
    chat-style backends this still works as a single user message — the
    leading system text is treated as part of the user turn.
    """
    direction_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(int(direction), "FLAT")
    pillar_line = (
        f"technical={float(pillars.get('technical', 0)):.0f}, "
        f"fundamental={float(pillars.get('fundamental', 0)):.0f}, "
        f"sentiment={float(pillars.get('sentiment', 0)):.0f}, "
        f"macro={float(pillars.get('macro', 0)):.0f}"
    )
    context_block = _format_context_block(chunks)

    return (
        _SYSTEM_SIGNAL_EXPLAIN
        + "\n\n=== SIGNAL ===\n"
        + f"Stock: {display_name} ({ticker}, {asset_class.upper()})\n"
        + f"Regime: {regime}\n"
        + f"Model proba(up, 5d): {proba_up:.3f}\n"
        + f"Rubric pillars (0-100): {pillar_line}\n"
        + f"Final signal: {direction_label} (conviction {conviction:.0f}/100)\n"
        + "\n=== NEWS CONTEXT ===\n"
        + context_block
        + "\n\n=== ANSWER ===\n"
        + "Write 3-4 sentences (max 80 words) explaining WHY the signal is "
        + f"{direction_label} for {ticker}. Cite each claim with [N]. "
        + "If the news context doesn't support the signal, say so plainly."
    )


def build_qa_prompt(*, question: str, chunks: list[RetrievedChunk]) -> str:
    """Open-ended Q&A prompt for an "Ask Omni" style chat (Phase D).

    Kept here so the contract is consistent across features.
    """
    context_block = _format_context_block(chunks)
    return (
        "You answer market questions using ONLY the provided context. "
        "Cite every claim with [N]. If the context doesn't contain the "
        "answer, say \"I don't have enough information.\" Never invent "
        "numbers, dates, or quotes."
        + "\n\n=== CONTEXT ===\n"
        + context_block
        + "\n\n=== QUESTION ===\n"
        + question.strip()
        + "\n\n=== ANSWER ===\n"
    )
