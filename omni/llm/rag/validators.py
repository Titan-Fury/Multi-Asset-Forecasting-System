"""Anti-hallucination layer for RAG outputs.

Three orthogonal checks, applied in order:

    1. Citation requirement — every factual sentence should have a [N] marker.
    2. Numeric grounding — every number in the answer must appear in at least
       one retrieved chunk. Unsupported numbers get replaced with [unverified].
    3. Sentence-level grounding score — embed each answer sentence, max-cosine
       vs retrieved chunks, mean across sentences. Threshold default 0.55.

If any hard check fails (no citations, grounding score below threshold), the
caller can decide to retry with stricter generation params or fall back to
a deterministic refusal.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from omni.llm.rag.retriever import RetrievedChunk

_log = logging.getLogger("omni.llm.rag.validators")


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

# Match dollar amounts, percentages, plain decimals, ratios. Tuned to be
# slightly aggressive — it's better to flag a year ("2024") and verify it
# came from context than to silently let invented numbers through.
_NUMBER_RE = re.compile(
    r"""(?ix)
    (?:                          # one of:
       \$\s?\d[\d,]*\.?\d*       #   $35,100  $1.5
       | \d+\.?\d*\s?(?:%|bps|basis\s*points)  # 4.2%   30 bps
       | \d{1,3}(?:,\d{3})+(?:\.\d+)?           # 1,234,567
       | \b\d+\.\d+\b            # 35.1   1.23
       | \b\d{4}\b               # 2024
       | \b\d+(?:[.,]\d+)?\s?(?:billion|million|trillion|B|M|T)\b  # 35B
    )
    """,
)


def extract_numbers(text: str) -> list[str]:
    """Return all numeric tokens in `text`, preserving original surface form."""
    if not text:
        return []
    return [m.group(0).strip() for m in _NUMBER_RE.finditer(text)]


def _normalize_number(s: str) -> str:
    """Best-effort normaliser so '$35.1B' and '35.1 billion' compare equal."""
    s = s.lower().replace(",", "").replace("$", "").strip()
    s = s.replace("billion", "b").replace("million", "m").replace("trillion", "t")
    s = re.sub(r"\s+", "", s)
    return s


def numeric_grounding_check(answer: str,
                              chunks: list[RetrievedChunk]
                              ) -> tuple[list[str], list[str]]:
    """Check every number in `answer` against the union of chunk text.

    Returns (supported, unsupported) lists of the original surface forms.
    """
    nums = extract_numbers(answer)
    if not nums:
        return [], []
    haystack = " ".join(c.text for c in chunks).lower()
    haystack_norm = haystack.replace(",", "").replace("$", "")
    supported, unsupported = [], []
    for n in nums:
        n_norm = _normalize_number(n)
        # Multi-strategy match: exact substring on raw, then on normalised
        if (n.lower() in haystack
            or n_norm in haystack_norm
            or n_norm.replace("b", " billion") in haystack
            or n_norm.replace("m", " million") in haystack):
            supported.append(n)
        else:
            unsupported.append(n)
    return supported, unsupported


def strip_unsupported_numbers(answer: str, unsupported: list[str]) -> str:
    """Replace each unsupported number in-place with `[unverified]`.

    We escape regex chars in each token (dollar signs, periods etc.) and do
    a single pass per token. Word boundaries are not used because some
    tokens contain non-word characters ($).
    """
    if not unsupported:
        return answer
    out = answer
    for n in unsupported:
        out = out.replace(n, "[unverified]")
    return out


# ---------------------------------------------------------------------------
# Citation check
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(\d+)\]")


def citation_check(answer: str, n_chunks: int) -> tuple[bool, set[int], list[int]]:
    """Verify the answer cites at least one source and that all referenced
    indices are within the retrieved set.

    Returns (has_any, valid_indices, invalid_indices). `valid_indices` is a
    set of 1-based citation numbers found that are <= n_chunks.
    `invalid_indices` is a list of citation numbers > n_chunks (hallucinated
    references).
    """
    found = [int(m.group(1)) for m in _CITATION_RE.finditer(answer or "")]
    if not found:
        return False, set(), []
    valid = {i for i in found if 1 <= i <= n_chunks}
    invalid = [i for i in found if i > n_chunks or i < 1]
    return True, valid, invalid


# ---------------------------------------------------------------------------
# Sentence-level grounding score
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text or "") if s.strip()]


def claim_grounding_score(answer: str, chunks: list[RetrievedChunk]) -> float:
    """Embed each answer sentence and each chunk text; for every sentence
    take max cosine vs all chunks, then return the mean across sentences.

    Score in [0, 1]. >0.55 is a good baseline for "well-grounded".

    If embedder fails (returns zeros), this returns 0.0 — caller treats
    as ungrounded.
    """
    sentences = split_sentences(answer)
    if not sentences or not chunks:
        return 0.0

    # Strip citation markers from sentences so they don't dominate similarity
    clean = [_CITATION_RE.sub("", s).strip() for s in sentences]
    clean = [s for s in clean if s]
    if not clean:
        return 0.0

    chunk_texts = [c.text for c in chunks]

    # Use the same embedder as retrieval — already cached so this is cheap
    try:
        from omni.llm.rag.embedder import get_embedder
        emb = get_embedder()
        sent_vecs = emb.embed_many(clean)
        chunk_vecs = emb.embed_many(chunk_texts)
    except Exception as e:
        _log.warning("grounding embed failed: %s", e)
        return 0.0

    # All-zero rows indicate failed embeddings; skip them
    valid_sent = np.linalg.norm(sent_vecs, axis=1) > 1e-6
    valid_chunk = np.linalg.norm(chunk_vecs, axis=1) > 1e-6
    if not valid_sent.any() or not valid_chunk.any():
        return 0.0

    sent_vecs = sent_vecs[valid_sent]
    chunk_vecs = chunk_vecs[valid_chunk]
    # Both already L2-normalised by embedder, so dot product = cosine
    sims = sent_vecs @ chunk_vecs.T  # (S, C)
    max_per_sent = sims.max(axis=1)
    return float(np.clip(max_per_sent.mean(), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Result type + top-level validate()
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    is_safe: bool
    grounding_score: float
    has_citations: bool
    invalid_citations: list[int] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    sanitized_text: str = ""
    issues: list[str] = field(default_factory=list)


def validate(answer: str,
              chunks: list[RetrievedChunk],
              *,
              grounding_min: float = 0.55,
              require_citations: bool = True,
              strip_unsupported: bool = True,
              ) -> ValidationResult:
    """Run all three checks and return a ValidationResult."""
    issues: list[str] = []
    sanitized = answer.strip()

    # Step 1: citations
    has_any, valid_set, invalid = citation_check(sanitized, len(chunks))
    if require_citations and not has_any:
        issues.append("no_citations")
    if invalid:
        issues.append(f"invalid_citations:{invalid}")

    # Step 2: numeric grounding
    _, unsupported = numeric_grounding_check(sanitized, chunks)
    if unsupported:
        issues.append(f"unsupported_numbers:{unsupported}")
        if strip_unsupported:
            sanitized = strip_unsupported_numbers(sanitized, unsupported)

    # Step 3: grounding score (skip if no chunks — meaningless)
    score = claim_grounding_score(sanitized, chunks) if chunks else 0.0
    if chunks and score < grounding_min:
        issues.append(f"low_grounding:{score:.2f}")

    is_safe = (
        (has_any or not require_citations)
        and not invalid
        and (score >= grounding_min or not chunks)
    )

    return ValidationResult(
        is_safe=is_safe,
        grounding_score=score,
        has_citations=has_any,
        invalid_citations=invalid,
        unsupported_numbers=unsupported,
        sanitized_text=sanitized,
        issues=issues,
    )
