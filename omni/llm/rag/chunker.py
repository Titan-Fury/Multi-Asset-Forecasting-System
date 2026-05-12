"""Section-aware chunking for finance documents.

Different document types have different structural cues:
    - news article          one chunk per article (split if description >800c)
    - 10-K / 10-Q filing    section-aware (Item 1A, Item 7, Item 7A, Item 8)
    - earnings transcript   speaker-turn aware

Chunks are kept smallish (typically <600 tokens) because reranker quality
degrades with longer context windows. We DO preserve enough context that
a single chunk is independently meaningful — never split a sentence.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

# 10-K section markers (case-insensitive). Order matters for split logic.
_FILING_SECTIONS = [
    "item 1a", "item 1b", "item 1", "item 2", "item 3", "item 4",
    "item 5", "item 6", "item 7a", "item 7", "item 8", "item 9",
]

# Paragraph splitter — used for filings + transcripts after section split
_PARA_RE = re.compile(r"\n\s*\n+")

# Sentence splitter (cheap; not perfect but adequate for chunk sizing)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Soft chunk-size targets (characters, NOT tokens — character count is a
# reasonable proxy and avoids importing tiktoken just for chunking)
_NEWS_MAX_CHARS = 1600          # ~400 tokens
_FILING_MAX_CHARS = 2400        # ~600 tokens
_FILING_TARGET_CHARS = 1800     # split ealier if we can find a sentence boundary
_TRANSCRIPT_MAX_CHARS = 1600


@dataclass
class Chunk:
    """A single retrievable unit. Goes into Qdrant payload + BM25 corpus."""
    text: str
    ticker: str
    asset_class: str
    source: str             # "newsapi" | "rss" | "sec_edgar" | "earnings_call"
    doc_type: str           # "news" | "filing_10k" | "filing_10q" | "transcript"
    date_iso: str           # YYYY-MM-DD
    chunk_index: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def stable_id(self) -> str:
        """Deterministic uint64 id for Qdrant point id.

        Hash includes (ticker, doc_type, source, date, chunk_index, text-prefix)
        so the SAME chunk re-ingested produces the same id (idempotent upsert).
        """
        key = "|".join([
            self.ticker, self.doc_type, self.source, self.date_iso,
            str(self.chunk_index), self.text[:200],
        ])
        h = hashlib.sha1(key.encode("utf-8")).digest()
        # Qdrant accepts uint64 ids; take low 64 bits of sha1
        return str(int.from_bytes(h[:8], "big"))


def _trim_to_sentence_boundary(text: str, max_chars: int) -> tuple[str, str]:
    """Split text at the latest sentence boundary <= max_chars. Returns (head, tail)."""
    if len(text) <= max_chars:
        return text, ""
    head = text[:max_chars]
    # Try to break at a sentence boundary
    m = list(_SENT_RE.finditer(head))
    if m:
        cut = m[-1].end()
        return text[:cut].rstrip(), text[cut:].lstrip()
    # Fallback: hard break at last whitespace
    cut = head.rfind(" ")
    if cut > max_chars // 2:
        return text[:cut].rstrip(), text[cut:].lstrip()
    return head, text[max_chars:]


def chunk_news_article(row: pd.Series, *, ticker: str, asset_class: str) -> list[Chunk]:
    """One chunk per news article: title + description, split if very long.

    Expected `row` schema (from omni.ingest.news):
        published_at, source, title, description, url, ticker_hint, lang,
        sent_score (optional, post-FinBERT)
    """
    title = str(row.get("title", "") or "").strip()
    desc = str(row.get("description", "") or "").strip()
    if not title and not desc:
        return []

    # Compose the chunk text. Title is short and headline-y; description is
    # the substantive content. Combine with a separator that the LLM treats
    # as natural prose.
    body = (f"{title}. {desc}".strip() if title and desc
            else (title or desc)).strip()

    # Date normalisation
    pub = row.get("published_at")
    if isinstance(pub, str):
        try:
            date_iso = datetime.fromisoformat(pub.replace("Z", "+00:00")).date().isoformat()
        except Exception:
            date_iso = datetime.utcnow().date().isoformat()
    else:
        try:
            date_iso = pd.to_datetime(pub).date().isoformat()
        except Exception:
            date_iso = datetime.utcnow().date().isoformat()

    source = str(row.get("source", "") or "unknown")
    extras = {
        "url": str(row.get("url", "") or ""),
        "title": title[:200],
        "lang": str(row.get("lang", "en") or "en"),
    }
    sent = row.get("sent_score")
    if sent is not None and not pd.isna(sent):
        try:
            extras["sent_score"] = float(sent)
        except (TypeError, ValueError):
            # Unparseable sentiment is fine; we just skip the metadata field.
            pass

    # Most articles fit in one chunk. Split only when very long.
    chunks: list[Chunk] = []
    remaining = body
    idx = 0
    while remaining:
        head, tail = _trim_to_sentence_boundary(remaining, _NEWS_MAX_CHARS)
        chunks.append(Chunk(
            text=head, ticker=ticker, asset_class=asset_class,
            source=source, doc_type="news", date_iso=date_iso,
            chunk_index=idx, extras=extras,
        ))
        idx += 1
        remaining = tail
        if idx > 4:  # safety: never produce >5 chunks per article
            break
    return chunks


def chunk_news_dataframe(df: pd.DataFrame, *, ticker: str,
                          asset_class: str) -> list[Chunk]:
    """Convenience: chunk every row in a news DataFrame."""
    if df is None or df.empty:
        return []
    out: list[Chunk] = []
    for _, row in df.iterrows():
        out.extend(chunk_news_article(row, ticker=ticker, asset_class=asset_class))
    return out


def _split_filing_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (section_label, section_text). If no section markers
    are found, returns a single ('full', text) entry."""
    lower = text.lower()
    # Find all section start positions
    hits: list[tuple[int, str]] = []
    for label in _FILING_SECTIONS:
        idx = 0
        while True:
            pos = lower.find(label, idx)
            if pos < 0:
                break
            # Require leading whitespace/newline so we don't match inside prose
            if pos == 0 or text[pos - 1] in " \n\t.":
                hits.append((pos, label))
            idx = pos + len(label)
    hits.sort()
    if not hits:
        return [("full", text)]

    out: list[tuple[str, str]] = []
    for i, (pos, label) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        section_text = text[pos:end].strip()
        out.append((label, section_text))
    return out


def chunk_filing(text: str, *, ticker: str, asset_class: str,
                  doc_type: str, date_iso: str, source: str = "sec_edgar"
                  ) -> list[Chunk]:
    """Section-aware chunker for 10-K / 10-Q.

    Each section becomes one or more chunks, broken at paragraph boundaries
    so a risk factor never gets split mid-sentence.
    """
    if not text or len(text) < 200:
        return []
    sections = _split_filing_sections(text)
    out: list[Chunk] = []
    cidx = 0
    for label, sec_text in sections:
        # Within a section, split at paragraphs and pack to target size
        paragraphs = [p.strip() for p in _PARA_RE.split(sec_text) if p.strip()]
        buf = ""
        for para in paragraphs:
            candidate = (buf + "\n\n" + para).strip() if buf else para
            if len(candidate) > _FILING_TARGET_CHARS and buf:
                # Flush current buffer as a chunk
                out.append(Chunk(
                    text=buf, ticker=ticker, asset_class=asset_class,
                    source=source, doc_type=doc_type, date_iso=date_iso,
                    chunk_index=cidx, extras={"section": label},
                ))
                cidx += 1
                buf = para
            else:
                buf = candidate
            # Safety: never exceed hard max
            if len(buf) > _FILING_MAX_CHARS:
                head, tail = _trim_to_sentence_boundary(buf, _FILING_MAX_CHARS)
                out.append(Chunk(
                    text=head, ticker=ticker, asset_class=asset_class,
                    source=source, doc_type=doc_type, date_iso=date_iso,
                    chunk_index=cidx, extras={"section": label},
                ))
                cidx += 1
                buf = tail
        if buf:
            out.append(Chunk(
                text=buf, ticker=ticker, asset_class=asset_class,
                source=source, doc_type=doc_type, date_iso=date_iso,
                chunk_index=cidx, extras={"section": label},
            ))
            cidx += 1
    return out


def chunk_transcript(text: str, *, ticker: str, asset_class: str,
                      date_iso: str, source: str = "earnings_call"
                      ) -> list[Chunk]:
    """Speaker-turn aware chunking for earnings call transcripts.

    Heuristic: lines starting with `Name (Title):` or `Name:` mark turns.
    Q&A pairs are kept together where possible.
    """
    if not text:
        return []
    # Split on speaker-turn boundaries
    turn_re = re.compile(r"\n(?=[A-Z][a-zA-Z'\.\- ]{1,60}(?:\([^)]*\))?:)")
    turns = [t.strip() for t in turn_re.split(text) if t.strip()]
    if not turns:
        turns = [text]
    out: list[Chunk] = []
    buf = ""
    cidx = 0
    for turn in turns:
        candidate = (buf + "\n\n" + turn).strip() if buf else turn
        if len(candidate) > _TRANSCRIPT_MAX_CHARS and buf:
            out.append(Chunk(
                text=buf, ticker=ticker, asset_class=asset_class,
                source=source, doc_type="transcript", date_iso=date_iso,
                chunk_index=cidx,
            ))
            cidx += 1
            buf = turn
        else:
            buf = candidate
    if buf:
        out.append(Chunk(
            text=buf, ticker=ticker, asset_class=asset_class,
            source=source, doc_type="transcript", date_iso=date_iso,
            chunk_index=cidx,
        ))
    return out
