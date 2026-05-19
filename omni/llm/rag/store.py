"""Qdrant local-mode persistent vector store.

We use `qdrant-client` in **local mode** (no Docker, no server). Storage
lives on disk under `data/rag/qdrant/`. This is the recommended setup for
single-process Streamlit apps with <10M vectors — Qdrant docs explicitly
support it.

One collection per `(asset_class, doc_type)` combination — e.g.
`india_news`, `us_filings_10k`. This makes filter+search blazing fast
because queries never scan documents from the wrong asset class.

Vector dim = 2048 (Nemotron VL 1B v2). Distance = cosine.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from omni.llm.rag.chunker import Chunk

_log = logging.getLogger("omni.llm.rag.store")

# Defer qdrant import to method calls — keeps module import cheap and lets
# callers test the rest of the package even if qdrant isn't installed.
_QDRANT_DIM = 2048
_DEFAULT_PATH = "data/rag/qdrant"
_BATCH_UPSERT = 256


def _coll_name(asset_class: str, doc_type: str) -> str:
    """Derive a stable collection name. Lowercase + underscore — Qdrant
    accepts most strings but we keep it boring for safety."""
    ac = (asset_class or "unknown").lower().strip().replace(" ", "_")
    dt = (doc_type or "unknown").lower().strip().replace(" ", "_")
    return f"{ac}__{dt}"


@dataclass
class StoreStats:
    collections: dict[str, int]
    total_points: int
    storage_path: str


class RAGStore:
    """Thin wrapper around `qdrant_client.QdrantClient` in local mode.

    Thread-safe by virtue of qdrant-client's internal locks; we only add
    a creation lock so two threads don't race to create the same collection.
    """

    def __init__(self, path: str | Path = _DEFAULT_PATH,
                  vector_size: int = _QDRANT_DIM):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.vector_size = vector_size
        self._client = None
        self._known_collections: set[str] = set()
        self._coll_lock = threading.Lock()

    @property
    def client(self):
        """Lazy-instantiate qdrant client; idempotent.

        Phase 24.16 — if `OMNI_QDRANT_URL` is set in the environment we
        connect to a real Qdrant server (see deploy/docker-compose.qdrant.yml)
        instead of opening the local file store. This is the migration
        path the auditor flagged: planned, not forced.
        """
        if self._client is None:
            try:
                from qdrant_client import QdrantClient  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "qdrant-client not installed. Run "
                    "`pip install qdrant-client`."
                ) from e
            import os
            url = os.environ.get("OMNI_QDRANT_URL")
            api_key = os.environ.get("OMNI_QDRANT_API_KEY")
            if url:
                # Server mode — production path
                _log.info("qdrant: connecting to server at %s", url)
                self._client = QdrantClient(
                    url=url,
                    api_key=api_key,
                    timeout=float(os.environ.get(
                        "OMNI_QDRANT_TIMEOUT_SECONDS", "10")),
                )
                # Surface connection failures to the degradation monitor
                # (matches the embedder/reranker pattern).
                try:
                    self._client.get_collections()
                except Exception as e:  # noqa: BLE001
                    try:
                        from omni.llm.rag.degradation import monitor
                        monitor.record(
                            component="qdrant",
                            severity="error",
                            event="server_unreachable",
                            details={
                                "url": url,
                                "error_type": type(e).__name__,
                                "error_message": str(e)[:300],
                            },
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    raise
            else:
                # Local file mode — original dev path
                self._client = QdrantClient(path=str(self.path))
            # Pre-cache known collections
            try:
                cols = self._client.get_collections().collections or []
                self._known_collections = {c.name for c in cols}
            except Exception:
                self._known_collections = set()
        return self._client

    def _ensure_collection(self, name: str) -> None:
        if name in self._known_collections:
            return
        with self._coll_lock:
            if name in self._known_collections:
                return
            from qdrant_client.http import models as rest  # type: ignore
            try:
                self.client.get_collection(name)
            except Exception:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=rest.VectorParams(
                        size=self.vector_size,
                        distance=rest.Distance.COSINE,
                    ),
                )
                _log.info("created qdrant collection: %s", name)
            self._known_collections.add(name)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Upsert chunks + their embeddings into the appropriate collections.

        chunks[i] must correspond to vectors[i, :]. Vectors should already
        be L2-normalised (Nemotron output is). Zero vectors are skipped
        (failed embeddings).

        Returns the number of points actually written.
        """
        if not chunks:
            return 0
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"vectors shape {vectors.shape} mismatches chunks {len(chunks)}"
            )
        if vectors.shape[1] != self.vector_size:
            raise ValueError(
                f"vector dim {vectors.shape[1]} != configured {self.vector_size}"
            )

        from qdrant_client.http import models as rest  # type: ignore

        # Bucket by collection
        buckets: dict[str, list[tuple[Chunk, np.ndarray]]] = {}
        for ch, vec in zip(chunks, vectors):
            if float(np.linalg.norm(vec)) < 1e-9:
                continue  # skip failed embeddings
            coll = _coll_name(ch.asset_class, ch.doc_type)
            buckets.setdefault(coll, []).append((ch, vec))

        n_written = 0
        for coll, items in buckets.items():
            self._ensure_collection(coll)
            # Upload in batches
            for i in range(0, len(items), _BATCH_UPSERT):
                batch = items[i : i + _BATCH_UPSERT]
                points = []
                for ch, vec in batch:
                    payload = {
                        "ticker": ch.ticker,
                        "asset_class": ch.asset_class,
                        "source": ch.source,
                        "doc_type": ch.doc_type,
                        "date": ch.date_iso,
                        "chunk_index": ch.chunk_index,
                        "text": ch.text,
                    }
                    payload.update(ch.extras or {})
                    points.append(rest.PointStruct(
                        id=int(ch.stable_id()) % (2 ** 63 - 1),  # qdrant prefers <2^63
                        vector=vec.tolist(),
                        payload=payload,
                    ))
                self.client.upsert(collection_name=coll, points=points, wait=False)
                n_written += len(points)
        return n_written

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, *, asset_class: str, doc_type: str,
                query_vec: np.ndarray, ticker: str | None = None,
                date_from: str | None = None, date_to: str | None = None,
                top_k: int = 50) -> list[dict]:
        """Search a single (asset_class, doc_type) collection.

        Filters by ticker + date range when provided. Returns a list of
        dicts with `score`, `text`, plus all payload fields.
        """
        coll = _coll_name(asset_class, doc_type)
        if coll not in self._known_collections:
            try:
                self._ensure_collection(coll)
            except Exception:
                return []
        from qdrant_client.http import models as rest  # type: ignore

        must: list = []
        if ticker:
            must.append(rest.FieldCondition(
                key="ticker", match=rest.MatchValue(value=ticker.upper()),
            ))
        if date_from or date_to:
            # qdrant 1.10+ has DatetimeRange that accepts ISO date strings.
            # Range (numeric) would reject "2026-04-27" with a pydantic error.
            try:
                must.append(rest.FieldCondition(
                    key="date",
                    range=rest.DatetimeRange(gte=date_from, lte=date_to),
                ))
            except (AttributeError, TypeError):
                # Older qdrant — fall back to no date filter rather than fail
                _log.debug("DatetimeRange unavailable; skipping date filter")
        flt = rest.Filter(must=must) if must else None

        try:
            # qdrant-client >= 1.10 prefers `query_points` over the older
            # `search` method (which is deprecated). The shape of the
            # response is also slightly different: `.points` instead of a
            # bare list of ScoredPoint.
            resp = self.client.query_points(
                collection_name=coll,
                query=query_vec.tolist(),
                query_filter=flt,
                limit=top_k,
                with_payload=True,
            )
            hits = getattr(resp, "points", resp) or []
        except Exception as e:
            _log.warning("qdrant query failed on %s: %s", coll, e)
            return []

        out: list[dict] = []
        for h in hits:
            d = dict(h.payload or {})
            d["score"] = float(h.score)
            d["_qdrant_id"] = h.id
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def stats(self) -> StoreStats:
        """Return per-collection point counts and total."""
        try:
            cols = self.client.get_collections().collections or []
        except Exception:
            return StoreStats(collections={}, total_points=0, storage_path=str(self.path))
        out: dict[str, int] = {}
        total = 0
        for c in cols:
            try:
                info = self.client.get_collection(c.name)
                n = int(getattr(info, "points_count", 0) or 0)
            except Exception:
                n = 0
            out[c.name] = n
            total += n
        return StoreStats(collections=out, total_points=total, storage_path=str(self.path))

    def purge_older_than(self, days: int) -> int:
        """Drop chunks with `date` < (today - days). Returns approximate
        count of points removed (qdrant doesn't always report)."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
        from qdrant_client.http import models as rest  # type: ignore
        n = 0
        try:
            cols = self.client.get_collections().collections or []
        except Exception:
            return 0
        for c in cols:
            try:
                flt = rest.Filter(must=[rest.FieldCondition(
                    key="date", range=rest.DatetimeRange(lt=cutoff),
                )])
            except (AttributeError, TypeError):
                # Older qdrant — skip purge silently (better than crashing
                # the dev tab button)
                continue
            try:
                self.client.delete(collection_name=c.name,
                                    points_selector=rest.FilterSelector(filter=flt))
                n += 1  # we don't get exact count back
            except Exception as e:
                _log.warning("purge failed on %s: %s", c.name, e)
        return n

    def collection_name(self, asset_class: str, doc_type: str) -> str:
        """Public accessor for the naming convention."""
        return _coll_name(asset_class, doc_type)
