"""Retriever: queries the vector store for relevant passages.

Accepts user message and optional enrichment context (emotions, need).
Returns a list of RetrievalResult, sorted by similarity score.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.rag.embeddings import EmbeddingModel
from src.rag.exceptions import IndexNotFoundError, RetrievalError
from src.rag.indexer import VectorStore
from src.rag.schemas import RetrievalResult

logger = logging.getLogger(__name__)


def _cosine_from_distance(distance: float) -> float:
    """Convert ChromaDB cosine *distance* (1 - similarity) to similarity."""
    return max(0.0, min(1.0, 1.0 - distance))


def _deduplicate(results: list[RetrievalResult], max_k: int) -> list[RetrievalResult]:
    """Remove results with identical record_id (keep highest-scoring chunk)."""
    seen: set[str] = set()
    deduped: list[RetrievalResult] = []
    for r in results:
        if r.record_id not in seen:
            seen.add(r.record_id)
            deduped.append(r)
            if len(deduped) >= max_k:
                break
    return deduped


class Retriever:
    """Semantic retriever over the persistent vector index."""

    def __init__(
        self,
        store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> None:
        self._store = store
        self._embed = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        *,
        emotions: list[str] | None = None,
        need: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant passages for *query*.

        Args:
            query: The user's current message (primary signal).
            emotions: Detected emotion labels (optional, appended to query).
            need: Detected need/context (optional, appended to query).
            top_k: Override the instance-level default.

        Returns:
            List of RetrievalResult sorted by descending similarity score.
            Empty list when no relevant evidence is found or index is empty.
        """
        if not query or not query.strip():
            logger.debug("[RAG Retriever] Empty query — returning empty results.")
            return []

        k = top_k if top_k is not None else self.top_k

        # Build enriched query: user message + detected emotional context
        enriched_parts = [query.strip()]
        if emotions:
            enriched_parts.append("Emotions: " + ", ".join(emotions))
        if need:
            enriched_parts.append("Need: " + need)
        enriched_query = " | ".join(enriched_parts)

        t0 = time.perf_counter()
        try:
            query_vec = self._embed.encode_single(enriched_query, normalize=True)
            raw = self._store.query(query_vec.tolist(), top_k=k * 2)  # fetch extra for dedup
        except IndexNotFoundError:
            logger.warning("[RAG Retriever] Index not found — returning empty results.")
            return []
        except Exception as exc:
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[RAG Retriever] Query complete in %.1f ms, raw hits=%d",
            latency_ms, len(raw),
        )

        results: list[RetrievalResult] = []
        for hit in raw:
            score = _cosine_from_distance(hit["distance"])
            if self.similarity_threshold is not None and score < self.similarity_threshold:
                continue
            meta: dict[str, Any] = hit.get("metadata", {})
            results.append(
                RetrievalResult(
                    chunk_id=hit["id"],
                    record_id=meta.get("record_id", hit["id"]),
                    source_file=meta.get("source_file", ""),
                    dataset_name=meta.get("dataset_name", ""),
                    text=hit.get("document", ""),
                    score=score,
                    source_text=meta.get("source_text", ""),
                    chapter_reference=meta.get("chapter_reference", ""),
                    characters_involved=_safe_json_list(meta.get("characters_involved")),
                    mental_health_themes=_safe_json_list(meta.get("mental_health_themes")),
                    spiritual_concepts=_safe_json_list(meta.get("spiritual_concepts")),
                    spiritual_summary=meta.get("spiritual_summary", ""),
                    psychological_summary=meta.get("psychological_summary", ""),
                )
            )

        # Sort by score descending, deduplicate by record
        results.sort(key=lambda r: r.score, reverse=True)
        results = _deduplicate(results, k)

        logger.info(
            "[RAG Retriever] Returning %d results (threshold=%s, top_k=%d).",
            len(results), self.similarity_threshold, k,
        )
        return results


def _safe_json_list(value: Any) -> list[str]:
    """Safely decode a JSON string or return an empty list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return []
