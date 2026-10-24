"""Persistent vector store: wraps ChromaDB for indexing and retrieval.

Stores embeddings, chunk text, and metadata. Persists between runs.
Tracks a manifest so stale indexes can be detected.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.rag.embeddings import EmbeddingModel
from src.rag.exceptions import IndexCorruptError, IndexNotFoundError
from src.rag.schemas import TextChunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "counselling_rag"
_MANIFEST_FILENAME = "rag_manifest.json"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _compute_manifest_hash(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    source_fingerprints: dict[str, str],
) -> str:
    payload = json.dumps(
        {
            "model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "sources": source_fingerprints,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _write_manifest(
    index_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    source_fingerprints: dict[str, str],
    chunk_count: int,
) -> None:
    manifest = {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "source_fingerprints": source_fingerprints,
        "chunk_count": chunk_count,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hash": _compute_manifest_hash(
            embedding_model, chunk_size, chunk_overlap, source_fingerprints
        ),
    }
    manifest_path = index_dir / _MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("[RAG Indexer] Manifest written to %s", manifest_path)


def _read_manifest(index_dir: Path) -> dict[str, Any] | None:
    manifest_path = index_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[RAG Indexer] Could not read manifest: %s", exc)
        return None


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """Persistent ChromaDB-backed vector store for TextChunks."""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self._client: Any = None
        self._collection: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import chromadb  # noqa: PLC0415

                self.index_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(self.index_dir))
                logger.info("[RAG Indexer] ChromaDB client initialised at %s", self.index_dir)
            except ImportError as exc:
                raise IndexCorruptError(
                    "chromadb is not installed. Run: pip install chromadb"
                ) from exc
        return self._client

    def _get_collection(self, create: bool = False) -> Any:
        if self._collection is None:
            client = self._get_client()
            if create:
                self._collection = client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                try:
                    self._collection = client.get_collection(_COLLECTION_NAME)
                except Exception as exc:
                    raise IndexNotFoundError(
                        f"Collection '{_COLLECTION_NAME}' not found in {self.index_dir}. "
                        "Build the index first: python -m src.rag.indexer"
                    ) from exc
        return self._collection

    def exists(self) -> bool:
        """Return True if the index directory and manifest both exist."""
        return (self.index_dir / _MANIFEST_FILENAME).exists()

    def manifest(self) -> dict[str, Any] | None:
        return _read_manifest(self.index_dir)

    def is_stale(
        self,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        source_fingerprints: dict[str, str],
    ) -> bool:
        """Return True when current settings differ from the stored manifest."""
        m = _read_manifest(self.index_dir)
        if m is None:
            return True
        expected = _compute_manifest_hash(
            embedding_model, chunk_size, chunk_overlap, source_fingerprints
        )
        return m.get("hash") != expected

    def count(self) -> int:
        """Return the number of chunks in the index (0 if not initialised)."""
        try:
            col = self._get_collection(create=False)
            return col.count()
        except (IndexNotFoundError, Exception):
            return 0

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build(
        self,
        chunks: list[TextChunk],
        embedding_model: EmbeddingModel,
        batch_size: int,
        source_fingerprints: dict[str, str],
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """Clear existing index and rebuild from *chunks*."""
        client = self._get_client()

        # Delete then recreate collection (safe full rebuild)
        try:
            client.delete_collection(_COLLECTION_NAME)
            logger.info("[RAG Indexer] Existing collection deleted for rebuild.")
        except Exception:
            pass  # Does not exist yet; that is fine.
        self._collection = None

        col = self._get_collection(create=True)

        texts = [c.text for c in chunks]
        logger.info(
            "[RAG Indexer] Embedding %d chunks (batch_size=%d) …", len(chunks), batch_size
        )
        embeddings = embedding_model.encode(
            texts, batch_size=batch_size, normalize=True, show_progress=True
        )

        # Upsert in batches of batch_size
        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_chunks = chunks[start:end]
            batch_embs = embeddings[start:end]

            ids = [c.chunk_id for c in batch_chunks]
            docs = [c.text for c in batch_chunks]
            metas = [_chunk_metadata(c) for c in batch_chunks]
            embeds = [e.tolist() for e in batch_embs]

            col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

        logger.info("[RAG Indexer] %d chunks stored in ChromaDB.", total)

        # Write manifest
        _write_manifest(
            self.index_dir,
            embedding_model.model_name,
            chunk_size,
            chunk_overlap,
            source_fingerprints,
            total,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return top_k nearest results as a list of dicts with keys:
        id, document, metadata, distance.
        """
        col = self._get_collection(create=False)
        try:
            results = col.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, col.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("[RAG Indexer] Query failed: %s", exc)
            return []

        out: list[dict[str, Any]] = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for cid, doc, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            out.append({
                "id": cid,
                "document": doc,
                "metadata": meta or {},
                "distance": float(dist),
            })
        return out


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _chunk_metadata(chunk: TextChunk) -> dict[str, Any]:
    """Serialise TextChunk metadata as flat Chroma-compatible dict."""
    return {
        "record_id": chunk.record_id,
        "source_file": chunk.source_file,
        "dataset_name": chunk.dataset_name,
        "source_text": chunk.source_text,
        "chapter_reference": chunk.chapter_reference,
        "characters_involved": json.dumps(chunk.characters_involved),
        "mental_health_themes": json.dumps(chunk.mental_health_themes),
        "spiritual_concepts": json.dumps(chunk.spiritual_concepts),
        "spiritual_summary": chunk.spiritual_summary[:500],
        "psychological_summary": chunk.psychological_summary[:500],
        "chunk_index": chunk.chunk_index,
    }
