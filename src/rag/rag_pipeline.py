"""RAGPipeline: public facade that coordinates loading, indexing, and retrieval.

Usage
-----
# At application startup:
pipeline = RAGPipeline.from_config(cfg_dict, project_root=Path(...))

# In the chatbot's chat() method:
results = pipeline.retrieve(user_message, emotions=["Sadness"], need="grief")
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from src.rag.config import RAGConfig
from src.rag.embeddings import get_embedding_model
from src.rag.exceptions import (
    ConfigurationError,
    EmbeddingModelError,
    IndexNotFoundError,
    RAGError,
)
from src.rag.indexer import VectorStore
from src.rag.loader import (
    load_counselling_csv_dataset,
    load_gita_csv_dataset,
    load_json_dataset,
)
from src.rag.retriever import Retriever
from src.rag.schemas import RetrievalResult, TextChunk
from src.rag.text_processor import records_to_chunks

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Manages dataset loading, index lifecycle, and retrieval.

    Call ``from_config()`` to construct and initialise.
    Call ``retrieve()`` to query.
    """

    def __init__(self, cfg: RAGConfig, root: Path) -> None:
        self._cfg = cfg
        self._root = root
        self._retriever: Retriever | None = None
        self._store: VectorStore | None = None
        self._ready = False
        self._init_error: str | None = None
        self._stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        raw_rag_cfg: dict[str, Any],
        project_root: Path,
    ) -> "RAGPipeline":
        """Parse config, resolve paths, and initialise the pipeline.

        This method never raises — errors are recorded internally and surfaced
        via ``is_ready`` and ``init_error``.  This ensures the chatbot starts
        even when RAG initialisation fails.
        """
        try:
            cfg = RAGConfig.from_dict(raw_rag_cfg)
        except ConfigurationError as exc:
            logger.error("[RAG Pipeline] Configuration error: %s", exc)
            pipe = cls(RAGConfig(enabled=False), project_root)
            pipe._init_error = str(exc)
            return pipe

        cfg = cfg.resolve_paths(project_root)
        pipe = cls(cfg, project_root)

        if not cfg.enabled:
            logger.info("[RAG Pipeline] RAG is disabled via configuration.")
            return pipe

        try:
            pipe._initialise()
        except EmbeddingModelError as exc:
            logger.error("[RAG Pipeline] Embedding model failed: %s", exc)
            pipe._init_error = str(exc)
            if not cfg.fallback_to_llm:
                raise
        except IndexNotFoundError as exc:
            logger.warning("[RAG Pipeline] Index not found: %s", exc)
            pipe._init_error = str(exc)
        except Exception as exc:
            logger.error("[RAG Pipeline] Unexpected init error: %s", exc, exc_info=True)
            pipe._init_error = str(exc)
            if not cfg.fallback_to_llm:
                raise RAGError(f"RAG failed to initialise: {exc}") from exc

        return pipe

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self) -> None:
        cfg = self._cfg
        logger.info("[RAG Pipeline] Initialising …")

        # Embedding model (may trigger download on first run)
        embed_model = get_embedding_model(cfg.embedding_model, cfg.device)
        # Force model load so we fail early if unavailable
        _ = embed_model.dimension  # triggers load

        store = VectorStore(cfg.index_dir)
        self._store = store

        # Compute source fingerprints for staleness detection
        fingerprints = _source_fingerprints(cfg.dataset_paths)

        needs_build = (
            cfg.rebuild_index
            or not store.exists()
            or store.is_stale(
                cfg.embedding_model,
                cfg.chunk_size,
                cfg.chunk_overlap,
                fingerprints,
            )
        )

        if needs_build:
            if cfg.rebuild_index:
                logger.info("[RAG Pipeline] Forced rebuild requested.")
            elif not store.exists():
                logger.info("[RAG Pipeline] Index not found — building for the first time.")
            else:
                logger.info("[RAG Pipeline] Index is stale — rebuilding.")
            self._build_index(store, embed_model, fingerprints)
        else:
            logger.info(
                "[RAG Pipeline] Using existing index at %s (%d chunks).",
                cfg.index_dir, store.count(),
            )

        self._store = store
        self._retriever = Retriever(
            store=store,
            embedding_model=embed_model,
            top_k=cfg.top_k,
            similarity_threshold=cfg.similarity_threshold,
        )
        self._stats["indexed_chunks"] = store.count()
        self._ready = True
        logger.info("[RAG Pipeline] Ready. Indexed chunks: %d", store.count())

    def _build_index(
        self,
        store: VectorStore,
        embed_model: Any,
        fingerprints: dict[str, str],
    ) -> None:
        cfg = self._cfg
        logger.info("[RAG Pipeline] Loading datasets from %s …", cfg.dataset_paths)

        all_chunks: list[TextChunk] = []
        records_loaded = 0

        for dataset_path in cfg.dataset_paths:
            path = Path(dataset_path)
            if not path.exists():
                logger.warning("[RAG Pipeline] Dataset path not found: %s — skipping.", path)
                continue

            # Detect dataset type by path hint
            path_lower = str(path).lower()
            if "master dataset" in path_lower or path.suffix == "" or "json" in path_lower:
                records = list(load_json_dataset(path))
            elif "gita" in path_lower:
                records = list(load_gita_csv_dataset(path))
            elif "counselling" in path_lower:
                records = list(load_counselling_csv_dataset(path))
            else:
                # Heuristic: try JSON first
                records = list(load_json_dataset(path))

            logger.info(
                "[RAG Pipeline] Loaded %d records from %s", len(records), path
            )
            records_loaded += len(records)
            chunks = records_to_chunks(
                records,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            )
            all_chunks.extend(chunks)

        self._stats["records_loaded"] = records_loaded
        self._stats["total_chunks"] = len(all_chunks)

        if not all_chunks:
            logger.warning("[RAG Pipeline] No chunks produced — index will be empty.")

        store.build(
            chunks=all_chunks,
            embedding_model=embed_model,
            batch_size=cfg.batch_size,
            source_fingerprints=fingerprints,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready and self._cfg.enabled

    @property
    def init_error(self) -> str | None:
        return self._init_error

    @property
    def stats(self) -> dict[str, Any]:
        s = dict(self._stats)
        if self._store:
            s["indexed_chunks"] = self._store.count()
        s["enabled"] = self._cfg.enabled
        s["ready"] = self.is_ready
        s["embedding_model"] = self._cfg.embedding_model
        return s

    def retrieve(
        self,
        query: str,
        *,
        emotions: list[str] | None = None,
        need: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant passages for the given query.

        Returns an empty list when RAG is disabled, not ready, or when
        no relevant evidence is found.  Never raises (when fallback_to_llm=True).
        """
        if not self.is_ready:
            logger.debug("[RAG Pipeline] Not ready — returning empty results.")
            return []
        try:
            return self._retriever.retrieve(  # type: ignore[union-attr]
                query, emotions=emotions, need=need, top_k=top_k
            )
        except Exception as exc:
            if self._cfg.fallback_to_llm:
                logger.warning("[RAG Pipeline] Retrieval error (fallback active): %s", exc)
                return []
            raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_fingerprints(dataset_paths: list[str]) -> dict[str, str]:
    """Compute a fingerprint dict for all dataset paths (file count + mtime hash)."""
    result: dict[str, str] = {}
    for dpath in dataset_paths:
        p = Path(dpath)
        if not p.exists():
            result[dpath] = "missing"
            continue
        files = sorted(p.rglob("*"))
        content = "".join(
            f"{f.name}:{f.stat().st_mtime:.0f}:{f.stat().st_size}"
            for f in files if f.is_file()
        )
        result[dpath] = hashlib.md5(content.encode()).hexdigest()[:12]  # noqa: S324
    return result
