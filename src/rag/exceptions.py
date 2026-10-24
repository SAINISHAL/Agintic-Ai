"""Custom exception hierarchy for the RAG subsystem."""

from __future__ import annotations


class RAGError(Exception):
    """Base exception for all RAG errors."""


class DatasetLoadError(RAGError):
    """Raised when a dataset directory cannot be discovered or opened."""


class MalformedRecordError(RAGError):
    """Raised (and caught internally) for individual malformed JSON records.

    Callers should catch this to skip the record and continue.
    """


class IndexNotFoundError(RAGError):
    """Raised when the persistent index does not exist at the expected location."""


class IndexCorruptError(RAGError):
    """Raised when the index exists but cannot be loaded or is stale."""


class EmbeddingModelError(RAGError):
    """Raised when the embedding model cannot be initialised."""


class RetrievalError(RAGError):
    """Raised when retrieval fails for a reason other than an empty index."""


class ConfigurationError(RAGError):
    """Raised when RAG configuration is invalid."""
