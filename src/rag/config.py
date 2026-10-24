"""RAG configuration: parse and validate the ``rag`` section of chatbot.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.rag.exceptions import ConfigurationError


@dataclass
class RAGConfig:
    """Holds all RAG runtime settings with validated defaults."""

    enabled: bool = True

    # Dataset source directories (relative to project root or absolute)
    dataset_paths: list[str] = field(default_factory=lambda: [
        "data/Master Dataset - old/Master Dataset"
    ])

    # Where to persist the Chroma vector index
    index_dir: str = "data/rag_index"

    # Sentence-transformer embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # "auto", "cpu", or "cuda"
    device: str = "auto"

    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float | None = None   # None = return all top_k

    # Text chunking (characters)
    chunk_size: int = 1200
    chunk_overlap: int = 100

    # Embedding batch size during indexing
    batch_size: int = 64

    # Whether to include source metadata in chat output
    return_sources: bool = True

    # Fall back to LLM-only when retrieval fails
    fallback_to_llm: bool = True

    # Force a full index rebuild on next startup
    rebuild_index: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Path | None = None) -> "RAGConfig":
        """Parse and validate a config dict (the ``rag`` sub-section)."""
        cfg = cls()

        # Simple scalars
        bool_fields = ("enabled", "return_sources", "fallback_to_llm", "rebuild_index")
        int_fields  = ("top_k", "chunk_size", "chunk_overlap", "batch_size")
        str_fields  = ("index_dir", "embedding_model", "device")

        for f in bool_fields:
            if f in data:
                cfg.__dict__[f] = bool(data[f])
        for f in int_fields:
            if f in data:
                v = int(data[f])
                if v <= 0:
                    raise ConfigurationError(f"RAG config: '{f}' must be > 0, got {v}")
                cfg.__dict__[f] = v
        for f in str_fields:
            if f in data and data[f] is not None:
                cfg.__dict__[f] = str(data[f])

        if "similarity_threshold" in data and data["similarity_threshold"] is not None:
            t = float(data["similarity_threshold"])
            if not (0.0 <= t <= 1.0):
                raise ConfigurationError(
                    f"RAG config: similarity_threshold must be in [0, 1], got {t}"
                )
            cfg.similarity_threshold = t

        if "dataset_paths" in data and data["dataset_paths"]:
            cfg.dataset_paths = [str(p) for p in data["dataset_paths"]]

        return cfg

    def resolve_paths(self, root: Path) -> "RAGConfig":
        """Return a copy with all paths resolved relative to *root*."""
        import copy
        c = copy.copy(self)
        c.dataset_paths = [
            str((root / p).resolve()) if not Path(p).is_absolute() else p
            for p in self.dataset_paths
        ]
        c.index_dir = str(
            (root / self.index_dir).resolve()
            if not Path(self.index_dir).is_absolute()
            else self.index_dir
        )
        return c
