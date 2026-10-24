"""Embedding wrapper: loads a sentence-transformer model and encodes text.

One singleton instance is created per (model_name, device) combination to
avoid duplicate model loads. Thread-safe for read operations.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from src.rag.exceptions import EmbeddingModelError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_instances: dict[str, "EmbeddingModel"] = {}


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer with batched encoding."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: "SentenceTransformer | None" = None

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info(
                "[RAG Embeddings] Loading embedding model '%s' on device '%s' …",
                self.model_name, self.device,
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("[RAG Embeddings] Model loaded. Embedding dim: %d", self.dimension)
        except ImportError as exc:
            raise EmbeddingModelError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            raise EmbeddingModelError(
                f"Failed to load embedding model '{self.model_name}': {exc}"
            ) from exc

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            self._load()
        return self._model  # type: ignore[return-value]

    @property
    def dimension(self) -> int:
        if self._model is None:
            self._load()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of strings and return a float32 numpy array (N, D)."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single string; returns 1-D float32 array."""
        return self.encode([text], normalize=normalize)[0]


def get_embedding_model(model_name: str, device: str = "auto") -> EmbeddingModel:
    """Return a (possibly cached) EmbeddingModel instance.

    Device resolution: "auto" → "cuda" if torch CUDA is available else "cpu".
    """
    resolved_device = _resolve_device(device)
    key = f"{model_name}::{resolved_device}"
    with _lock:
        if key not in _instances:
            _instances[key] = EmbeddingModel(model_name, resolved_device)
        return _instances[key]


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch  # noqa: PLC0415

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device
