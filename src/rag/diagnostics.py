"""Diagnostics module: runtime status and stats for the RAG subsystem."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.rag.rag_pipeline import RAGPipeline


def get_diagnostics(pipeline: "RAGPipeline") -> dict[str, Any]:
    """Return a diagnostics dict suitable for display in the Streamlit UI."""
    stats = pipeline.stats
    m = None
    if pipeline._store is not None:
        m = pipeline._store.manifest()
    return {
        "rag_enabled": stats.get("enabled", False),
        "rag_ready": stats.get("ready", False),
        "init_error": pipeline.init_error,
        "indexed_chunks": stats.get("indexed_chunks", 0),
        "records_loaded": stats.get("records_loaded", 0),
        "embedding_model": m.get("embedding_model", "?") if m else "?",
        "index_dir": str(pipeline._cfg.index_dir),
        "built_at": m.get("built_at", "?") if m else "?",
    }


def format_retrieval_sources(results: list[Any]) -> str:
    """Format retrieval results as a human-readable string for diagnostics."""
    if not results:
        return "(no sources retrieved)"
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        ref = r.chapter_reference or r.source_file or r.record_id
        src = r.source_text or r.dataset_name
        lines.append(f"  [{i}] {src} — {ref} (score: {r.score:.3f})")
    return "\n".join(lines)
