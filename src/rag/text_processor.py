"""Text processor: converts KnowledgeRecords into embeddable TextChunks.

Each record produces one or more chunks. Short records are kept whole;
long records are split with configurable size and overlap.
"""

from __future__ import annotations

import logging
from typing import Iterator

from src.rag.schemas import KnowledgeRecord, TextChunk

logger = logging.getLogger(__name__)

# Default max chars for a single chunk before splitting
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Record → searchable text
# ---------------------------------------------------------------------------

def _join_nonempty(*parts: str, sep: str = "\n") -> str:
    return sep.join(p.strip() for p in parts if p and p.strip())


def record_to_text(record: KnowledgeRecord) -> str:
    """Build the full searchable text representation of a KnowledgeRecord.

    Fields are labelled so that semantic search can match conceptual queries
    against specific fields (e.g. "Psychological Summary: ...").
    """
    parts: list[str] = []

    if record.source_text:
        parts.append(f"Source: {record.source_text}")
    if record.chapter_reference:
        parts.append(f"Reference: {record.chapter_reference}")
    if record.characters_involved:
        parts.append(f"Characters: {', '.join(record.characters_involved)}")

    if record.paragraph:
        parts.append(f"Passage:\n{record.paragraph}")
    if record.situation:
        parts.append(f"Situation: {record.situation}")
    if record.key_teaching:
        parts.append(f"Key Teaching: {record.key_teaching}")
    if record.outcome:
        parts.append(f"Outcome: {record.outcome}")

    if record.mental_health_themes:
        themes = ", ".join(record.mental_health_themes)
        parts.append(f"Mental Health Themes: {themes}")
    if record.mental_health_synonyms:
        syns = "; ".join(
            f"{k}: {v}" for k, v in record.mental_health_synonyms.items() if v
        )
        if syns:
            parts.append(f"Theme Synonyms: {syns}")
    if record.spiritual_concepts:
        concepts = "; ".join(record.spiritual_concepts)
        parts.append(f"Spiritual Concepts: {concepts}")

    if record.spiritual_summary:
        parts.append(f"Spiritual Summary:\n{record.spiritual_summary}")
    if record.psychological_summary:
        parts.append(f"Psychological Summary:\n{record.psychological_summary}")

    return _join_nonempty(*parts, sep="\n\n")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split *text* into overlapping character chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def records_to_chunks(
    records: list[KnowledgeRecord],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Convert a list of KnowledgeRecords to a flat list of TextChunks."""
    chunks: list[TextChunk] = []
    for record in records:
        for chunk in record_to_chunks(record, chunk_size, chunk_overlap):
            chunks.append(chunk)
    logger.info("[RAG TextProcessor] %d records → %d chunks", len(records), len(chunks))
    return chunks


def record_to_chunks(
    record: KnowledgeRecord,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterator[TextChunk]:
    """Yield one or more TextChunks from a single KnowledgeRecord."""
    full_text = record_to_text(record)
    if not full_text.strip():
        return

    raw_chunks = _split_text(full_text, chunk_size, chunk_overlap)
    for i, text in enumerate(raw_chunks):
        yield TextChunk(
            chunk_id=f"{record.record_id}#{i}",
            record_id=record.record_id,
            source_file=record.source_file,
            dataset_name=record.dataset_name,
            text=text,
            source_text=record.source_text,
            chapter_reference=record.chapter_reference,
            characters_involved=record.characters_involved,
            mental_health_themes=record.mental_health_themes,
            spiritual_concepts=record.spiritual_concepts,
            spiritual_summary=record.spiritual_summary,
            psychological_summary=record.psychological_summary,
            chunk_index=i,
        )
