"""Typed schemas for RAG records, chunks, and retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeRecord:
    """A single parsed knowledge record from a source dataset file."""

    record_id: str          # e.g. "master_dataset/1"
    source_file: str        # original relative filename, e.g. "1.json"
    dataset_name: str       # e.g. "master_dataset", "gita_csv", "counselling_csv"

    # Core text fields (always present; empty string if missing in source)
    paragraph: str = ""
    source_text: str = ""
    chapter_reference: str = ""
    situation: str = ""
    key_teaching: str = ""
    outcome: str = ""
    spiritual_summary: str = ""
    psychological_summary: str = ""

    # Structured fields preserved as-is (may be empty)
    characters_involved: list[str] = field(default_factory=list)
    mental_health_themes: list[str] = field(default_factory=list)
    mental_health_synonyms: dict[str, str] = field(default_factory=dict)
    spiritual_concepts: list[str] = field(default_factory=list)

    # Extra fields not in the standard schema
    extra: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return True when the record has no usable text content."""
        return not any([
            self.paragraph, self.source_text, self.situation,
            self.key_teaching, self.spiritual_summary, self.psychological_summary,
        ])


@dataclass
class TextChunk:
    """A chunk derived from a KnowledgeRecord, ready for embedding."""

    chunk_id: str           # globally unique, e.g. "master_dataset/1#0"
    record_id: str          # parent record
    source_file: str        # for display
    dataset_name: str
    text: str               # the text to embed

    # Metadata kept for prompt injection and provenance
    source_text: str = ""
    chapter_reference: str = ""
    characters_involved: list[str] = field(default_factory=list)
    mental_health_themes: list[str] = field(default_factory=list)
    spiritual_concepts: list[str] = field(default_factory=list)
    spiritual_summary: str = ""
    psychological_summary: str = ""

    chunk_index: int = 0    # index within parent record


@dataclass
class RetrievalResult:
    """A single retrieved chunk with similarity score and provenance."""

    chunk_id: str
    record_id: str
    source_file: str
    dataset_name: str
    text: str               # the retrieved passage text
    score: float            # cosine similarity in [0, 1]

    # Optional metadata for prompt construction
    source_text: str = ""
    chapter_reference: str = ""
    characters_involved: list[str] = field(default_factory=list)
    mental_health_themes: list[str] = field(default_factory=list)
    spiritual_concepts: list[str] = field(default_factory=list)
    spiritual_summary: str = ""
    psychological_summary: str = ""
