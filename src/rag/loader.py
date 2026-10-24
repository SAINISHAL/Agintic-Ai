"""Dataset loader for the Master Dataset (JSON files) and CSV sources.

Discovers and parses knowledge records from configured source directories.
Handles inconsistent schemas, missing fields, malformed files, and duplicates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from src.rag.exceptions import DatasetLoadError, MalformedRecordError
from src.rag.schemas import KnowledgeRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _str_or_empty(value: Any) -> str:
    """Coerce a value to a non-None string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    return str(value).strip()


def _list_of_strings(value: Any) -> list[str]:
    """Normalise a field that should be a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    return []


def _parse_synonyms(value: Any) -> dict[str, str]:
    """Parse the Mental_Health_Theme_Synonyms field."""
    if isinstance(value, dict):
        return {str(k): _str_or_empty(v) for k, v in value.items()}
    return {}


def _parse_spiritual_concepts(value: Any) -> list[str]:
    """Parse the Spiritual_Psychological_Concept_Indian field."""
    if isinstance(value, list):
        concepts = []
        for item in value:
            if isinstance(item, dict):
                concepts.append(_str_or_empty(item.get("Concept") or item.get("concept", "")))
            elif isinstance(item, str):
                concepts.append(item.strip())
        return [c for c in concepts if c]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _file_fingerprint(path: Path) -> str:
    """Return a stable hex fingerprint for a file (SHA-256 of content)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Master-Dataset JSON loader (one record per file)
# ---------------------------------------------------------------------------

_PARAGRAPH_KEYS = ("Paragraph", "paragraph")
_SOURCE_KEYS = ("Source_Text", "source_text", "Source")
_CHAPTER_KEYS = ("Chapter/Canto/Verse_Reference", "chapter_reference", "Chapter")
_CHARACTERS_KEYS = ("Characters_Involved", "characters_involved")
_SITUATION_KEYS = ("Situation/Problem_Faced", "Situation_Problem_Faced", "situation")
_TEACHING_KEYS = ("Key_Teaching/Action_Taken", "Key_Teaching_Action_Taken", "key_teaching")
_OUTCOME_KEYS = ("Outcome/Resolution", "outcome_resolution", "outcome")
_MH_THEMES_KEYS = ("Mental_Health_Themes", "mental_health_themes")
_MH_SYNONYMS_KEYS = ("Mental_Health_Theme_Synonyms", "Mental_Health_Themes_Synonyms", "mental_health_synonyms")
_SPIRITUAL_CONCEPTS_KEYS = ("Spiritual_Psychological_Concept_Indian",)
_SPIRITUAL_SUMMARY_KEYS = ("Brief_Summary_Spiritual", "spiritual_summary")
_PSYCH_SUMMARY_KEYS = ("Brief_Summary_Psychological", "psychological_summary")


def _first(d: dict[str, Any], *keys: str) -> Any:
    """Return the value of the first matching key in dict *d*."""
    for k in keys:
        if k in d:
            return d[k]
    return None


def _parse_json_record(data: dict[str, Any], record_id: str, source_file: str) -> KnowledgeRecord:
    """Convert a raw JSON dict to a KnowledgeRecord."""
    extra = {
        k: v for k, v in data.items()
        if k not in {
            *_PARAGRAPH_KEYS, *_SOURCE_KEYS, *_CHAPTER_KEYS, *_CHARACTERS_KEYS,
            *_SITUATION_KEYS, *_TEACHING_KEYS, *_OUTCOME_KEYS, *_MH_THEMES_KEYS,
            *_MH_SYNONYMS_KEYS, *_SPIRITUAL_CONCEPTS_KEYS, *_SPIRITUAL_SUMMARY_KEYS,
            *_PSYCH_SUMMARY_KEYS,
        }
    }
    return KnowledgeRecord(
        record_id=record_id,
        source_file=source_file,
        dataset_name="master_dataset",
        paragraph=_str_or_empty(_first(data, *_PARAGRAPH_KEYS)),
        source_text=_str_or_empty(_first(data, *_SOURCE_KEYS)),
        chapter_reference=_str_or_empty(_first(data, *_CHAPTER_KEYS)),
        situation=_str_or_empty(_first(data, *_SITUATION_KEYS)),
        key_teaching=_str_or_empty(_first(data, *_TEACHING_KEYS)),
        outcome=_str_or_empty(_first(data, *_OUTCOME_KEYS)),
        spiritual_summary=_str_or_empty(_first(data, *_SPIRITUAL_SUMMARY_KEYS)),
        psychological_summary=_str_or_empty(_first(data, *_PSYCH_SUMMARY_KEYS)),
        characters_involved=_list_of_strings(_first(data, *_CHARACTERS_KEYS)),
        mental_health_themes=_list_of_strings(_first(data, *_MH_THEMES_KEYS)),
        mental_health_synonyms=_parse_synonyms(_first(data, *_MH_SYNONYMS_KEYS)),
        spiritual_concepts=_parse_spiritual_concepts(_first(data, *_SPIRITUAL_CONCEPTS_KEYS)),
        extra=extra,
    )


def load_json_dataset(directory: str | Path) -> Iterator[KnowledgeRecord]:
    """Yield KnowledgeRecords from all JSON files in *directory* (non-recursive first level,
    then recurse one level for the "Master Dataset" sub-folder pattern).

    Stats are logged; individual bad files are skipped, not raised.
    """
    root = Path(directory)
    if not root.exists():
        raise DatasetLoadError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise DatasetLoadError(f"Dataset path is not a directory: {root}")

    json_files = sorted(root.rglob("*.json"), key=lambda p: (p.parent, _numeric_sort_key(p.stem)))
    discovered = len(json_files)
    loaded = skipped = failed = 0
    seen_hashes: set[str] = set()

    logger.info("[RAG Loader] Discovering JSON files in %s — found %d", root, discovered)

    for json_path in json_files:
        rel = json_path.relative_to(root)
        try:
            text = json_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("[RAG Loader] Cannot read %s: %s", rel, exc)
            failed += 1
            continue

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("[RAG Loader] Malformed JSON in %s: %s", rel, exc)
            failed += 1
            continue

        # Support both a single object and a list of objects
        records_raw: list[dict[str, Any]] = []
        if isinstance(data, dict):
            records_raw = [data]
        elif isinstance(data, list):
            records_raw = [r for r in data if isinstance(r, dict)]
        else:
            logger.warning("[RAG Loader] Unexpected JSON root type in %s: %s", rel, type(data).__name__)
            skipped += 1
            continue

        for idx, raw in enumerate(records_raw):
            # Deduplicate by content fingerprint
            content_hash = hashlib.md5(  # noqa: S324 (not security-critical)
                json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            if content_hash in seen_hashes:
                logger.debug("[RAG Loader] Duplicate record skipped in %s[%d]", rel, idx)
                skipped += 1
                continue
            seen_hashes.add(content_hash)

            suffix = f"#{idx}" if len(records_raw) > 1 else ""
            record_id = f"master_dataset/{rel.stem}{suffix}"
            try:
                record = _parse_json_record(raw, record_id=record_id, source_file=str(rel))
                if record.is_empty():
                    logger.debug("[RAG Loader] Empty record in %s%s, skipping.", rel, suffix)
                    skipped += 1
                    continue
                loaded += 1
                yield record
            except MalformedRecordError as exc:
                logger.warning("[RAG Loader] Bad record in %s%s: %s", rel, suffix, exc)
                failed += 1

    logger.info(
        "[RAG Loader] JSON dataset complete. discovered=%d loaded=%d skipped=%d failed=%d",
        discovered, loaded, skipped, failed,
    )


def _numeric_sort_key(stem: str) -> tuple[int, str]:
    """Sort '1', '2', ..., '10' numerically, others lexicographically."""
    try:
        return (int(stem), "")
    except ValueError:
        return (999_999, stem)


# ---------------------------------------------------------------------------
# Gita CSV loader
# ---------------------------------------------------------------------------

def load_gita_csv_dataset(directory: str | Path) -> Iterator[KnowledgeRecord]:
    """Yield KnowledgeRecords from Bhagavad Gita Chapter CSV files.

    Expected columns (from actual inspection): question, answer, chapter (inferred from filename).
    """
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        logger.warning("[RAG Loader] Gita CSV directory not found: %s — skipping.", root)
        return

    csv_files = sorted(root.glob("Chapter_*.csv"))
    total_loaded = 0
    for csv_path in csv_files:
        chapter_name = csv_path.stem.replace("_", " ")  # e.g. "Chapter 2 QA"
        try:
            with csv_path.open(encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.DictReader(fh)
                for idx, row in enumerate(reader):
                    q = (row.get("question") or row.get("Question") or "").strip()
                    a = (row.get("answer") or row.get("Answer") or "").strip()
                    if not q and not a:
                        continue
                    record_id = f"gita_csv/{csv_path.stem}/{idx}"
                    record = KnowledgeRecord(
                        record_id=record_id,
                        source_file=str(csv_path.name),
                        dataset_name="gita_csv",
                        paragraph=f"Q: {q}\nA: {a}" if q else a,
                        source_text="Bhagavad Gita",
                        chapter_reference=chapter_name,
                        situation=q,
                        key_teaching=a,
                        mental_health_themes=["spiritual guidance"],
                    )
                    total_loaded += 1
                    yield record
        except (OSError, csv.Error) as exc:
            logger.warning("[RAG Loader] Cannot read Gita CSV %s: %s", csv_path.name, exc)

    logger.info("[RAG Loader] Gita CSV dataset loaded %d records.", total_loaded)


# ---------------------------------------------------------------------------
# Counselling CSV loader
# ---------------------------------------------------------------------------

def load_counselling_csv_dataset(directory: str | Path) -> Iterator[KnowledgeRecord]:
    """Yield KnowledgeRecords from counselling conversation CSV files."""
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        logger.warning("[RAG Loader] Counselling CSV directory not found: %s — skipping.", root)
        return

    csv_files = list(root.glob("*.csv"))
    total_loaded = 0
    for csv_path in csv_files:
        try:
            with csv_path.open(encoding="utf-8", errors="replace", newline="") as fh:
                reader = csv.DictReader(fh)
                for idx, row in enumerate(reader):
                    # Try common column name variants
                    context = (
                        row.get("Context") or row.get("context") or
                        row.get("question") or row.get("input") or ""
                    ).strip()
                    response = (
                        row.get("Response") or row.get("response") or
                        row.get("answer") or row.get("output") or ""
                    ).strip()
                    if not context and not response:
                        continue
                    record_id = f"counselling_csv/{csv_path.stem}/{idx}"
                    record = KnowledgeRecord(
                        record_id=record_id,
                        source_file=csv_path.name,
                        dataset_name="counselling_csv",
                        paragraph=f"User: {context}\nCounsellor: {response}" if context else response,
                        source_text="Counselling Dataset",
                        situation=context,
                        key_teaching=response,
                        mental_health_themes=["counselling", "emotional support"],
                    )
                    total_loaded += 1
                    yield record
        except (OSError, csv.Error) as exc:
            logger.warning("[RAG Loader] Cannot read counselling CSV %s: %s", csv_path.name, exc)

    logger.info("[RAG Loader] Counselling CSV dataset loaded %d records.", total_loaded)
