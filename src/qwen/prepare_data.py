"""Not part of the Phase 1 run flow. Kept for a later fine-tuning phase (Gita SFT / QLoRA)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.emotion.conversation import (
    Conversation,
    Turn,
    context_before_turn,
    dataframe_to_conversations,
    repair_missing_ids,
)
from src.emotion.quality import load_counselling_csv
from src.utils.config import resolve_path
from src.utils.emotions import parse_emotions
from src.utils.io import ensure_dir, save_json, save_jsonl

COUNSELLING_SYSTEM = (
    "You are an empathetic counselling assistant. Listen carefully, understand the user's "
    "emotions, and respond with warmth, respect and practical support. Do not judge the user. "
    "Do not diagnose mental disorders. Do not claim to be a licensed therapist. "
    "Do not force spiritual teachings into every response. When culturally or spiritually "
    "relevant, you may use principles grounded in the Bhagavad Gita as complementary guidance, "
    "not as a replacement for practical or professional support."
)

GITA_SYSTEM = (
    "You are an empathetic counselling assistant with knowledge grounded in Bhagavad Gita "
    "teachings. Use spiritual grounding only when it is relevant to the person's situation. "
    "Do not insert Gita quotations mechanically. Do not shame or judge the user."
)

CULTURAL_SYSTEM = (
    "You are an empathetic counselling assistant. First understand and validate the person's "
    "emotional state. Then address the actual need. Practical counselling strategies such as "
    "validation, reflection, normalization and concrete next steps are often enough. "
    "Introduce Bhagavad Gita principles only when they naturally complement the situation "
    "(for example duty, sincere effort without being consumed by outcomes, inner balance, "
    "compassion). Never force a Gita reference. Never substitute spirituality for practical help."
)


def _is_patient(speaker: str, patient_speakers: list[str]) -> bool:
    return speaker.strip().upper() in {s.upper() for s in patient_speakers}


def _is_therapist(speaker: str, therapist_speakers: list[str]) -> bool:
    return speaker.strip().upper() in {s.upper() for s in therapist_speakers}


def _history_text(turns: list[Turn]) -> str:
    if not turns:
        return "(none)"
    return "\n".join(f"{t.speaker}: {t.utterance}" for t in turns)


def counselling_examples(
    conversations: dict[str, Conversation],
    patient_speakers: list[str],
    therapist_speakers: list[str],
    num_context_turns: int,
    task_type: str = "counselling_response",
    system_prompt: str = COUNSELLING_SYSTEM,
) -> list[dict[str, Any]]:
    rows = []
    for conv in conversations.values():
        turns = conv.sorted_turns()
        for idx, turn in enumerate(turns):
            if not _is_therapist(turn.speaker, therapist_speakers):
                continue
            if not turn.utterance.strip():
                continue
            prior = turns[:idx]
            patient_turns = [t for t in prior if _is_patient(t.speaker, patient_speakers) and t.utterance.strip()]
            if not patient_turns:
                continue
            current = patient_turns[-1]
            history = context_before_turn(turns, idx, num_context_turns)
            history_without_current = [t for t in history if not (t.conversation_id == current.conversation_id and t.turn_index == current.turn_index and t.utterance == current.utterance)]
            emotions = ", ".join(current.emotions) if current.emotions else "Unknown"
            user_content = (
                f"Conversation history:\n{_history_text(history_without_current)}\n\n"
                f"Detected emotion: {emotions}\n"
                f"Current message: {current.utterance}\n\n"
                "Generate the next counselling response. Validate the emotion, stay non-judgmental, "
                "and use Gita/cultural grounding only if it is naturally relevant. Do not force it."
            )
            rows.append(
                {
                    "task_type": task_type,
                    "conversation_id": conv.conversation_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": turn.utterance.strip()},
                    ],
                    "true_emotion": current.emotions,
                    "user_message": current.utterance,
                }
            )
    return rows


def load_gita_dir(gita_dir: Path) -> pd.DataFrame:
    files = sorted(gita_dir.glob("Chapter_*_QA.csv")) + sorted(gita_dir.glob("chapter_*_qa.csv"))
    unique = []
    seen = set()
    for path in files:
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        unique.append(path)
    if not unique:
        raise FileNotFoundError(f"No Chapter_*_QA.csv files found in {gita_dir}")
    frames = []
    for path in unique:
        df = pd.read_csv(path)
        rename = {c: str(c).strip().lower() for c in df.columns}
        df = df.rename(columns=rename)
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def gita_examples(df: pd.DataFrame, max_examples: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for _, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer or question.lower() == "nan" or answer.lower() == "nan":
            continue
        chapter = row.get("chapter", "")
        verse = row.get("verse_source", "")
        user = f"Question: {question}"
        meta = []
        if pd.notna(chapter) and str(chapter).strip():
            meta.append(f"Chapter: {chapter}")
        if pd.notna(verse) and str(verse).strip():
            meta.append(f"Verse source: {verse}")
        if meta:
            user = "\n".join(meta) + "\n" + user
        rows.append(
            {
                "task_type": "gita_grounded_response",
                "messages": [
                    {"role": "system", "content": GITA_SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
                "chapter": None if pd.isna(chapter) else chapter,
                "verse_source": None if pd.isna(verse) else str(verse),
            }
        )
        if max_examples and len(rows) >= max_examples:
            break
    return rows


_THEME_TO_EMOTION = [
    (re.compile(r"fear|afraid|anxious|worry|scared", re.I), ["Scared"]),
    (re.compile(r"sad|grief|loss|sorrow", re.I), ["Sadness"]),
    (re.compile(r"anger|angry|rage|hate", re.I), ["Anger"]),
    (re.compile(r"confus|doubt|uncertain", re.I), ["Confusion"]),
    (re.compile(r"duty|work|action|effort|failure|success", re.I), ["Anticipation", "Scared"]),
    (re.compile(r"peace|calm|balance|equanim", re.I), ["Serenity"]),
]


def emotion_aware_cultural_examples(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Wrap counselling-relevant Gita Q&A in an emotion-aware counselling prompt.

    The assistant target remains the original Gita answer (not paraphrased).
    These examples are a subset: they teach optional cultural grounding, not that
    every counselling turn must cite the Gita.
    """
    rows = []
    for _, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer or question.lower() == "nan":
            continue
        emotions = []
        for pattern, labels in _THEME_TO_EMOTION:
            if pattern.search(question):
                emotions = labels
                break
        if not emotions:
            continue
        user = (
            "Conversation history: (none)\n"
            f"Detected emotion: {', '.join(emotions)}\n"
            f"Current message: {question}\n\n"
            "The person is asking for guidance. Respond empathetically. Cultural or Gita "
            "grounding is appropriate here because the question itself seeks that kind of reflection."
        )
        rows.append(
            {
                "task_type": "emotion_aware_cultural_response",
                "messages": [
                    {"role": "system", "content": CULTURAL_SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
                "true_emotion": emotions,
                "user_message": question,
            }
        )
    return rows


def load_split_conversations(csv_path: Path, split: str) -> dict[str, Conversation]:
    df = load_counselling_csv(csv_path)
    repaired, _ = repair_missing_ids(df)
    conversations, _ = dataframe_to_conversations(repaired, split, parse_emotions)
    return conversations


def prepare_qwen_dataset(config: dict[str, Any], root: Path) -> dict[str, Any]:
    paths = config["paths"]
    processed = ensure_dir(resolve_path(paths["processed_dir"], root))
    patient = config.get("patient_speakers", ["P"])
    therapist = config.get("therapist_speakers", ["T"])
    n_ctx = int(config.get("num_context_turns", 8))
    mix = config.get("mix", {})

    train_convs = load_split_conversations(resolve_path(paths["counselling_train"], root), "train")
    val_convs = load_split_conversations(resolve_path(paths["counselling_val"], root), "val")
    test_convs = load_split_conversations(resolve_path(paths["counselling_test"], root), "test")

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    if mix.get("include_counselling", True):
        c_train = counselling_examples(train_convs, patient, therapist, n_ctx)
        c_val = counselling_examples(val_convs, patient, therapist, n_ctx)
        c_test = counselling_examples(test_convs, patient, therapist, n_ctx, task_type="counselling_eval")
        max_c = mix.get("max_counselling_examples")
        if max_c:
            c_train = c_train[: int(max_c)]
        train_rows.extend(c_train)
        val_rows.extend(c_val)
        save_jsonl(c_test, processed / "heldout_counselling.jsonl")

    gita_df = None
    gita_dir = resolve_path(paths["gita_dir"], root)
    if mix.get("include_gita", True) or mix.get("include_emotion_aware_cultural", True):
        gita_df = load_gita_dir(gita_dir)

    if mix.get("include_gita", True) and gita_df is not None:
        g_rows = gita_examples(gita_df, mix.get("max_gita_examples"))
        # Keep a small gita slice in validation for format stability.
        split_at = max(1, int(0.95 * len(g_rows))) if g_rows else 0
        train_rows.extend(g_rows[:split_at])
        val_rows.extend(g_rows[split_at:])

    if mix.get("include_emotion_aware_cultural", True) and gita_df is not None:
        cultural = emotion_aware_cultural_examples(gita_df)
        split_at = max(1, int(0.95 * len(cultural))) if cultural else 0
        train_rows.extend(cultural[:split_at])
        val_rows.extend(cultural[split_at:])

    # Counselling examples that keep therapist ground truth teach the model NOT to
    # force Gita text when the real therapist response did not.
    if mix.get("include_counselling", True):
        extra = counselling_examples(
            train_convs,
            patient,
            therapist,
            n_ctx,
            task_type="counselling_with_optional_grounding",
            system_prompt=CULTURAL_SYSTEM,
        )
        train_rows.extend(extra)

    save_jsonl(train_rows, processed / "sft_train.jsonl")
    save_jsonl(val_rows, processed / "sft_val.jsonl")
    counts = {}
    for row in train_rows:
        counts[row["task_type"]] = counts.get(row["task_type"], 0) + 1
    summary = {
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "train_task_counts": counts,
        "note": "Gita answers are preserved verbatim. Therapist utterances are used as counselling targets.",
    }
    save_json(summary, processed / "sft_summary.json")
    return summary
