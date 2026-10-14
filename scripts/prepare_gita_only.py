"""Prepare complete Bhagavad Gita dataset (all 18 chapters) for Qwen3-8B fine-tuning.

Extracts all 12,900+ Q&A pairs from data/gita/Chapter_*_QA.csv, creates chat-formatted
training examples with diverse empathetic and philosophical prompts, and outputs
train and validation JSONL splits.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, save_json, save_jsonl

GITA_COUNSELLING_SYSTEM = (
    "You are an empathetic counsellor grounded in the timeless wisdom of the Bhagavad Gita. "
    "Listen deeply, validate the user's emotional state, and offer practical, balanced guidance "
    "rooted in Gita principles such as detached action (karma yoga), equanimity (samatvam), "
    "inner strength, and self-awareness."
)

GITA_SCHOLARLY_SYSTEM = (
    "You are a wise and compassionate guide grounded in the teachings of the Bhagavad Gita. "
    "Provide clear, profound counsel addressing duty, inner conflict, and righteous action."
)

_THEME_TO_EMOTION = [
    (re.compile(r"fear|afraid|anxious|worry|scared|panic|nervous", re.I), ["Scared"]),
    (re.compile(r"sad|grief|loss|sorrow|depressed|hopeless|cry|lonely", re.I), ["Sadness"]),
    (re.compile(r"anger|angry|rage|hate|frustrat|bitter|resent", re.I), ["Anger"]),
    (re.compile(r"confus|doubt|uncertain|indecis|lost|torn", re.I), ["Confusion"]),
    (re.compile(r"duty|work|action|effort|failure|success|career|job|exam|interview", re.I), ["Anticipation"]),
    (re.compile(r"peace|calm|balance|equanim|seren|meditat", re.I), ["Serenity"]),
    (re.compile(r"joy|happy|gratitude|bless|content", re.I), ["Joy"]),
    (re.compile(r"trust|faith|devot|surrender|bhakti", re.I), ["Trust"]),
]


def detect_question_emotion(text: str) -> list[str]:
    for pattern, labels in _THEME_TO_EMOTION:
        if pattern.search(text):
            return labels
    return ["Neutral"]


def build_gita_sft_dataset(
    gita_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    random.seed(seed)
    ensure_dir(output_dir)

    files = sorted(gita_dir.glob("Chapter_*_QA.csv")) + sorted(gita_dir.glob("chapter_*_qa.csv"))
    unique_files = []
    seen = set()
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique_files.append(f)

    if not unique_files:
        raise FileNotFoundError(f"No Chapter_*_QA.csv files found in {gita_dir}")

    all_rows: list[dict[str, Any]] = []
    chapter_stats = {}

    for path in unique_files:
        df = pd.read_csv(path)
        rename = {c: str(c).strip().lower() for c in df.columns}
        df = df.rename(columns=rename)
        chapter_count = 0

        for _, row in df.iterrows():
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer or question.lower() == "nan" or answer.lower() == "nan":
                continue

            chapter = str(row.get("chapter", "")).strip()
            verse = str(row.get("verse_source", "")).strip()
            emotions = detect_question_emotion(question)

            # Style 1: Direct counselling style with emotion awareness
            emotion_str = ", ".join(emotions)
            direct_msg = [
                {"role": "system", "content": GITA_COUNSELLING_SYSTEM},
                {
                    "role": "user",
                    "content": f"Detected emotion: {emotion_str}\nCurrent message: {question}\n\nPlease offer guidance grounded in Bhagavad Gita wisdom.",
                },
                {"role": "assistant", "content": answer},
            ]

            # Style 2: Question with chapter/verse metadata
            meta_parts = []
            if chapter and chapter != "nan":
                meta_parts.append(f"Chapter: {chapter}")
            if verse and verse != "nan":
                meta_parts.append(f"Verse source: {verse}")
            meta_header = ("\n".join(meta_parts) + "\n") if meta_parts else ""

            scholarly_msg = [
                {"role": "system", "content": GITA_SCHOLARLY_SYSTEM},
                {"role": "user", "content": f"{meta_header}Question: {question}"},
                {"role": "assistant", "content": answer},
            ]

            # Alternate styles to create a robust model capable of handling both prompts
            chosen_msg = direct_msg if (chapter_count % 2 == 0) else scholarly_msg

            all_rows.append({
                "messages": chosen_msg,
                "chapter": chapter if chapter != "nan" else "",
                "verse_source": verse if verse != "nan" else "",
                "emotions": emotions,
                "source_file": path.name,
            })
            chapter_count += 1

        chapter_stats[path.name] = chapter_count

    # Shuffle dataset
    random.shuffle(all_rows)

    val_count = max(1, int(len(all_rows) * val_ratio))
    val_rows = all_rows[:val_count]
    train_rows = all_rows[val_count:]

    train_file = output_dir / "sft_train.jsonl"
    val_file = output_dir / "sft_val.jsonl"
    save_jsonl(train_rows, train_file)
    save_jsonl(val_rows, val_file)

    summary = {
        "total_examples": len(all_rows),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "chapters_processed": len(chapter_stats),
        "chapter_breakdown": chapter_stats,
        "train_path": str(train_file),
        "val_path": str(val_file),
    }
    save_json(summary, output_dir / "dataset_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare complete Bhagavad Gita Q&A dataset for Qwen3-8B fine-tuning")
    parser.add_argument("--gita-dir", default=str(ROOT / "data" / "gita"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed" / "gita_8b"))
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading Bhagavad Gita CSVs from: {args.gita_dir}")
    summary = build_gita_sft_dataset(
        gita_dir=Path(args.gita_dir),
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print("\nDataset preparation completed successfully!")
    print(f"Total Q&A pairs: {summary['total_examples']}")
    print(f"Train split: {summary['train_examples']} rows -> {summary['train_path']}")
    print(f"Val split: {summary['val_examples']} rows -> {summary['val_path']}")


if __name__ == "__main__":
    main()
