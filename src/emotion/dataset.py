from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.emotion.conversation import Turn, context_before_turn
from src.utils.emotions import labels_to_multihot


def format_emotion_input(history: list[Turn], current: Turn) -> str:
    lines = []
    if history:
        lines.append("[CONTEXT]")
        for turn in history:
            speaker = turn.speaker or "?"
            lines.append(f"{speaker}: {turn.utterance}")
    lines.append("[CURRENT]")
    speaker = current.speaker or "?"
    lines.append(f"{speaker}: {current.utterance}")
    return "\n".join(lines)


def build_emotion_examples(
    conversations: dict,
    emotion_list: list[str],
    num_context_turns: int,
    require_labels: bool = True,
) -> list[dict[str, Any]]:
    examples = []
    for conv in conversations.values():
        turns = conv.sorted_turns()
        for idx, turn in enumerate(turns):
            if not turn.utterance.strip():
                continue
            if require_labels and not turn.emotions:
                continue
            history = context_before_turn(turns, idx, num_context_turns)
            text = format_emotion_input(history, turn)
            labels = labels_to_multihot(turn.emotions, emotion_list)
            examples.append(
                {
                    "conversation_id": turn.conversation_id,
                    "turn_index": turn.turn_index,
                    "speaker": turn.speaker,
                    "text": text,
                    "utterance": turn.utterance,
                    "labels": labels,
                    "emotions": turn.emotions,
                    "split": turn.source_split,
                }
            )
    return examples


class EmotionMultiLabelDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.examples[index]
        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["labels"], dtype=torch.float32),
        }
