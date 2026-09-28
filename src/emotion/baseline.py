from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from src.utils.emotions import labels_to_multihot


KEYWORD_RULES: dict[str, list[str]] = {
    "Anger": [r"\bangry\b", r"\bfurious\b", r"\brage\b", r"\bhate\b"],
    "Annoyance": [r"\bannoy", r"\birritat", r"\bfrustrated\b"],
    "Anticipation": [r"\bhope", r"\blooking forward", r"\bwait"],
    "Confusion": [r"\bconfused\b", r"\bdon't know\b", r"\bdo not know\b", r"\bunsure\b"],
    "Contempt": [r"\bpathetic\b", r"\buseless\b", r"\bworthless\b"],
    "Disgust": [r"\bdisgust", r"\bnauseat", r"\brepuls"],
    "Joy": [r"\bhappy\b", r"\bglad\b", r"\bjoy", r"\bexcited\b"],
    "Sadness": [r"\bsad\b", r"\bdepress", r"\bcry", r"\bhopeless\b", r"\bunhappy\b"],
    "Scared": [r"\bscar", r"\bafraid\b", r"\banxi", r"\bworry", r"\bfear", r"\bterrified\b", r"\bnervous\b"],
    "Serenity": [r"\bcalm\b", r"\bpeace", r"\brelief"],
    "Surprise": [r"\bsurpris", r"\bshock", r"\bunexpected"],
    "Trust": [r"\btrust\b", r"\bbelieve in\b"],
}


def keyword_predict(text: str, emotion_list: list[str]) -> list[str]:
    lowered = text.lower()
    hits = []
    for label in emotion_list:
        if label == "Neutral":
            continue
        patterns = KEYWORD_RULES.get(label, [])
        if any(re.search(pattern, lowered) for pattern in patterns):
            hits.append(label)
    if not hits:
        return ["Neutral"] if "Neutral" in emotion_list else []
    return hits


def majority_label(train_examples: list[dict[str, Any]], emotion_list: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    for item in train_examples:
        counts.update(item.get("emotions") or [])
    if not counts:
        return ["Neutral"] if "Neutral" in emotion_list else [emotion_list[0]]
    return [counts.most_common(1)[0][0]]


def evaluate_baseline(
    examples: list[dict[str, Any]],
    emotion_list: list[str],
    strategy: str,
    majority: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([item["labels"] for item in examples], dtype=int)
    rows = []
    for item in examples:
        if strategy == "keyword":
            pred = keyword_predict(item["utterance"], emotion_list)
        else:
            pred = majority or ["Neutral"]
        rows.append(labels_to_multihot(pred, emotion_list))
    y_pred = np.array(rows, dtype=int)
    return y_true, y_pred
