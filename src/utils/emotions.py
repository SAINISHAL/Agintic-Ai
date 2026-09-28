from __future__ import annotations

import re
from typing import Iterable

CANONICAL_EMOTIONS = [
    "Anger",
    "Annoyance",
    "Anticipation",
    "Confusion",
    "Contempt",
    "Disgust",
    "Joy",
    "Neutral",
    "Sadness",
    "Scared",
    "Serenity",
    "Surprise",
    "Trust",
]

_NORMALIZED = {name.lower(): name for name in CANONICAL_EMOTIONS}

ALIASES = {
    "fear": "Scared",
    "afraid": "Scared",
    "anxious": "Scared",
    "anxiety": "Scared",
    "sad": "Sadness",
    "happy": "Joy",
    "happiness": "Joy",
    "calm": "Serenity",
    "surprised": "Surprise",
}


def parse_emotions(raw: object) -> list[str]:
    """Split multi-emotion strings into canonical labels. Not a new class."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    parts = re.split(r"[,;/|]+", text)
    labels: list[str] = []
    seen = set()
    for part in parts:
        token = part.strip()
        if not token:
            continue
        key = re.sub(r"\s+", " ", token).lower()
        mapped = _NORMALIZED.get(key) or ALIASES.get(key)
        if mapped and mapped not in seen:
            labels.append(mapped)
            seen.add(mapped)
    return labels


def labels_to_multihot(labels: Iterable[str], emotion_list: list[str]) -> list[int]:
    index = {name: i for i, name in enumerate(emotion_list)}
    vector = [0] * len(emotion_list)
    for label in labels:
        if label in index:
            vector[index[label]] = 1
    return vector
