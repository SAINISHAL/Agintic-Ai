from __future__ import annotations

import re
from typing import Any


NEED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("career / academic pressure", re.compile(r"placement|interview|exam|job|career|college|university|grade|fail", re.I)),
    ("self-worth / comparison", re.compile(r"useless|not enough|everyone else|better than me|worthless|imposter", re.I)),
    ("relationship strain", re.compile(r"parent|family|friend|partner|lonely|relationship", re.I)),
    ("loss or grief", re.compile(r"died|death|grief|lost my", re.I)),
    ("uncertainty about the future", re.compile(r"future|don't know what|no idea what", re.I)),
    ("stress / overwhelm", re.compile(r"overwhelm|stress|too much|cannot cope|can't cope", re.I)),
    ("inner conflict / duty", re.compile(r"duty|should|obligation|responsibility|guilt", re.I)),
]


def infer_need(user_message: str, emotions: list[str]) -> str:
    matches = [name for name, pattern in NEED_PATTERNS if pattern.search(user_message)]
    emotion_bit = ", ".join(emotions) if emotions else "unspecified emotion"
    if matches:
        return f"{matches[0]} (also detected: {emotion_bit})"
    return f"emotional support related to {emotion_bit}"


def format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "(none)"
    lines = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def grounding_is_relevant(user_message: str, need: str, emotions: list[str]) -> bool:
    """Phase 1 heuristic only. Later phases may replace this with retrieval."""
    text = f"{user_message} {need} {' '.join(emotions)}".lower()
    cues = [
        "meaning",
        "purpose",
        "duty",
        "dharma",
        "gita",
        "spiritual",
        "karma",
        "outcome",
        "failure",
        "success",
        "placement",
        "future",
        "control",
        "fear of failure",
        "inner",
        "peace",
        "balance",
        "action",
    ]
    return any(cue in text for cue in cues)


def build_context(
    user_message: str,
    conversation_history: list[dict[str, str]],
    emotions: list[dict[str, Any]],
    max_turns: int = 12,
) -> dict[str, Any]:
    recent = conversation_history[-max_turns:]
    labels = [item["label"] for item in emotions]
    need = infer_need(user_message, labels)
    allow_grounding = grounding_is_relevant(user_message, need, labels)
    return {
        "history_text": format_history(recent),
        "need": need,
        "allow_cultural_grounding": allow_grounding,
        "emotion_labels": labels,
    }
