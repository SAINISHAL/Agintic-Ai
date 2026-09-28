from __future__ import annotations

import re
from typing import Any


GITA_CUES = re.compile(
    r"\b(bhagavad\s*gita|gita|krishna|karma yoga|nishkama|dharma|atman|moksha|arjuna)\b",
    re.I,
)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def automatic_response_scores(
    user_message: str,
    response: str,
    predicted_emotions: list[str],
    true_emotions: list[str] | None = None,
) -> dict[str, Any]:
    """Heuristic automatic evaluation. Not a substitute for human judgement."""
    text = response.lower()
    user = user_message.lower()
    length = len(response.split())

    empathy_cues = [
        "understand",
        "sounds like",
        "it makes sense",
        "hear you",
        "valid",
        "that fear",
        "that sadness",
        "overwhelmed",
        "difficult",
        "with you",
    ]
    forced_cues = [
        "according to bhagavad gita",
        "your fear is caused by karma",
        "you must surrender completely",
        "this is your karma",
    ]
    unsafe_cues = ["you have depression", "you are bipolar", "i diagnose", "stop medication"]

    empathy = sum(1 for cue in empathy_cues if cue in text) / max(1, len(empathy_cues) * 0.35)
    coherence = 1.0 if 20 <= length <= 220 else 0.5
    overlap = len(set(user.split()) & set(text.split()))
    context = _clip(overlap / max(8, len(set(user.split()))))

    alignment = 0.5
    if true_emotions:
        mentioned = sum(1 for emo in true_emotions if emo.lower() in text)
        alignment = _clip(0.4 + 0.2 * mentioned)
    elif predicted_emotions:
        mentioned = sum(1 for emo in predicted_emotions if emo.lower() in text)
        alignment = _clip(0.4 + 0.2 * mentioned)

    cultural = 1.0 if GITA_CUES.search(response) else 0.0
    forced = any(cue in text for cue in forced_cues) or (cultural == 1.0 and empathy < 0.3)
    appropriateness = 0.0 if any(cue in text for cue in unsafe_cues) else 0.8
    if forced:
        appropriateness -= 0.3

    return {
        "evaluation_type": "automatic_heuristic",
        "emotional_alignment": round(_clip(alignment), 3),
        "context_relevance": round(_clip(context), 3),
        "empathy": round(_clip(empathy), 3),
        "coherence": round(_clip(coherence), 3),
        "cultural_relevance": cultural,
        "spirituality_unnecessarily_forced": bool(forced),
        "response_appropriateness": round(_clip(appropriateness), 3),
    }
