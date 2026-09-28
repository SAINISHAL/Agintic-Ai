from __future__ import annotations

import re
from typing import Any


CRISIS_PATTERNS = re.compile(
    r"\b(suicid|kill myself|end my life|self[- ]harm|want to die|hurt myself|no reason to live)\b",
    re.I,
)

HARM_OTHERS = re.compile(r"\b(kill them|hurt them|i will attack)\b", re.I)


CRISIS_RESPONSE = (
    "I'm really glad you told me this. I am not a therapist, and what you are describing "
    "sounds like you need immediate human support rather than a chatbot.\n\n"
    "If you are in danger right now, please contact local emergency services.\n"
    "In India you can reach KIRAN at 1800-599-0019, or iCall at +91 9152987821.\n"
    "If you can, tell someone you trust what you are going through.\n\n"
    "You deserve support. Please reach out to a professional or emergency service now."
)


def safety_check(user_message: str) -> dict[str, Any]:
    crisis = bool(CRISIS_PATTERNS.search(user_message))
    harm = bool(HARM_OTHERS.search(user_message))
    note = (
        "Safety: If the user is in crisis, encourage professional/emergency help. "
        "Do not diagnose. Do not provide harmful instructions."
    )
    override = None
    if crisis:
        override = CRISIS_RESPONSE
    elif harm:
        override = (
            "I cannot help with harming others. If you are in crisis, please contact local "
            "emergency services. If you want to talk about anger or fear in a safe way, I can listen."
        )
    return {
        "crisis": crisis,
        "harm_to_others": harm,
        "override_response": override,
        "prompt_note": note,
    }


def grounding_used(response: str) -> bool:
    return bool(
        re.search(
            r"\b(bhagavad\s*gita|the gita|krishna|karma yoga|nishkama|dharma)\b",
            response,
            re.I,
        )
    )
