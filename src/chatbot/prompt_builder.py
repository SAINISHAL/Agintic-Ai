from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You are an empathetic counselling assistant.

Your role is to listen carefully, understand the user's emotions,
and respond with warmth, respect and practical support.

Do not judge the user.
Do not diagnose mental disorders.
Do not claim to be a licensed therapist.
Do not claim certainty about the user's mental state.
Do not encourage harmful behaviour.
Do not force spiritual teachings into every response.

When culturally or spiritually relevant, you may use principles
grounded in the Bhagavad Gita — for example focusing on sincere
action rather than being consumed by outcomes, inner balance,
compassion, or duty. Use this as complementary guidance, not as
a replacement for practical or professional support.

Never open with "According to the Bhagavad Gita" unless the user
asked for that framing. Prefer: validate emotion, name the need,
offer one or two practical next steps, and only then, if relevant,
add a brief cultural reflection.
"""


def format_emotion_block(emotions: list[dict[str, Any]]) -> str:
    if not emotions:
        return "Unknown"
    labels = ", ".join(item["label"] for item in emotions)
    conf = "\n".join(
        f"- {item['label']}: {item.get('confidence', 0):.2f}" for item in emotions
    )
    return f"{labels}\nConfidence:\n{conf}"


def build_messages(
    user_message: str,
    conversation_history: list[dict[str, str]],
    emotions: list[dict[str, Any]],
    context: dict[str, Any],
    extra_safety_note: str | None = None,
) -> list[dict[str, str]]:
    grounding_instruction = (
        "Cultural grounding: optional and relevant to this turn. You may briefly draw on "
        "Gita principles if they genuinely help. Do not lecture. Do not replace practical help."
        if context.get("allow_cultural_grounding")
        else "Cultural grounding: not needed this turn. Do not mention the Gita, karma, or scripture."
    )
    safety = extra_safety_note or ""
    user_block = f"""Conversation:
{context.get("history_text", "(none)")}

Detected emotion:
{format_emotion_block(emotions)}

Current user message:
{user_message}

Need/context:
{context.get("need", "emotional support")}

{grounding_instruction}
{safety}

Generate the next response. Be concise, warm, and specific to what the user said.
Do not reveal chain-of-thought or internal scoring.
"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in conversation_history[-8:]:
        role = turn.get("role")
        if role in {"user", "assistant"} and turn.get("content"):
            messages.append({"role": role, "content": turn["content"]})
    messages.append({"role": "user", "content": user_block})
    return messages
