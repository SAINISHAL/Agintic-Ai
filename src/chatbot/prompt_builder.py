from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.schemas import RetrievalResult


# ---------------------------------------------------------------------------
# System prompts per message type
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CASUAL = """You are a warm, friendly conversational companion.

Rules for this turn:
- Respond naturally and briefly (1–3 short sentences).
- Do NOT mention emotional support, mental health, counselling, or spiritual teachings.
- Do NOT assume the user is emotionally distressed.
- Match the user's energy: if they say "hii", respond with a friendly greeting.
- Be warm, genuine, and conversational — like chatting with a good friend.
"""

SYSTEM_PROMPT_HAPPY = """You are a warm, encouraging companion who celebrates the user's good news.

Rules for this turn:
- Congratulate or share in the user's happiness naturally.
- Be genuinely enthusiastic but not over-the-top.
- Respond in 1–3 short sentences.
- Do NOT introduce sadness, counselling, spiritual teachings, or unrelated advice.
- Do NOT dampen their joy with warnings or caveats.
"""

SYSTEM_PROMPT_INFORMATIONAL = """You are a knowledgeable companion well-versed in the Ramayana, Mahabharata, Bhagavad Gita, and Indian spiritual literature.

Rules for this turn:
- Answer the user's question directly, accurately, and concisely.
- Do NOT treat the user as emotionally distressed.
- Do NOT add emotional-support language unless they asked for it.
- Use accurate facts. If unsure about a specific quote, paraphrase the teaching instead.
- Do not invent scriptural quotations or falsely attribute sayings.
"""

SYSTEM_PROMPT_EMOTIONAL = """You are an empathetic, culturally grounded counselling companion.

Your approach:
1. Listen carefully and understand what the user actually said.
2. Acknowledge their specific situation and feelings first.
3. Validate without exaggerating — don't turn mild disappointment into a crisis.
4. Offer practical, specific guidance related to their actual problem.
5. When genuinely relevant and helpful, draw on stories and teachings from the Ramayana, Mahabharata, or Bhagavad Gita — but only when they truly fit the situation.

Storytelling guidelines (when using epic references):
- Briefly introduce the character and their situation.
- Explain the struggle or emotional conflict they faced.
- Describe the meaningful turning point or teaching.
- Connect the lesson directly to the user's circumstances.
- Do NOT force a story into every response. Only use one when it genuinely helps.

Story selection (choose based on the user's actual need):
- Fear, confusion, difficult decisions → Arjuna's dilemma and Krishna's guidance
- Failure, hardship, perseverance → Rama's exile and continued commitment
- Grief, separation → Episodes of Rama and Sita, handled with sensitivity
- Self-doubt, forgotten strengths → Hanuman remembering his abilities before leaping to Lanka
- Loyalty, friendship, support → Krishna and Sudama's friendship
- Anger and its consequences → Relevant Mahabharata episodes
- Patience during uncertainty → Episodes from the Ramayana showing perseverance
- Attachment, loss, acceptance → Teachings from the Bhagavad Gita, explained compassionately

Strict rules:
- Do NOT diagnose mental disorders.
- Do NOT claim to be a therapist.
- Do NOT say "everything will be fine", "just be positive", "you're overthinking", "take it lightly", or "don't be sad".
- Do NOT preach, moralize, or shame the user.
- Do NOT repeat generic advice (meditation, journaling, breathing) in every answer.
- Do NOT invent scriptural quotations or falsely attribute sayings to Krishna or Rama.
- Do NOT use spiritual stories to suggest suffering is deserved or that the user must endure mistreatment silently.
- Never open with "According to the Bhagavad Gita" unless the user specifically asked for that framing.
- Treat the user's explicit statements as the source of truth over emotion-model predictions.
"""


# ---------------------------------------------------------------------------
# Response-length instructions per intensity
# ---------------------------------------------------------------------------

_LENGTH_INSTRUCTIONS = {
    "none": "Keep your response to 1–3 short sentences.",
    "mild": "Keep your response brief and encouraging (3–5 sentences). Include a practical next step if appropriate. A spiritual reference is optional — only if it genuinely adds value.",
    "moderate": "Respond thoughtfully. First acknowledge the specific situation, then validate feelings, offer encouragement, and if relevant, connect to a suitable story or teaching from the Ramayana, Mahabharata, or Bhagavad Gita. End with a gentle next step or invitation to share more.",
    "deep": "Respond with empathy and depth. Make the user feel truly heard before offering any advice. If a story from the Ramayana, Mahabharata, or Bhagavad Gita genuinely fits, tell it concisely — explain the struggle, the turning point, and connect it to their situation. Offer one or two realistic steps. Invite them to continue sharing without pressuring them. A longer response is appropriate here when it provides meaningful support.",
}


def _get_system_prompt(msg_type: str) -> str:
    """Select the appropriate system prompt based on message type."""
    if msg_type == "casual":
        return SYSTEM_PROMPT_CASUAL
    if msg_type == "happy":
        return SYSTEM_PROMPT_HAPPY
    if msg_type == "informational":
        return SYSTEM_PROMPT_INFORMATIONAL
    return SYSTEM_PROMPT_EMOTIONAL


def format_emotion_block(emotions: list[dict[str, Any]]) -> str:
    if not emotions:
        return "Unknown"
    labels = ", ".join(item["label"] for item in emotions)
    conf = "\n".join(
        f"- {item['label']}: {item.get('confidence', 0):.2f}" for item in emotions
    )
    return f"{labels}\nConfidence:\n{conf}"


def format_rag_block(retrieved: list["RetrievalResult"]) -> str:
    """Format retrieved evidence as a prompt section.

    Returns an empty string when retrieved is empty so the section is omitted.
    """
    if not retrieved:
        return ""

    # Limit to top 3 most relevant sources to avoid overwhelming the prompt
    top = retrieved[:3]

    lines: list[str] = ["RETRIEVED KNOWLEDGE (use as factual grounding — do not copy verbatim)"]
    for i, r in enumerate(top, 1):
        lines.append(f"\n[Source {i}]")
        if r.source_text:
            lines.append(f"Source: {r.source_text}")
        if r.chapter_reference:
            lines.append(f"Reference: {r.chapter_reference}")
        if r.characters_involved:
            lines.append(f"Characters: {', '.join(r.characters_involved)}")
        if r.mental_health_themes:
            lines.append(f"Themes: {', '.join(r.mental_health_themes)}")
        lines.append(f"Content:\n{r.text.strip()}")

    lines.append(
        "\nGROUNDING INSTRUCTIONS\n"
        "Use the retrieved material ONLY when genuinely relevant to the user's situation. "
        "Weave insights naturally into your response — do not dump raw passages. "
        "Do not invent facts, quotations, chapter references, or claims. "
        "Do not mention source counts, retrieval steps, or metadata to the user. "
        "If the retrieved evidence doesn't fit, ignore it and respond with compassionate general support."
    )
    return "\n".join(lines)


def build_messages(
    user_message: str,
    conversation_history: list[dict[str, str]],
    emotions: list[dict[str, Any]],
    context: dict[str, Any],
    extra_safety_note: str | None = None,
    retrieved: list["RetrievalResult"] | None = None,
) -> list[dict[str, str]]:
    msg_type = context.get("message_type", "emotional")
    intensity = context.get("intensity", "moderate")
    system_prompt = _get_system_prompt(msg_type)

    # For casual and happy messages, build a minimal user block
    if msg_type in ("casual", "happy"):
        user_block = f"User says: {user_message}\n\nRespond naturally."
        messages = [{"role": "system", "content": system_prompt}]
        for turn in conversation_history[-6:]:
            role = turn.get("role")
            if role in {"user", "assistant"} and turn.get("content"):
                messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": user_block})
        return messages

    # For informational messages
    if msg_type == "informational":
        rag_block = format_rag_block(retrieved or [])
        user_block = f"User's question: {user_message}"
        if rag_block:
            user_block += f"\n\n{rag_block}"
        user_block += "\n\nAnswer the question directly and accurately."
        messages = [{"role": "system", "content": system_prompt}]
        for turn in conversation_history[-6:]:
            role = turn.get("role")
            if role in {"user", "assistant"} and turn.get("content"):
                messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": user_block})
        return messages

    # For emotional messages — full pipeline
    grounding_instruction = (
        "Cultural grounding: You may draw on stories and teachings from the Ramayana, "
        "Mahabharata, or Bhagavad Gita when they genuinely help. Do not force them. "
        "Do not lecture. Do not replace practical help."
        if context.get("allow_cultural_grounding")
        else "Cultural grounding: not needed this turn. Do not insert spiritual stories or references."
    )

    length_instruction = _LENGTH_INSTRUCTIONS.get(intensity, _LENGTH_INSTRUCTIONS["moderate"])
    safety = extra_safety_note or ""
    rag_block = format_rag_block(retrieved or [])

    user_block = f"""Conversation context:
{context.get("history_text", "(none)")}

Detected emotion (hints, not facts — trust the user's words over these labels):
{format_emotion_block(emotions)}

Current user message:
{user_message}

Identified need:
{context.get("need", "emotional support")}

Emotional intensity: {intensity}

{grounding_instruction}

{length_instruction}
{safety}

{rag_block}

IMPORTANT RULES:
- Treat the current user message as the source of truth.
- Do not reverse the user's explicit meaning (e.g., "0 preparation" means unprepared).
- Do not reveal internal scoring, chain-of-thought, emotion labels, or retrieval metadata to the user.
- Generate ONLY the response text — no analysis, no labels, no markdown headers.
"""
    messages = [{"role": "system", "content": system_prompt}]
    for turn in conversation_history[-8:]:
        role = turn.get("role")
        if role in {"user", "assistant"} and turn.get("content"):
            messages.append({"role": role, "content": turn["content"]})
    messages.append({"role": "user", "content": user_block})
    return messages
