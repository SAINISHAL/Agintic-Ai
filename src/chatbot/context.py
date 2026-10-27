from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Message-type classification
# ---------------------------------------------------------------------------

# Patterns that strongly indicate casual / greeting messages
_GREETING_PATTERNS = re.compile(
    r"^(?:h+[ei]+y*|h+i+y*|hello+|yo+|hey+|sup|what'?s\s*up|howdy|namaste|namaskar"
    r"|good\s*(?:morning|afternoon|evening|night)"
    r"|thanks?(?:\s*you)?|thank\s*u|thx|ty"
    r"|ok(?:ay)?|sure|alright|cool|nice|great|fine|yep|yeah?|nope?|no"
    r"|bye+|goodbye|see\s*(?:ya|you)|take\s*care"
    r"|(?:how\s*are\s*(?:you|u))|(?:what(?:'s|\s*is)\s*(?:up|good))"
    r"|(?:you'?re?\s*welcome)|(?:no\s*(?:problem|worries))"
    r"|(?:good\s*(?:to|2)\s*(?:talk|chat|see|hear))"
    r"|(?:i'?m?\s*(?:good|fine|okay|ok|great|well|doing\s*(?:good|great|fine|well|ok(?:ay)?)))"
    r")[\s!?.,]*$",
    re.I,
)

# Short casual messages that don't carry emotional content
_CASUAL_MAX_WORDS = 8

# Happy / positive event patterns
_HAPPY_PATTERNS = re.compile(
    r"\b(?:got\s+(?:selected|accepted|promoted|hired|the\s+job|into|an?\s+(?:offer|internship))"
    r"|passed\s+(?:my|the)|won\s+|achieved|made\s*it|cleared|cracked"
    r"|congratulat|celebrating|excited\s+(?:to|about|for)|so\s+happy|feeling\s+(?:great|amazing|wonderful)"
    r"|good\s+news|proud\s+of|dream\s+(?:come|came)\s+true)\b",
    re.I,
)

# Informational / knowledge-seeking patterns (about epics, teachings, etc.)
_INFORMATIONAL_PATTERNS = re.compile(
    r"\b(?:what\s+is|what\s+are|who\s+(?:is|was|were)|tell\s+me\s+about|explain|meaning\s+of"
    r"|teach(?:ing)?s?\s+of|main\s+(?:message|lesson|teaching)|summarize|summary\s+of"
    r"|difference\s+between|define|how\s+(?:many|much|does|did)"
    r"|can\s+you\s+(?:tell|explain|describe))\b",
    re.I,
)

# Epic / spiritual topic patterns (for informational routing)
_EPIC_TOPIC_PATTERNS = re.compile(
    r"\b(?:bhagavad\s*gita|gita|ramayana|mahabharata|krishna|rama|sita|arjuna|hanuman"
    r"|pandava|kaurava|dharma|karma|vedanta|upanishad|hindu|spiritual|vedic"
    r"|scripture|epic|sudama|draupadi|bhishma|yudhishthira|duryodhana|lakshman"
    r"|ravana|lanka|ayodhya|kurukshetra)\b",
    re.I,
)

# Deep distress indicators
_DEEP_DISTRESS_PATTERNS = re.compile(
    r"\b(?:everything\s+(?:is\s+)?falling\s+apart|can(?:'t|not)\s+(?:go\s+on|take\s+(?:it|this)\s+anymore)"
    r"|(?:feel(?:ing)?|am)\s+(?:completely\s+)?(?:broken|shattered|destroyed|empty|numb|lost|worthless|hopeless)"
    r"|no\s+(?:point|hope|reason|purpose)|don'?t\s+know\s+(?:how\s+to\s+(?:continue|go\s+on|cope|survive))"
    r"|exhausted\s+and|give\s*up|giving\s*up|can(?:'t|not)\s+(?:cope|handle|bear|stand)"
    r"|(?:life|world)\s+(?:is\s+)?(?:meaningless|pointless|not\s+worth)"
    r"|i(?:'m|\s+am)\s+(?:done|finished|tired\s+of\s+(?:everything|trying|living))"
    r"|nobody\s+(?:cares|loves|understands)|completely\s+alone"
    r"|unbearable|devastating|drowning\s+in)\b",
    re.I,
)

# Moderate distress / emotional-pain indicators
_MODERATE_DISTRESS_PATTERNS = re.compile(
    r"\b(?:fail(?:ed|ure)?|rejected|lonely|alone|struggling|stuck|confused|lost"
    r"|not\s+good\s+enough|compared\s+to|everyone\s+else|left\s+(?:me|behind)"
    r"|broken\s+(?:heart|up)|breakup|broke\s+up|betrayed|cheated"
    r"|pressure|overwhelm|stressed|anxious|worried|scared|afraid|depressed"
    r"|hurt(?:ing)?|pain(?:ful)?|crying|miss(?:ing)?\s+(?:them|her|him|you)"
    r"|don'?t\s+(?:feel|belong)|self[- ]?doubt|imposter"
    r"|relationship|argument|fight|conflict|misunderstand"
    r"|guilt(?:y)?|regret|ashamed|shame)\b",
    re.I,
)

# Mild negative patterns
_MILD_PATTERNS = re.compile(
    r"\b(?:low\s+marks|bad\s+(?:day|mood|grade)|disappointed|annoyed|irritated|frustrated"
    r"|bored|tired|sick\s+of|fed\s+up|meh|not\s+(?:my|a\s+good)\s+day"
    r"|didn'?t\s+(?:go|work|turn)\s+(?:well|out)|could\s+(?:have\s+)?been\s+better"
    r"|a\s+(?:little|bit)\s+(?:sad|upset|worried|down))\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Need / context patterns  (kept from original, slightly expanded)
# ---------------------------------------------------------------------------

NEED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("career / academic pressure", re.compile(r"placement|interview|exam|job|career|college|university|grade|fail(?:ed)?|internship|resume", re.I)),
    ("self-worth / comparison", re.compile(r"useless|not enough|everyone else|better than me|worthless|imposter|not good enough|compared to", re.I)),
    ("relationship strain", re.compile(r"parent|family|friend|partner|lonely|relationship|breakup|broke up|left me|betrayed|cheated", re.I)),
    ("loss or grief", re.compile(r"died|death|grief|lost my|passing|funeral|gone forever", re.I)),
    ("uncertainty about the future", re.compile(r"future|don't know what|no idea what|what should i do|confused about", re.I)),
    ("stress / overwhelm", re.compile(r"overwhelm|stress|too much|cannot cope|can't cope|pressure|burnout|exhausted", re.I)),
    ("inner conflict / duty", re.compile(r"duty|should|obligation|responsibility|guilt|right thing", re.I)),
    ("heartbreak / separation", re.compile(r"heartbr|broken heart|miss (?:them|her|him)|separation|moved on", re.I)),
    ("anger / frustration", re.compile(r"angry|furious|rage|hate|unfair|injustice", re.I)),
    ("fear / anxiety", re.compile(r"afraid|scared|terrified|anxious|panic|worry|nervous", re.I)),
]


def classify_message_type(user_message: str, emotions: list[str]) -> str:
    """Classify a message into one of: casual, happy, informational, emotional, crisis.

    This classification drives prompt selection, RAG gating, and token-limit decisions.
    """
    text = user_message.strip()
    word_count = len(text.split())

    # Very short messages that match greeting patterns
    if _GREETING_PATTERNS.match(text):
        return "casual"

    # Short messages with no emotional cues
    if word_count <= _CASUAL_MAX_WORDS and not _MODERATE_DISTRESS_PATTERNS.search(text) and not _MILD_PATTERNS.search(text):
        # Check if it's just a very short neutral statement
        if not _HAPPY_PATTERNS.search(text) and not _INFORMATIONAL_PATTERNS.search(text):
            # If emotions are neutral or empty, treat as casual
            emotional_labels = [e for e in emotions if e.lower() not in ("neutral", "serenity", "trust", "anticipation")]
            if not emotional_labels:
                return "casual"

    # Happy / celebratory messages
    if _HAPPY_PATTERNS.search(text):
        return "happy"

    # Informational queries (about epics, teachings, general knowledge)
    if _INFORMATIONAL_PATTERNS.search(text):
        return "informational"

    # Everything else is emotional
    return "emotional"


def estimate_intensity(user_message: str, emotions: list[str]) -> str:
    """Estimate emotional intensity: none, mild, moderate, deep.

    This drives response length, story inclusion, and RAG retrieval decisions.
    """
    text = user_message.strip()

    if _DEEP_DISTRESS_PATTERNS.search(text):
        return "deep"

    if _MODERATE_DISTRESS_PATTERNS.search(text):
        # Check for compounding signals
        moderate_matches = len(_MODERATE_DISTRESS_PATTERNS.findall(text))
        if moderate_matches >= 2 or len(text.split()) > 25:
            return "deep"
        return "moderate"

    if _MILD_PATTERNS.search(text):
        return "mild"

    # Check emotion labels for intensity hint
    distress_emotions = {"Sadness", "Scared", "Anger", "Contempt", "Disgust"}
    detected_distress = [e for e in emotions if e in distress_emotions]
    if len(detected_distress) >= 2:
        return "moderate"
    if detected_distress:
        return "mild"

    return "none"


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


def grounding_is_relevant(user_message: str, need: str, emotions: list[str], msg_type: str, intensity: str) -> bool:
    """Decide whether cultural/spiritual grounding is relevant for this turn.

    Grounding is NOT relevant for casual, happy, or none-intensity messages.
    For informational questions about epics, grounding is relevant.
    For emotional messages, only at moderate+ intensity and only when topic cues match.
    """
    # Never ground casual or happy messages
    if msg_type in ("casual", "happy"):
        return False

    # Always ground informational questions specifically about epics
    if msg_type == "informational" and _EPIC_TOPIC_PATTERNS.search(user_message):
        return True

    # For emotional messages, require moderate+ intensity
    if msg_type == "emotional" and intensity in ("moderate", "deep"):
        text = f"{user_message} {need} {' '.join(emotions)}".lower()
        cues = [
            "meaning", "purpose", "duty", "dharma", "gita", "spiritual",
            "karma", "outcome", "failure", "inner", "peace", "balance",
            "strength", "courage", "perseverance", "patience",
            "confusion", "decision", "right thing", "path",
            "ramayana", "mahabharata", "krishna", "rama", "arjuna", "hanuman",
        ]
        return any(cue in text for cue in cues)

    # Moderate+ emotional intensity even without keyword cues — allow grounding
    # but only for deep distress where a story might genuinely help
    if msg_type == "emotional" and intensity == "deep":
        return True

    return False


def should_retrieve(msg_type: str, intensity: str, user_message: str) -> bool:
    """Decide whether RAG retrieval should run for this message.

    Skips retrieval for casual, happy, and low-intensity messages to avoid
    pulling irrelevant spiritual documents for greetings.
    """
    if msg_type == "casual":
        return False
    if msg_type == "happy":
        return False
    if msg_type == "informational" and _EPIC_TOPIC_PATTERNS.search(user_message):
        return True
    if msg_type == "informational":
        return False  # General questions don't need spiritual RAG
    # Emotional messages: retrieve only at moderate+ intensity
    if intensity in ("moderate", "deep"):
        return True
    return False


def get_max_tokens(msg_type: str, intensity: str) -> int:
    """Return the appropriate max_new_tokens for the given message type and intensity."""
    if msg_type == "casual":
        return 60
    if msg_type == "happy":
        return 80
    if msg_type == "informational":
        return 200
    # Emotional
    if intensity == "mild":
        return 150
    if intensity == "moderate":
        return 256
    if intensity == "deep":
        return 400
    return 150  # default


def build_context(
    user_message: str,
    conversation_history: list[dict[str, str]],
    emotions: list[dict[str, Any]],
    max_turns: int = 12,
) -> dict[str, Any]:
    recent = conversation_history[-max_turns:]
    labels = [item["label"] for item in emotions]
    msg_type = classify_message_type(user_message, labels)
    intensity = estimate_intensity(user_message, labels)
    need = infer_need(user_message, labels)
    allow_grounding = grounding_is_relevant(user_message, need, labels, msg_type, intensity)
    return {
        "history_text": format_history(recent),
        "need": need,
        "allow_cultural_grounding": allow_grounding,
        "emotion_labels": labels,
        "message_type": msg_type,
        "intensity": intensity,
        "max_tokens": get_max_tokens(msg_type, intensity),
        "should_retrieve": should_retrieve(msg_type, intensity, user_message),
    }
