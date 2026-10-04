from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chatbot.chatbot import CounsellingChatbot


st.set_page_config(page_title="Culturally Grounded Counselling Chatbot", layout="centered")
st.title("Culturally Grounded Counselling Chatbot")
st.caption("Phase 1 — emotion-aware responses. Not a therapist. Short-term conversation history only.")


@st.cache_resource
def load_bot() -> CounsellingChatbot:
    bot = CounsellingChatbot()
    bot.initialize()
    return bot


bot = load_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

st.subheader("Conversation")
for turn in st.session_state.messages:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_text = st.chat_input("Share what is on your mind...")
if user_text:
    conversation_history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner("Listening..."):
            result = bot.chat(user_text, conversation_history=conversation_history, persist=False)
        st.markdown(result["response"])
    emotions = result.get("emotions") or []
    emotion_summary = ", ".join(
        f"{item['label']} ({item.get('confidence', 0):.2f})" for item in emotions
    ) or "none"
    safety = result.get("safety") or {}
    print(
        "[chatbot] "
        f"emotions={emotion_summary}; "
        f"need/context={result.get('context') or 'unspecified'}; "
        f"grounding_used={result.get('grounding_used', False)}; "
        f"crisis={safety.get('crisis', False)}; "
        f"harm_to_others={safety.get('harm_to_others', False)}",
        flush=True,
    )
    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
    st.rerun()

st.divider()
st.caption(
    "This assistant does not diagnose mental disorders and does not replace professional care. "
    "If you are in crisis in India, call KIRAN 1800-599-0019."
)
