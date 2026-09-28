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
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None

if st.button("Clear conversation"):
    bot.clear_history()
    st.session_state.messages = []
    st.session_state.last_meta = None
    st.rerun()

st.subheader("Conversation")
for turn in st.session_state.messages:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_text = st.chat_input("Share what is on your mind...")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner("Listening..."):
            result = bot.chat(user_text)
        st.markdown(result["response"])
    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
    st.session_state.last_meta = result
    st.rerun()

meta = st.session_state.last_meta
if meta:
    st.divider()
    emotions = meta.get("emotions") or []
    labels = ", ".join(item["label"] for item in emotions) if emotions else "—"
    st.markdown(f"**Emotion detected:** {labels}")
    st.markdown("**Confidence:**")
    for item in emotions:
        st.markdown(f"- {item['label']}: {item.get('confidence', 0):.2f}")
    st.markdown(f"**Need / context:** {meta.get('context')}")
    if meta.get("grounding_used"):
        st.caption("This reply included optional cultural / Gita-inspired grounding.")
    st.divider()
    st.markdown("**Response:**")
    st.write(meta.get("response"))

st.divider()
st.caption(
    "This assistant does not diagnose mental disorders and does not replace professional care. "
    "If you are in crisis in India, call KIRAN 1800-599-0019."
)
