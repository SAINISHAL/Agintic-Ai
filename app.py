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
        # Progressive token streaming for real-time perceived responsiveness
        response_stream = bot.chat_stream(
            user_text,
            conversation_history=conversation_history,
            persist=False,
        )
        full_response = st.write_stream(response_stream)

    # Perform background metadata logging without delaying user response
    emotions = bot.detect_emotion(user_text).get("emotions", [])
    emotion_summary = ", ".join(
        f"{item['label']} ({item.get('confidence', 0):.2f})" for item in emotions
    ) or "none"
    print(
        "[chatbot] "
        f"emotions={emotion_summary}; "
        f"length={len(full_response)} chars",
        flush=True,
    )
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
