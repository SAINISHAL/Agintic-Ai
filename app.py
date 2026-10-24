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

# ── RAG status sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    rag_ready = bot._rag_enabled
    if rag_ready:
        st.success("✅ RAG: Enabled")
        if bot._rag is not None:
            stats = bot._rag.stats
            st.caption(f"Indexed chunks: {stats.get('indexed_chunks', '?')}")
            st.caption(f"Embedding: {stats.get('embedding_model', 'all-MiniLM-L6-v2').split('/')[-1]}")
    else:
        st.warning("⚠️ RAG: Disabled / not ready")
        if bot._rag is not None and bot._rag.init_error:
            st.caption(f"Reason: {bot._rag.init_error[:120]}")

    st.divider()
    st.caption("Emotion model: ✅ Loaded" if bot.emotion_model else "Emotion model: Keyword fallback")
    provider = getattr(bot, "_provider", "qwen")
    provider_label = "Mental Health LLM" if provider == "mental_llm" else "Qwen"
    st.caption(f"Generator: {provider_label}")

# ── Chat UI ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

st.subheader("Conversation")
for turn in st.session_state.messages:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        sources = turn.get("sources")
        if sources:
            with st.expander(f"📚 {len(sources)} retrieved source(s) used"):
                for i, src in enumerate(sources, 1):
                    ref = src.get("reference") or "?"
                    source = src.get("source") or "?"
                    score = src.get("score", 0.0)
                    st.caption(f"[{i}] {source} — {ref} (relevance: {score:.2f})")

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

        rag_sources = result.get("rag_sources", [])
        if rag_sources:
            with st.expander(f"📚 {len(rag_sources)} retrieved source(s) used"):
                for i, src in enumerate(rag_sources, 1):
                    ref = src.get("reference") or "?"
                    source = src.get("source") or "?"
                    score = src.get("score", 0.0)
                    st.caption(f"[{i}] {source} — {ref} (relevance: {score:.2f})")

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
        f"rag_sources={len(rag_sources)}; "
        f"crisis={safety.get('crisis', False)}; "
        f"harm_to_others={safety.get('harm_to_others', False)}",
        flush=True,
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "sources": rag_sources,
    })
    st.rerun()
