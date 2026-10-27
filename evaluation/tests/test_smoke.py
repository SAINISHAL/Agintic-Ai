"""Smoke tests for Stage 3 validation of datasets, models, pipeline, and metrics."""

import os
import sys
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.chatbot.safety import safety_check, CRISIS_RESPONSE, grounding_used
from src.chatbot.context import build_context, infer_need
from src.chatbot.prompt_builder import build_messages
from src.emotion.baseline import keyword_predict
from src.emotion.metrics import compute_multilabel_metrics
from src.rag.indexer import VectorStore
from src.rag.embeddings import EmbeddingModel

def test_safety_smoke():
    """Verify safety catches explicit crisis and passes non-crisis."""
    # 1. Explicit crisis
    res = safety_check("I want to kill myself tonight")
    assert res["crisis"] is True
    assert res["override_response"] == CRISIS_RESPONSE
    assert "1800-599-0019" in res["override_response"] or "iCall" in res["override_response"]

    # 2. Harm to others
    res_harm = safety_check("I will attack and hurt them")
    assert res_harm["harm_to_others"] is True
    assert "cannot help with harming others" in res_harm["override_response"]

    # 3. Non-crisis normal sadness
    res_safe = safety_check("I had a hard day at work today and feel sad.")
    assert res_safe["crisis"] is False
    assert res_safe["harm_to_others"] is False
    assert res_safe["override_response"] is None

def test_context_smoke():
    """Verify context determination and cultural grounding gate."""
    emotions = [{"label": "Sadness", "confidence": 0.85}]
    ctx = build_context("I'm feeling so sad and lost about my career future", [], emotions)
    assert "need" in ctx
    assert "allow_cultural_grounding" in ctx
    assert isinstance(ctx["allow_cultural_grounding"], bool)

def test_prompt_builder_smoke():
    """Verify prompt builder formats messages correctly."""
    emotions = [{"label": "Sadness", "confidence": 0.85}]
    ctx = build_context("I'm stressed about exams", [], emotions)
    messages = build_messages(
        user_message="I'm stressed about exams",
        conversation_history=[],
        emotions=emotions,
        context=ctx,
        retrieved=[]
    )
    assert isinstance(messages, list)
    assert len(messages) >= 2
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "Detected emotion:" in messages[-1]["content"]

def test_emotion_baseline_smoke():
    """Verify keyword emotion baseline returns valid predictions."""
    labels = ["Neutral", "Sadness", "Joy", "Anger"]
    preds = keyword_predict("I am feeling so sad, depressed, and hopeless.", labels)
    assert "Sadness" in preds

def test_metrics_smoke():
    """Verify multilabel metrics computation works correctly."""
    import numpy as np
    y_true = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    y_prob = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.1], [0.7, 0.6, 0.2]])
    labels = ["Sadness", "Joy", "Anger"]
    metrics = compute_multilabel_metrics(y_true, y_prob, labels, threshold=0.5)
    assert "micro_f1" in metrics
    assert "macro_f1" in metrics
    assert "exact_match_ratio" in metrics
    assert metrics["exact_match_ratio"] == 1.0

def test_retrieval_smoke():
    """Verify ChromaDB vector store can retrieve chunks."""
    index_dir = "data/rag_index"
    if os.path.exists(index_dir):
        store = VectorStore(index_dir=index_dir)
        embed_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        query_vec = embed_model.encode(["Dealing with intense grief and sorrow"])[0]
        results = store.query(query_vec, top_k=2)
        assert len(results) > 0
        assert "document" in results[0]
        assert len(results[0]["document"]) > 0

def test_grounding_used_smoke():
    """Verify cultural grounding regex detection."""
    assert grounding_used("According to the Bhagavad Gita, we must focus on action.") is True
    assert grounding_used("I understand you are having a difficult time today.") is False
