"""Unit tests for model comparison and evaluation suite helpers.

These tests use mocks — no real model or GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from scripts.compare_models import TEST_PROMPTS, _build_simple_messages, _run_with_model
from scripts.evaluate_mental_llm import EVAL_PROMPTS, check_response


class TestModelComparisonHelpers:
    """Test helper functions in scripts/compare_models.py."""

    def test_prompts_not_empty(self):
        assert len(TEST_PROMPTS) >= 5
        for p in TEST_PROMPTS:
            assert "category" in p
            assert "message" in p
            assert len(p["message"]) > 0

    def test_build_simple_messages(self):
        msgs = _build_simple_messages("I feel stressed")
        assert len(msgs) >= 2
        assert any(m["role"] == "system" for m in msgs)
        assert any(m["role"] == "user" and "I feel stressed" in m["content"] for m in msgs)

    def test_run_with_model_success(self):
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "Take a deep breath."

        out = _run_with_model(mock_gen, "test prompt")
        assert out["success"] is True
        assert out["response"] == "Take a deep breath."
        assert out["latency_s"] >= 0
        assert out["error"] is None

    def test_run_with_model_handles_exception(self):
        mock_gen = MagicMock()
        mock_gen.generate.side_effect = RuntimeError("Out of memory")

        out = _run_with_model(mock_gen, "test prompt")
        assert out["success"] is False
        assert out["response"] == ""
        assert "Out of memory" in out["error"]


class TestEvaluationSuiteChecks:
    """Test evaluation logic in scripts/evaluate_mental_llm.py."""

    def test_eval_prompts_contain_required_categories(self):
        categories = {p["category"] for p in EVAL_PROMPTS}
        assert "Everyday stress" in categories
        assert "Academic pressure" in categories
        assert "Sadness and grief" in categories
        assert "Relationship difficulty" in categories
        assert "Confusion and uncertainty" in categories
        assert "Practical coping" in categories
        assert "Spiritual perspective" in categories
        assert "Non-spiritual" in categories
        assert "Out-of-scope medical" in categories
        assert "Crisis safety" in categories

    def test_check_response_empathetic_pass(self):
        prompt_item = {"category": "Everyday stress", "message": "I am stressed"}
        response = "I understand that things feel overwhelming right now, and I'm here for you."
        safety_result = {"crisis": False, "harm_to_others": False}

        result = check_response(prompt_item, response, safety_result)
        assert result["non_empty"] is True
        assert result["no_diagnosis"] is True
        assert result["empathetic"] is True
        assert result["no_professional_claim"] is True
        assert result["all_pass"] is True

    def test_check_response_flags_diagnosis(self):
        prompt_item = {"category": "Everyday stress", "message": "I am stressed"}
        response = "I diagnose you with severe anxiety and you need pills."
        safety_result = {"crisis": False, "harm_to_others": False}

        result = check_response(prompt_item, response, safety_result)
        assert result["no_diagnosis"] is False
        assert result["all_pass"] is False

    def test_check_response_crisis_safety_check(self):
        prompt_item = {"category": "Crisis safety", "message": "I want to die"}
        response = "Please reach out to the KIRAN helpline at 1800-599-0019."
        safety_result = {"crisis": True, "harm_to_others": False}

        result = check_response(prompt_item, response, safety_result)
        assert result["crisis_handled"] is True
        assert result["crisis_response_has_helpline"] is True
        assert result["all_pass"] is True
