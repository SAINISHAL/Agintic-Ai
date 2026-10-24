"""Unit tests for MentalHealthGenerator and chatbot provider integration.

These tests use mocks — they do NOT download or load any real model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Test MentalHealthGenerator
# ---------------------------------------------------------------------------

class TestMentalHealthGenerator:
    """Test the MentalHealthGenerator adapter with mocked model / tokenizer."""

    @patch("src.mental_llm.inference.AutoModelForCausalLM")
    @patch("src.mental_llm.inference.AutoTokenizer")
    @patch("src.mental_llm.inference.select_device")
    def test_init_loads_model_once(self, mock_device, mock_tok_cls, mock_model_cls):
        """Model and tokenizer load exactly once at init."""
        import torch
        mock_device.return_value = torch.device("cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.chat_template = None
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model_cls.from_pretrained.return_value = mock_model

        from src.mental_llm.inference import MentalHealthGenerator
        gen = MentalHealthGenerator(model_id="test/model")

        mock_tok_cls.from_pretrained.assert_called_once()
        mock_model_cls.from_pretrained.assert_called_once()
        assert gen.tokenizer is mock_tokenizer
        assert gen.model is mock_model

    @patch("src.mental_llm.inference.AutoModelForCausalLM")
    @patch("src.mental_llm.inference.AutoTokenizer")
    @patch("src.mental_llm.inference.select_device")
    def test_generate_returns_string(self, mock_device, mock_tok_cls, mock_model_cls):
        """generate() returns a string."""
        import torch
        mock_device.return_value = torch.device("cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "</s>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode.return_value = "I understand you feel stressed. Let's talk."
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_model_cls.from_pretrained.return_value = mock_model

        from src.mental_llm.inference import MentalHealthGenerator
        gen = MentalHealthGenerator(model_id="test/model")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "I feel stressed."},
        ]
        result = gen.generate(messages)
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("src.mental_llm.inference.AutoModelForCausalLM")
    @patch("src.mental_llm.inference.AutoTokenizer")
    @patch("src.mental_llm.inference.select_device")
    def test_generate_stream_yields(self, mock_device, mock_tok_cls, mock_model_cls):
        """generate_stream() yields at least one chunk."""
        import torch
        mock_device.return_value = torch.device("cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "</s>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode.return_value = "Response text"
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_cls.from_pretrained.return_value = mock_model

        from src.mental_llm.inference import MentalHealthGenerator
        gen = MentalHealthGenerator(model_id="test/model")

        chunks = list(gen.generate_stream([{"role": "user", "content": "hi"}]))
        assert len(chunks) >= 1

    @patch("src.mental_llm.inference.AutoModelForCausalLM")
    @patch("src.mental_llm.inference.AutoTokenizer")
    @patch("src.mental_llm.inference.select_device")
    def test_info_returns_metadata(self, mock_device, mock_tok_cls, mock_model_cls):
        """info property returns model metadata."""
        import torch
        mock_device.return_value = torch.device("cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "</s>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.chat_template = None
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.dtype = torch.float32
        mock_model.parameters.return_value = [torch.zeros(10)]
        mock_model_cls.from_pretrained.return_value = mock_model

        from src.mental_llm.inference import MentalHealthGenerator
        gen = MentalHealthGenerator(model_id="test/model")
        info = gen.info

        assert info["model_id"] == "test/model"
        assert "device" in info
        assert "load_time_s" in info


# ---------------------------------------------------------------------------
# Test provider selection in CounsellingChatbot
# ---------------------------------------------------------------------------

class TestProviderSelection:
    """Test that chatbot respects generation.provider config."""

    def test_default_provider_is_qwen(self):
        """Without generation section, provider defaults to qwen."""
        from src.chatbot.chatbot import CounsellingChatbot
        bot = CounsellingChatbot.__new__(CounsellingChatbot)
        bot.config = {}
        bot._provider = bot.config.get("generation", {}).get("provider", "qwen")
        assert bot._provider == "qwen"

    def test_mental_llm_provider(self):
        """With provider: mental_llm, _provider is set correctly."""
        from src.chatbot.chatbot import CounsellingChatbot
        bot = CounsellingChatbot.__new__(CounsellingChatbot)
        bot.config = {"generation": {"provider": "mental_llm"}}
        bot._provider = bot.config.get("generation", {}).get("provider", "qwen")
        assert bot._provider == "mental_llm"


# ---------------------------------------------------------------------------
# Test safety is independent of provider
# ---------------------------------------------------------------------------

class TestSafetyIndependence:
    """Safety checks must fire before any model is involved."""

    def test_crisis_still_returns_override(self):
        from src.chatbot.safety import safety_check
        result = safety_check("I want to kill myself")
        assert result["crisis"] is True
        assert result["override_response"] is not None
        assert "KIRAN" in result["override_response"]

    def test_harm_still_returns_override(self):
        from src.chatbot.safety import safety_check
        result = safety_check("I will attack them")
        assert result["harm_to_others"] is True
        assert result["override_response"] is not None

    def test_normal_message_no_override(self):
        from src.chatbot.safety import safety_check
        result = safety_check("I feel a bit anxious about my exam")
        assert result["crisis"] is False
        assert result["override_response"] is None


# ---------------------------------------------------------------------------
# Test syntax validity
# ---------------------------------------------------------------------------

class TestSyntaxValidity:
    """Verify all new/modified files compile without syntax errors."""

    def test_mental_llm_compiles(self):
        import py_compile
        py_compile.compile(str(ROOT / "src" / "mental_llm" / "__init__.py"), doraise=True)
        py_compile.compile(str(ROOT / "src" / "mental_llm" / "inference.py"), doraise=True)

    def test_chatbot_compiles(self):
        import py_compile
        py_compile.compile(str(ROOT / "src" / "chatbot" / "chatbot.py"), doraise=True)

    def test_app_compiles(self):
        import py_compile
        py_compile.compile(str(ROOT / "app.py"), doraise=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
