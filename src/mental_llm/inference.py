"""MentalHealthGenerator — local inference adapter for mental-health LLMs.

Supports any HuggingFace ``AutoModelForCausalLM`` model.  The primary
targets are:

* **Falcon3-1B-MentalHealth** (``ShivomH/Falcon3-1B-MentalHealth``)
  — 1.67 B parameters, Apache-2.0 license, CPU-feasible.
* **MentaLLaMA-chat-7B** (``klyang/MentaLLaMA-chat-7B``)
  — 6.7 B parameters, MIT + LLaMA-2 license, GPU-recommended.

The class follows the same public API contract as ``QwenGenerator``:
    - ``generate(messages: list[dict]) -> str``
    - ``generate_stream(messages: list[dict]) -> Iterator[str]``

This keeps the chatbot orchestration layer model-agnostic.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.device import select_device

logger = logging.getLogger(__name__)


class MentalHealthGenerator:
    """100 % local text generator for mental-health-focused LLMs.

    Loads a HuggingFace causal-LM once at init and reuses it for all
    subsequent ``generate()`` / ``generate_stream()`` calls.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier or local path.
    device : str
        ``"auto"`` (default), ``"cpu"``, or ``"cuda"``.
    torch_dtype : str
        ``"auto"`` (default), ``"float32"``, ``"float16"``, ``"bfloat16"``.
    generation_config : dict
        Generation hyperparameters (``max_new_tokens``, ``temperature``, …).
    trust_remote_code : bool
        Forwarded to ``from_pretrained``.
    """

    # Class-level registry so we never load the same model twice in one
    # process (important for Streamlit ``@st.cache_resource``).
    _instances: dict[str, "MentalHealthGenerator"] = {}

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        generation_config: dict[str, Any] | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self._device_str = device
        self._dtype_str = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.generation_config: dict[str, Any] = generation_config or {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }

        self.tokenizer: AutoTokenizer | None = None
        self.model: Any | None = None
        self.device: torch.device | None = None
        self._load_time: float = 0.0

        self._load_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if self._device_str == "auto":
            return select_device()
        return torch.device(self._device_str)

    def _resolve_dtype(self, device: torch.device) -> torch.dtype:
        if self._dtype_str == "auto":
            if device.type == "cuda":
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(self._dtype_str, torch.float32)

    def _load_model(self) -> None:
        """Load tokenizer and model onto the resolved device."""
        t0 = time.time()
        self.device = self._resolve_device()
        dtype = self._resolve_dtype(self.device)

        logger.info(
            "[MentalHealthGenerator] Loading model '%s' on %s (dtype=%s)…",
            self.model_id, self.device, dtype,
        )
        print(f"[MentalHealthGenerator] Loading '{self.model_id}' on {self.device} ({dtype})…")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[MentalHealthGenerator] Failed to load tokenizer for '{self.model_id}': {exc}"
            ) from exc

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }

        if self.device.type == "cuda":
            load_kwargs["device_map"] = "auto"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[MentalHealthGenerator] Failed to load model '{self.model_id}': {exc}"
            ) from exc

        if self.device.type != "cuda":
            self.model.to(self.device)

        self.model.eval()
        self._load_time = time.time() - t0
        print(
            f"[MentalHealthGenerator] Model loaded on {self.device} "
            f"in {self._load_time:.1f}s (local inference ready)."
        )

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        """Build a text prompt from chat messages.

        Tries ``apply_chat_template`` first (works for Falcon3, LLaMA-2,
        etc.).  Falls back to a simple manual format if the tokenizer does
        not have a chat template.
        """
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                logger.debug(
                    "[MentalHealthGenerator] apply_chat_template failed; using manual format."
                )

        # Manual fallback (works for most instruction models)
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"### System:\n{content}\n")
            elif role == "user":
                parts.append(f"### User:\n{content}\n")
            elif role == "assistant":
                parts.append(f"### Assistant:\n{content}\n")
        parts.append("### Assistant:\n")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public API — matches QwenGenerator interface
    # ------------------------------------------------------------------

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a single response (blocking)."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(
                "[MentalHealthGenerator] Model not initialised. "
                "Ensure __init__ completed without error."
            )

        prompt = self._format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.generation_config.get("max_new_tokens", 256)),
            "temperature": float(self.generation_config.get("temperature", 0.7)),
            "top_p": float(self.generation_config.get("top_p", 0.9)),
            "repetition_penalty": float(
                self.generation_config.get("repetition_penalty", 1.05)
            ),
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated = output_ids[0, input_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Strip any residual role markers the model may echo back
        text = re.sub(r"^(### ?(User|System|Assistant):?\s*)", "", text, flags=re.I).strip()
        return text

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield the full response as a single chunk (true streaming
        requires ``TextIteratorStreamer`` which adds threading complexity;
        for now this satisfies the ``chat_stream()`` interface).
        """
        yield self.generate(messages)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def info(self) -> dict[str, Any]:
        """Return model metadata for diagnostics / sidebar display."""
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": str(self.model.dtype) if self.model else "n/a",
            "load_time_s": round(self._load_time, 1),
            "parameters": (
                sum(p.numel() for p in self.model.parameters())
                if self.model
                else 0
            ),
        }
