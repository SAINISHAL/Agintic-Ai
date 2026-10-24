from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.device import select_device


class QwenGenerator:
    """100% Local Qwen text generator.

    Runs entirely locally using PyTorch. Does not import or call Modal or any cloud services.
    """

    def __init__(
        self,
        model_path: str | Path,
        base_model_name: str | None = None,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        generation_config: dict[str, Any] | None = None,
        inference_backend: str = "local",
    ) -> None:
        if inference_backend != "local":
            raise ValueError(
                f"Unsupported inference_backend: '{inference_backend}'. "
                "The application is strictly configured for local inference only."
            )

        self.device = select_device()
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.inference_backend = "local"
        self.generation_config = generation_config or {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }
        self.tokenizer: AutoTokenizer | None = None
        self.model: Any | None = None

        self._load_local_model(load_in_4bit, trust_remote_code)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_local_model(self, load_in_4bit: bool, trust_remote_code: bool) -> None:
        """Load tokenizer and model onto the local device without cloud dependencies."""
        is_adapter = (self.model_path / "adapter_config.json").exists()
        tokenizer_src = str(
            self.model_path
            if (self.model_path / "tokenizer_config.json").exists()
            else (self.base_model_name or self.model_path)
        )

        print(f"[QwenGenerator] Loading tokenizer from: {tokenizer_src}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_src, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant = None
        if load_in_4bit and self.device.type == "cuda":
            compute = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}

        if quant is not None and self.device.type == "cuda":
            load_kwargs["quantization_config"] = quant
            load_kwargs["device_map"] = "auto"
        elif self.device.type == "cuda":
            load_kwargs["dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            load_kwargs["device_map"] = "auto"
        else:
            # Local CPU mode
            load_kwargs["dtype"] = torch.float32
            load_kwargs["low_cpu_mem_usage"] = True

        if is_adapter:
            if not self.base_model_name:
                raise ValueError(
                    "base_model_name is required when loading a LoRA adapter directory"
                )
            print(f"[QwenGenerator] Loading base model '{self.base_model_name}' on {self.device}...")
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_name, **load_kwargs
            )
            print(f"[QwenGenerator] Attaching LoRA adapter from '{self.model_path}'...")
            self.model = PeftModel.from_pretrained(base, str(self.model_path))
        else:
            source = str(
                self.model_path
                if self.model_path.exists() and any(self.model_path.iterdir())
                else (self.base_model_name or self.model_path)
            )
            print(f"[QwenGenerator] Loading local model from '{source}' on {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)

        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()
        print(f"[QwenGenerator] Model successfully loaded on {self.device} (local inference ready).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Run pure local inference without any external network calls."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(
                "QwenGenerator: tokenizer or model is not initialized."
            )

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(self.generation_config.get("max_new_tokens", 150)),
                temperature=float(self.generation_config.get("temperature", 0.7)),
                top_p=float(self.generation_config.get("top_p", 0.9)),
                repetition_penalty=float(
                    self.generation_config.get("repetition_penalty", 1.05)
                ),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        return text

