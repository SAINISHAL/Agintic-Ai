from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.device import select_device


class QwenGenerator:
    def __init__(
        self,
        model_path: str | Path,
        base_model_name: str | None = None,
        load_in_4bit: bool = True,
        trust_remote_code: bool = True,
        generation_config: dict[str, Any] | None = None,
    ) -> None:
        self.device = select_device()
        self.model_path = Path(model_path)
        self.generation_config = generation_config or {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }
        is_adapter = (self.model_path / "adapter_config.json").exists()
        tokenizer_src = str(self.model_path if (self.model_path / "tokenizer_config.json").exists() else (base_model_name or self.model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant = None
        if load_in_4bit and self.device.type == "cuda":
            compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if quant is not None:
            load_kwargs["quantization_config"] = quant
            load_kwargs["device_map"] = "auto"
        elif self.device.type == "cuda":
            load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            load_kwargs["device_map"] = "auto"

        if is_adapter:
            if not base_model_name:
                raise ValueError("base_model_name is required when loading a LoRA adapter directory")
            base = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
            self.model = PeftModel.from_pretrained(base, str(self.model_path))
        else:
            source = str(self.model_path if self.model_path.exists() and any(self.model_path.iterdir()) else (base_model_name or model_path))
            self.model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        self.model.eval()

    def generate(self, messages: list[dict[str, str]]) -> str:
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
                max_new_tokens=int(self.generation_config.get("max_new_tokens", 256)),
                temperature=float(self.generation_config.get("temperature", 0.7)),
                top_p=float(self.generation_config.get("top_p", 0.9)),
                repetition_penalty=float(self.generation_config.get("repetition_penalty", 1.05)),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        return text
