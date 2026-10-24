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
        self.base_model_name = base_model_name
        self.generation_config = generation_config or {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }
        self.use_modal = False
        self.modal_fn = None
        # Always initialise to None; set by _load_local_model() when needed
        self.tokenizer: AutoTokenizer | None = None
        self.model: Any | None = None

        # If local machine lacks a dedicated CUDA GPU, try the deployed Modal A100 GPU
        if self.device.type != "cuda":
            try:
                import modal

                self.modal_fn = modal.Function.from_name("nishal-work", "generate_gita")
                self.use_modal = True
                print(
                    "[INFO] Local CUDA GPU not detected. "
                    "Using deployed Modal A100 GPU for inference."
                )
            except Exception as e:
                print(
                    f"[INFO] Modal remote inference not available ({e}). "
                    "Loading model locally on CPU."
                )
                self.use_modal = False

        if not self.use_modal:
            self._load_local_model(load_in_4bit, trust_remote_code)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_local_model(self, load_in_4bit: bool, trust_remote_code: bool) -> None:
        """Load tokenizer and model onto the local device."""
        is_adapter = (self.model_path / "adapter_config.json").exists()
        tokenizer_src = str(
            self.model_path
            if (self.model_path / "tokenizer_config.json").exists()
            else (self.base_model_name or self.model_path)
        )
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
        if quant is not None:
            load_kwargs["quantization_config"] = quant
            load_kwargs["device_map"] = "auto"
        elif self.device.type == "cuda":
            load_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["low_cpu_mem_usage"] = True

        if is_adapter:
            if not self.base_model_name:
                raise ValueError(
                    "base_model_name is required when loading a LoRA adapter directory"
                )
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_name, **load_kwargs
            )
            self.model = PeftModel.from_pretrained(base, str(self.model_path))
        else:
            source = str(
                self.model_path
                if self.model_path.exists() and any(self.model_path.iterdir())
                else (self.base_model_name or self.model_path)
            )
            self.model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, messages: list[dict[str, str]]) -> str:
        # ── Modal (remote A100) path ─────────────────────────────────
        if self.use_modal and self.modal_fn is not None:
            user_msg = ""
            emotion_str = "Neutral"
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                    break

            if "Detected emotion:" in user_msg:
                for line in user_msg.split("\n"):
                    if line.startswith("Detected emotion:"):
                        emotion_str = line.replace("Detected emotion:", "").strip()
                    elif line.startswith("Current message:"):
                        user_msg = line.replace("Current message:", "").strip()

            try:
                res = self.modal_fn.remote(prompt=user_msg, emotion=emotion_str)
                if res and res.strip():
                    return res.strip()
            except Exception as e:
                print(f"[WARNING] Modal remote call failed: {e}")
                # No local CUDA → no usable fallback; surface a clear error
                if self.tokenizer is None or self.model is None:
                    raise RuntimeError(
                        "Modal inference failed and no local CUDA GPU is available "
                        "for a fallback. Please ensure the Modal app is deployed "
                        "(`modal deploy nishal_work.py`) or run on a CUDA machine. "
                        f"Original error: {e}"
                    ) from e

        # ── Local inference path ─────────────────────────────────────
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(
                "QwenGenerator: tokenizer/model not initialised. "
                "Modal inference was expected but is not reachable. "
                "Deploy the Modal app with: modal deploy nishal_work.py"
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
                max_new_tokens=int(self.generation_config.get("max_new_tokens", 256)),
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
