from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

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
        self.modal_service = None
        self.modal_fn = None
        self.tokenizer: AutoTokenizer | None = None
        self.model: Any | None = None

        # If local machine lacks a dedicated CUDA GPU, connect to the deployed warm Modal A100 service
        if self.device.type != "cuda":
            try:
                import modal

                # Connect to the warm class-based inference service
                gita_cls = modal.Cls.from_name("nishal-work", "GitaInference")
                self.modal_service = gita_cls()
                self.use_modal = True
                print(
                    "[INFO] Local CUDA GPU not detected. "
                    "Connected to warm Modal A100 GitaInference service."
                )
            except Exception as e_cls:
                try:
                    import modal
                    self.modal_fn = modal.Function.from_name("nishal-work", "generate_gita")
                    self.use_modal = True
                    print(
                        f"[INFO] GitaInference class connection fell back to function ({e_cls}). "
                        "Using deployed Modal A100 function."
                    )
                except Exception as e_fn:
                    print(
                        f"[INFO] Modal remote inference not available ({e_fn}). "
                        "Loading model locally on CPU."
                    )
                    self.use_modal = False

        if not self.use_modal:
            self._load_local_model(load_in_4bit, trust_remote_code)

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

    def _extract_prompt_and_emotion(self, messages: list[dict[str, str]]) -> tuple[str, str]:
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

        return user_msg, emotion_str

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate complete response synchronously."""
        max_tokens = int(self.generation_config.get("max_new_tokens", 256))

        # ── Modal (remote A100) path ─────────────────────────────────
        if self.use_modal:
            user_msg, emotion_str = self._extract_prompt_and_emotion(messages)
            try:
                if self.modal_service is not None:
                    res = self.modal_service.generate.remote(
                        prompt=user_msg, emotion=emotion_str, max_new_tokens=max_tokens
                    )
                elif self.modal_fn is not None:
                    res = self.modal_fn.remote(prompt=user_msg, emotion=emotion_str)
                else:
                    res = None

                if res and res.strip():
                    return res.strip()
            except Exception as e:
                print(f"[WARNING] Modal remote call failed: {e}")
                if self.tokenizer is None or self.model is None:
                    raise RuntimeError(
                        "Modal inference failed and no local CUDA GPU is available. "
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
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
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

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream response tokens as they are generated."""
        max_tokens = int(self.generation_config.get("max_new_tokens", 256))

        if self.use_modal and self.modal_service is not None:
            user_msg, emotion_str = self._extract_prompt_and_emotion(messages)
            try:
                for chunk in self.modal_service.generate_stream.remote_gen(
                    prompt=user_msg, emotion=emotion_str, max_new_tokens=max_tokens
                ):
                    if chunk:
                        yield chunk
                return
            except Exception as e:
                print(f"[WARNING] Modal streaming failed: {e}. Falling back to full generate...")
                yield self.generate(messages)
                return

        # Fallback to single yield if local streaming is not configured
        yield self.generate(messages)
