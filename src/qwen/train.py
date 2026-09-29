"""Not part of the Phase 1 run flow. Kept for a later fine-tuning phase (Gita SFT / QLoRA)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.utils.device import select_device
from src.utils.io import ensure_dir, save_json
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def _quantization_config(config: dict[str, Any], device: torch.device) -> BitsAndBytesConfig | None:
    if device.type != "cuda":
        print("4-bit / 8-bit quantization is disabled on CPU.")
        return None
    try:
        if config.get("load_in_4bit", True):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if config.get("bf16", True) and torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        if config.get("load_in_8bit", False):
            return BitsAndBytesConfig(load_in_8bit=True)
    except Exception as exc:
        print(f"Quantization config failed ({exc}). Continuing without bitsandbytes.")
        return None
    return None


def load_qwen_model_and_tokenizer(config: dict[str, Any], model_name: str | None = None):
    device = select_device()
    name = model_name or config.get("model_name", "Qwen/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=config.get("trust_remote_code", True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant = _quantization_config(config, device)
    dtype = torch.float32
    if device.type == "cuda":
        if config.get("bf16", True) and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif config.get("fp16", False):
            dtype = torch.float16

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.get("trust_remote_code", True),
        "device_map": "auto" if device.type == "cuda" else None,
    }
    if quant is not None:
        model_kwargs["quantization_config"] = quant
    else:
        model_kwargs["torch_dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    except Exception as exc:
        print(f"Quantized load failed ({exc}). Retrying without bitsandbytes.")
        model_kwargs.pop("quantization_config", None)
        model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    if device.type != "cuda":
        model.to(device)
    return model, tokenizer, device


def train_qwen(config: dict[str, Any], root: Path) -> dict[str, Any]:
    set_seed(int(config.get("seed", 42)))
    paths = config["paths"]
    processed = root / paths["processed_dir"]
    output_dir = ensure_dir(root / paths["output_dir"])
    logger = setup_logger("qwen.train", output_dir / "train.log")
    train_file = processed / "sft_train.jsonl"
    val_file = processed / "sft_val.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Missing {train_file}. Run python scripts/prepare_qwen_data.py first.")

    dataset = load_dataset("json", data_files={"train": str(train_file), "validation": str(val_file)})
    model, tokenizer, device = load_qwen_model_and_tokenizer(config)

    lora_cfg = config.get("lora", {})
    lora = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    use_bf16 = bool(config.get("bf16", True) and device.type == "cuda" and torch.cuda.is_bf16_supported())
    use_fp16 = bool(config.get("fp16", False) and device.type == "cuda" and not use_bf16)

    num_epochs = float(config.get("epochs", 2))
    batch_size = int(config.get("batch_size", 1))
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 8))

    # TRL 1.14.0 uses warmup_steps instead of warmup_ratio
    warmup_ratio = float(config.get("warmup_ratio", 0.03))

    steps_per_epoch = (
        len(dataset["train"])
        + batch_size * gradient_accumulation_steps
        - 1
    ) // (batch_size * gradient_accumulation_steps)

    total_training_steps = int(steps_per_epoch * num_epochs)
    warmup_steps = int(total_training_steps * warmup_ratio)

    common_args = dict(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(config.get("eval_batch_size", 1)),
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(config.get("learning_rate", 1e-4)),
        lr_scheduler_type=str(config.get("lr_scheduler_type", "cosine")),
        warmup_steps=warmup_steps,
        logging_steps=int(config.get("logging_steps", 10)),
        save_steps=int(config.get("save_steps", 200)),
        eval_steps=int(config.get("eval_steps", 200)),
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=bool(
            config.get("gradient_checkpointing", True)
        ),
        report_to=[],
        seed=int(config.get("seed", 42)),
        packing=False,
    )
    try:
        sft_args = SFTConfig(
            **common_args,
            eval_strategy="steps",
            max_length=int(config.get("max_seq_length", 1536)),
        )
    except TypeError:
        try:
            sft_args = SFTConfig(
                **common_args,
                evaluation_strategy="steps",
                max_length=int(config.get("max_seq_length", 1536))
            )
        except TypeError:
            sft_args = SFTConfig(**common_args)

    def formatting(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

    trainer_kwargs = dict(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting,
    )
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)
    logger.info("Starting QLoRA / LoRA fine-tuning of %s", config.get("model_name"))
    trainer.train()
    best = output_dir / "best"
    ensure_dir(best)
    trainer.save_model(str(best))
    tokenizer.save_pretrained(best)
    save_json({"model_name": config.get("model_name"), "lora": lora_cfg}, best / "finetune_meta.json")
    metrics = trainer.state.log_history
    save_json(metrics, output_dir / "training_metrics.json")
    return {"output_dir": str(best)}
