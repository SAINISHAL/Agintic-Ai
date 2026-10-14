"""Modal A100-80GB Fine-Tuning and Inference pipeline for Qwen3-8B on the Bhagavad Gita Dataset.

App Name: nishal-work
GPU: NVIDIA A100 80GB VRAM
Storage: Persistent Modal Volume (nishal-qwen8b-checkpoints)
"""

import json
import os
from pathlib import Path
# pyrefly: ignore [missing-import]
import modal

# Create Modal App
app = modal.App("nishal-work")

# Image definition with CUDA, PyTorch, Hugging Face transformers, PEFT, TRL, bitsandbytes
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.48.0",
        "peft>=0.14.0",
        "trl>=0.14.0",
        "bitsandbytes>=0.45.0",
        "accelerate>=1.2.0",
        "datasets>=3.2.0",
        "pandas>=2.2.0",
        "sentencepiece",
        "protobuf",
    )
)

volume = modal.Volume.from_name("nishal-qwen8b-checkpoints", create_if_missing=True)


def verify_checkpoint(ckpt_dir: Path) -> dict[str, any]:
    """Verify that a checkpoint directory contains all files required for a true resume."""
    required_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    ]
    missing = [f for f in required_files if not (ckpt_dir / f).exists()]
    if missing:
        return {"valid": False, "missing": missing, "step": None, "epoch": None}

    try:
        with open(ckpt_dir / "trainer_state.json", "r", encoding="utf-8") as f:
            state = json.load(f)
        step = state.get("global_step")
        epoch = state.get("epoch")
        return {"valid": True, "missing": [], "step": step, "epoch": epoch}
    except Exception as e:
        return {"valid": False, "missing": [f"trainer_state_corrupt: {e}"], "step": None, "epoch": None}


def find_latest_valid_checkpoint(base_dir: Path) -> tuple[str | None, int | None]:
    """Scan the checkpoints directory and return the latest verified valid checkpoint."""
    if not base_dir.exists():
        return None, None

    ckpts = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not ckpts:
        return None, None

    def get_step(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except ValueError:
            return -1

    sorted_ckpts = sorted(ckpts, key=get_step, reverse=True)
    for ckpt in sorted_ckpts:
        check = verify_checkpoint(ckpt)
        if check["valid"]:
            print(f"[RESUME VERIFIED] Valid checkpoint found: {ckpt.name} (Global Step: {check['step']}, Epoch: {check['epoch']:.2f})")
            return str(ckpt), check["step"]
        else:
            print(f"[RESUME WARNING] Checkpoint {ckpt.name} is incomplete/corrupted (Missing: {check['missing']}). Checking earlier checkpoint...")

    return None, None


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=300,
)
def inspect_storage():
    """Inspect persistent checkpoints and datasets without allocating an expensive GPU."""
    vol_path = Path("/vol")
    data_dir = vol_path / "data"
    ckpt_dir = vol_path / "qwen3_8b_gita_lora" / "checkpoints"

    dataset_files = [f.name for f in data_dir.iterdir()] if data_dir.exists() else []
    checkpoints = {}
    if ckpt_dir.exists():
        for d in sorted(ckpt_dir.iterdir()):
            if d.is_dir():
                checkpoints[d.name] = verify_checkpoint(d)

    best_dir = vol_path / "qwen3_8b_gita_lora" / "best"
    best_exists = best_dir.exists() and (best_dir / "adapter_config.json").exists()

    latest_valid, step = find_latest_valid_checkpoint(ckpt_dir)

    return {
        "dataset_files": dataset_files,
        "checkpoints": checkpoints,
        "latest_valid_checkpoint": latest_valid,
        "latest_valid_step": step,
        "final_model_saved": best_exists,
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=600,
)
def check_gpu():
    """Verify A100-80GB GPU allocation, memory, and CUDA environment."""
    import torch

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    return {
        "status": "ready",
        "gpu": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "bf16_supported": bf16_ok,
    }


class VolumeCommitCallback:
    """Callback to commit the Modal volume after each checkpoint save."""
    def __init__(self, vol):
        self.vol = vol

    def on_save(self, args, state, control, **kwargs):
        print(f"\n[VOLUME COMMIT] Checkpoint saved at step {state.global_step}. Committing persistent volume...")
        try:
            self.vol.commit()
            print("[VOLUME COMMIT] Volume committed successfully.")
        except Exception as e:
            print(f"[VOLUME COMMIT ERROR] Failed to commit volume: {e}")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,  # 2 hours per execution chunk
    volumes={"/vol": volume},
)
def train_qwen8b(
    epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 2048,
    lora_r: int = 32,
    lora_alpha: int = 64,
    save_steps: int = 100,  # Frequent saving every 100 steps
    resume: bool = True,
):
    """Fine-tune Qwen3-8B using QLoRA on the full Bhagavad Gita dataset with safe checkpoint resume."""
    from transformers import TrainerCallback
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train_file = Path("/vol/data/sft_train.jsonl")
    val_file = Path("/vol/data/sft_val.jsonl")

    if not train_file.exists():
        raise FileNotFoundError(f"Missing {train_file} in volume. Please upload dataset first.")

    print(f"Loading dataset from {train_file}...")
    dataset = load_dataset("json", data_files={"train": str(train_file), "validation": str(val_file)})
    print(f"Dataset loaded: {len(dataset['train'])} train examples, {len(dataset['validation'])} val examples")

    model_id = "Qwen/Qwen3-8B"
    output_dir = Path("/vol/qwen3_8b_gita_lora")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Detect checkpoint if resume requested
    resume_checkpoint = None
    resumed_step = 0
    if resume:
        resume_checkpoint, resumed_step = find_latest_valid_checkpoint(ckpt_dir)

    if resume_checkpoint:
        print(f"\n=======================================================")
        print(f" >>> RESUMING TRAINING from Step {resumed_step} <<<")
        print(f" Checkpoint path: {resume_checkpoint}")
        print(f"=======================================================\n")
    else:
        print(f"\n=======================================================")
        print(f" >>> STARTING FRESH TRAINING RUN from Step 0 <<<")
        print(f"=======================================================\n")

    print(f"Loading tokenizer and base model: {model_id} on A100-80GB...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_training_steps = int((len(dataset["train"]) // (batch_size * gradient_accumulation_steps)) * epochs)
    warmup_steps = int(total_training_steps * 0.05)

    sft_args = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_steps=save_steps,
        eval_steps=save_steps,
        save_total_limit=4,  # Keep up to 4 recent checkpoints for safe fallback
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        report_to=[],
        seed=42,
        max_length=max_seq_length,
    )

    class AutoCommitHFCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            print(f"\n[VOLUME COMMIT] Checkpoint successfully saved at step {state.global_step}.")
            try:
                volume.commit()
                print("[VOLUME COMMIT] Persistent storage committed.")
            except Exception as e:
                print(f"[VOLUME COMMIT WARNING] Could not commit volume: {e}")

    def formatting(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting,
        processing_class=tokenizer,
        callbacks=[AutoCommitHFCallback()],
    )

    print("Beginning execution loop...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Commit final weights
    volume.commit()
    print(f"\n[COMPLETE] Fine-tuning finished! LoRA adapter saved to {best_dir} on persistent volume.")
    return {
        "status": "success",
        "output_dir": str(best_dir),
        "resumed_from": resume_checkpoint,
        "epochs": epochs,
        "dataset_size": len(dataset["train"]),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=600,
    volumes={"/vol": volume},
)
def generate_gita(prompt: str, emotion: str = "Neutral") -> str:
    """Run inference with the fine-tuned Qwen3-8B model on A100-80GB."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "Qwen/Qwen3-8B"
    vol_path = Path("/vol")
    best_adapter_path = vol_path / "qwen3_8b_gita_lora" / "best"
    ckpt_dir = vol_path / "qwen3_8b_gita_lora" / "checkpoints"

    adapter_path = None
    if best_adapter_path.exists() and (best_adapter_path / "adapter_config.json").exists():
        adapter_path = str(best_adapter_path)
        print(f"Loading final trained LoRA adapter from {adapter_path}")
    else:
        latest_ckpt, step = find_latest_valid_checkpoint(ckpt_dir)
        if latest_ckpt:
            adapter_path = latest_ckpt
            print(f"Final adapter not generated yet. Loading latest valid checkpoint from step {step}: {adapter_path}")

    tokenizer_source = adapter_path if adapter_path else model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("No adapter found, falling back to base Qwen3-8B model.")
        model = base_model
    model.eval()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an empathetic counsellor grounded in the timeless wisdom of the Bhagavad Gita. "
                "Listen deeply, validate the user's emotional state, and offer practical, balanced guidance."
            ),
        },
        {"role": "user", "content": f"Detected emotion: {emotion}\nCurrent message: {prompt}"},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
        )
    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    if "</think>" in raw:
        response = raw.split("</think>")[-1].strip()
    else:
        response = raw.strip()
    return response


@app.local_entrypoint()
def main(action: str = "check"):
    if action == "inspect":
        print("Inspecting persistent storage and checkpoints on Modal volume...")
        res = inspect_storage.remote()
        print(json.dumps(res, indent=2))
    elif action == "check":
        print("Checking A100-80GB GPU allocation...")
        result = check_gpu.remote()
        print("GPU Status:", result)
    elif action == "train":
        print("Launching Qwen3-8B fine-tuning on Bhagavad Gita dataset using A100-80GB (Safe Resume Enabled)...")
        result = train_qwen8b.remote()
        print("Train Result:", result)
    elif action == "ask":
        prompt = "I feel anxious about whether my efforts will ever lead to success."
        print(f"Testing generation on A100 for: '{prompt}'")
        res = generate_gita.remote(prompt=prompt, emotion="Scared, Anticipation")
        print("\nResponse:\n", res)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,
    volumes={"/vol": volume},
)
def train():
    """Standalone remote training entrypoint for pure detached execution."""
    return train_qwen8b.local(
        epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_seq_length=2048,
        lora_r=32,
        lora_alpha=64,
        save_steps=100,
        resume=True,
    )

