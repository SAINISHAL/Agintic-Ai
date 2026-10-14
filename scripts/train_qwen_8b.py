"""Fine-tune Qwen3-8B using QLoRA / LoRA on NVIDIA A100-80GB GPU.

Supports training on:
1. Complete Bhagavad Gita dataset (all 18 chapters, 12,900+ Q&A pairs)
2. Mixed Bhagavad Gita + Empathetic Counselling dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.qwen.train import train_qwen
from src.utils.config import load_config, project_root
from src.utils.device import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA / LoRA fine-tuning of Qwen3-8B")
    parser.add_argument("--config", default=str(ROOT / "configs" / "qwen_8b.yaml"))
    parser.add_argument(
        "--dataset",
        choices=["gita", "mixed"],
        default="gita",
        help="Choose between 'gita' (pure Bhagavad Gita) or 'mixed' (Gita + counselling)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Adjust processed data directory based on dataset choice
    if args.dataset == "gita":
        config["paths"]["processed_dir"] = "data/processed/gita_8b"
        config["paths"]["output_dir"] = "models/qwen_8b_gita"
    else:
        config["paths"]["processed_dir"] = "data/processed/qwen_8b"
        config["paths"]["output_dir"] = "models/qwen_8b"

    print("==================================================")
    print("   Qwen3-8B Fine-Tuning on NVIDIA A100-80GB       ")
    print("==================================================")
    print(f"Model: {config.get('model_name')}")
    print(f"Dataset mode: {args.dataset} ({config['paths']['processed_dir']})")
    print(f"LoRA Rank (r): {config.get('lora', {}).get('r', 32)}")
    print(f"LoRA Alpha: {config.get('lora', {}).get('alpha', 64)}")
    print(f"Batch Size: {config.get('batch_size', 4)} (Grad Accum: {config.get('gradient_accumulation_steps', 4)})")
    print(f"Max Sequence Length: {config.get('max_seq_length', 2048)}")
    print(f"Precision: bf16={config.get('bf16', True)}, 4-bit={config.get('load_in_4bit', True)}")

    # Device check
    device = select_device()
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected GPU: {gpu_name} ({total_vram:.2f} GB VRAM)")
    else:
        print("WARNING: No CUDA GPU detected locally. Run on Modal for A100 execution.")

    print("==================================================\n")

    root = project_root()
    result = train_qwen(config, root)
    print("\nTraining completed successfully!")
    print(f"Model checkpoints saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
