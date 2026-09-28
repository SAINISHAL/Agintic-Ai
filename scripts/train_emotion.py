from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.emotion.train import train_emotion_model
from src.utils.config import load_config, resolve_path
from src.utils.io import ensure_dir, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa multi-label emotion classifier")
    parser.add_argument("--config", default=str(ROOT / "configs" / "emotion.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    processed = resolve_path(config["paths"]["processed_dir"], ROOT)
    train_path = processed / "train_examples.jsonl"
    val_path = processed / "val_examples.jsonl"
    if not train_path.exists():
        raise FileNotFoundError("Run python scripts/prepare_emotion_data.py first.")
    train_examples = load_jsonl(train_path)
    val_examples = load_jsonl(val_path)
    output_dir = ensure_dir(resolve_path(config["paths"]["output_dir"], ROOT))
    result = train_emotion_model(train_examples, val_examples, config, output_dir)
    print("Emotion training complete.")
    print(result)


if __name__ == "__main__":
    main()
