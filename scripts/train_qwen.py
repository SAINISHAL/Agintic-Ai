from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qwen.train import train_qwen
from src.utils.config import load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA / LoRA fine-tune Qwen3-4B")
    parser.add_argument("--config", default=str(ROOT / "configs" / "qwen.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    result = train_qwen(config, project_root())
    print("Qwen training complete.")
    print(result)


if __name__ == "__main__":
    main()
