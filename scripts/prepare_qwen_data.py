"""Not part of the Phase 1 run flow. Kept for a later fine-tuning phase (Gita SFT / QLoRA).

Requires the Gita ``Chapter_*_QA.csv`` files, which Phase 1 does not use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qwen.prepare_data import prepare_qwen_dataset
from src.utils.config import load_config, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qwen SFT jsonl from counselling + Gita Q&A")
    parser.add_argument("--config", default=str(ROOT / "configs" / "qwen.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    summary = prepare_qwen_dataset(config, project_root())
    print("Qwen SFT data preparation complete.")
    print(summary)


if __name__ == "__main__":
    main()
