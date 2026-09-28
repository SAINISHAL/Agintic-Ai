from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.emotion.prepare import prepare_emotion_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality checks + conversation reconstruction + emotion examples")
    parser.add_argument("--config", default=str(ROOT / "configs" / "emotion.yaml"))
    args = parser.parse_args()
    result = prepare_emotion_data(args.config)
    print("Emotion data preparation complete.")
    print(result)


if __name__ == "__main__":
    main()
