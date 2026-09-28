from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.emotion.evaluate import evaluate_checkpoint
from src.utils.config import load_config, resolve_path
from src.utils.io import ensure_dir, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate emotion classifier on the official test split")
    parser.add_argument("--config", default=str(ROOT / "configs" / "emotion.yaml"))
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    processed = resolve_path(config["paths"]["processed_dir"], ROOT)
    model_dir = Path(args.model_dir) if args.model_dir else resolve_path(config["paths"]["output_dir"], ROOT) / "best"
    reports = ensure_dir(resolve_path(config["paths"]["reports_dir"], ROOT))
    test_examples = load_jsonl(processed / "test_examples.jsonl")
    train_examples = load_jsonl(processed / "train_examples.jsonl")
    report = evaluate_checkpoint(model_dir, test_examples, train_examples, config, reports)
    print("Emotion evaluation complete.")
    print("Threshold used:", report.get("threshold_used"))
    print("Fine-tuned micro F1:", report["fine_tuned"]["micro_f1"])
    print("Fine-tuned macro F1:", report["fine_tuned"]["macro_f1"])
    if report.get("baselines"):
        print("Baselines:", report["baselines"])


if __name__ == "__main__":
    main()
