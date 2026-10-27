"""Emotion classifier evaluator compiling multi-label performance against baselines."""

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath("."))


def evaluate_emotion_system() -> dict[str, Any]:
    metrics_path = "outputs/emotion/test_metrics.json"
    baselines_path = "outputs/emotion/emotion_baseline_metrics.json"

    if not os.path.exists(metrics_path) or not os.path.exists(baselines_path):
        raise FileNotFoundError(f"Missing emotion metrics in outputs/emotion: {metrics_path} or {baselines_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        fine_tuned = json.load(f)

    with open(baselines_path, "r", encoding="utf-8") as f:
        baselines = json.load(f)

    # Comparative summary table
    comparison = {
        "Fine-tuned RoBERTa": {
            "micro_f1": fine_tuned["micro_f1"],
            "macro_f1": fine_tuned["macro_f1"],
            "weighted_f1": fine_tuned["weighted_f1"],
            "precision_micro": fine_tuned["precision_micro"],
            "recall_micro": fine_tuned["recall_micro"],
            "exact_match_ratio": fine_tuned["exact_match_ratio"]
        },
        "Keyword Baseline": {
            "micro_f1": baselines["keyword"]["micro_f1"],
            "macro_f1": baselines["keyword"]["macro_f1"],
            "weighted_f1": baselines["keyword"]["weighted_f1"],
            "precision_micro": baselines["keyword"]["precision_micro"],
            "recall_micro": baselines["keyword"]["recall_micro"],
            "exact_match_ratio": baselines["keyword"]["exact_match_ratio"]
        },
        "Majority Baseline": {
            "micro_f1": baselines["majority"]["micro_f1"],
            "macro_f1": baselines["majority"]["macro_f1"],
            "weighted_f1": baselines["majority"]["weighted_f1"],
            "precision_micro": baselines["majority"]["precision_micro"],
            "recall_micro": baselines["majority"]["recall_micro"],
            "exact_match_ratio": baselines["majority"]["exact_match_ratio"]
        }
    }

    relative_gain_over_keyword = {
        "micro_f1": (fine_tuned["micro_f1"] - baselines["keyword"]["micro_f1"]) / baselines["keyword"]["micro_f1"],
        "macro_f1": (fine_tuned["macro_f1"] - baselines["keyword"]["macro_f1"]) / baselines["keyword"]["macro_f1"],
        "exact_match": (fine_tuned["exact_match_ratio"] - baselines["keyword"]["exact_match_ratio"]) / baselines["keyword"]["exact_match_ratio"]
    }

    report = {
        "dataset": "data/counselling/my_dataset_test.csv",
        "n_test": fine_tuned["n_examples"],
        "threshold": fine_tuned["threshold"],
        "model_comparison": comparison,
        "relative_gain_over_keyword": relative_gain_over_keyword,
        "per_label": fine_tuned["per_label"],
        "per_label_confusion": fine_tuned["per_label_confusion"]
    }

    os.makedirs("outputs/evaluation", exist_ok=True)
    with open("outputs/evaluation/emotion_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Emotion evaluation summary compiled: Fine-Tuned Macro-F1 = {fine_tuned['macro_f1']:.4f} vs Keyword = {baselines['keyword']['macro_f1']:.4f} vs Majority = {baselines['majority']['macro_f1']:.4f}")
    return report


if __name__ == "__main__":
    evaluate_emotion_system()
