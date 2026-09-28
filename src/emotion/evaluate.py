from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.emotion.baseline import evaluate_baseline, majority_label
from src.emotion.dataset import EmotionMultiLabelDataset
from src.emotion.metrics import compute_multilabel_metrics
from src.emotion.model import RobertaMultiLabelClassifier
from src.emotion.train import collect_probs
from src.utils.device import select_device
from src.utils.io import load_json, save_json, save_jsonl


def prediction_examples(
    examples: list[dict[str, Any]],
    y_prob,
    labels: list[str],
    threshold: float,
    limit: int = 30,
) -> list[dict[str, Any]]:
    rows = []
    for i, item in enumerate(examples[:limit]):
        probs = y_prob[i]
        predicted = [
            {"label": labels[j], "confidence": round(float(probs[j]), 4)}
            for j in range(len(labels))
            if probs[j] >= threshold
        ]
        predicted.sort(key=lambda x: x["confidence"], reverse=True)
        rows.append(
            {
                "conversation_id": item.get("conversation_id"),
                "utterance": item.get("utterance"),
                "true_emotions": item.get("emotions"),
                "predicted_emotions": predicted,
            }
        )
    return rows


def evaluate_checkpoint(
    model_dir: Path,
    test_examples: list[dict[str, Any]],
    train_examples: list[dict[str, Any]] | None,
    config: dict[str, Any],
    reports_dir: Path,
) -> dict[str, Any]:
    device = select_device()
    artifacts = load_json(model_dir / "label_mapping.json")
    labels = artifacts["labels"]
    threshold = float(artifacts["threshold"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = RobertaMultiLabelClassifier.from_pretrained(model_dir, map_location=device).to(device)
    test_ds = EmotionMultiLabelDataset(test_examples, tokenizer, int(artifacts.get("max_length", 384)))
    loader = DataLoader(test_ds, batch_size=int(config.get("eval_batch_size", 16)), shuffle=False)
    y_true, y_prob, test_loss = collect_probs(model, loader, device)
    metrics = compute_multilabel_metrics(y_true, y_prob, labels, threshold)
    metrics["test_loss"] = test_loss
    metrics["n_examples"] = len(test_examples)

    examples = prediction_examples(test_examples, y_prob, labels, threshold)
    save_json(metrics, reports_dir / "test_metrics.json")
    save_jsonl(examples, reports_dir / "prediction_examples.jsonl")

    baseline_report = {}
    if train_examples:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        majority = majority_label(train_examples, labels)
        for name in ("majority", "keyword"):
            yt, yp = evaluate_baseline(test_examples, labels, name, majority)
            baseline_report[name] = {
                "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
                "precision_micro": float(precision_score(yt, yp, average="micro", zero_division=0)),
                "recall_micro": float(recall_score(yt, yp, average="micro", zero_division=0)),
                "exact_match_ratio": float(accuracy_score(yt, yp)),
            }
        save_json(baseline_report, reports_dir / "emotion_baseline_metrics.json")

    report = {
        "fine_tuned": metrics,
        "baselines": baseline_report,
        "threshold_used": threshold,
        "threshold_note": "Threshold was selected on validation data; 0.5 was not assumed.",
    }
    save_json(report, reports_dir / "evaluation_summary.json")
    return report
