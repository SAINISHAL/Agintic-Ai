from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)


def binarize(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(int)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    threshold: float,
) -> dict[str, Any]:
    y_pred = binarize(y_prob, threshold)
    metrics = {
        "threshold": threshold,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match_ratio": float(accuracy_score(y_true, y_pred)),
    }

    per_label = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_label"] = per_label

    cms = multilabel_confusion_matrix(y_true, y_pred)
    metrics["per_label_confusion"] = {
        labels[i]: {
            "tn": int(cms[i][0, 0]),
            "fp": int(cms[i][0, 1]),
            "fn": int(cms[i][1, 0]),
            "tp": int(cms[i][1, 1]),
        }
        for i in range(len(labels))
    }
    return metrics


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    min_t: float = 0.15,
    max_t: float = 0.85,
    step: float = 0.05,
    selection_metric: str = "micro_f1",
) -> dict[str, Any]:
    thresholds = np.round(np.arange(min_t, max_t + 1e-9, step), 4)
    rows = []
    best = None
    for threshold in thresholds:
        metrics = compute_multilabel_metrics(y_true, y_prob, labels, float(threshold))
        score = metrics[selection_metric]
        rows.append(
            {
                "threshold": float(threshold),
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "exact_match_ratio": metrics["exact_match_ratio"],
            }
        )
        if best is None or score > best["score"]:
            best = {"threshold": float(threshold), "score": score, "metrics": metrics}

    return {
        "selection_metric": selection_metric,
        "best_threshold": best["threshold"] if best else 0.5,
        "best_score": best["score"] if best else 0.0,
        "curve": rows,
        "best_metrics": best["metrics"] if best else {},
    }
