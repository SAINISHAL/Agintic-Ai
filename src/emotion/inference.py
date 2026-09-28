from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.emotion.model import RobertaMultiLabelClassifier
from src.utils.io import load_json


class EmotionClassifier:
    def __init__(
        self,
        model_path: str | Path,
        device: torch.device | None = None,
        threshold: float | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        artifacts = load_json(self.model_path / "label_mapping.json")
        self.labels: list[str] = artifacts["labels"]
        self.threshold = float(threshold if threshold is not None else artifacts.get("threshold", 0.4))
        self.max_length = int(artifacts.get("max_length", 384))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path / "tokenizer")
        self.model = RobertaMultiLabelClassifier.from_pretrained(self.model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, text: str) -> list[float]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])["logits"]
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        return [float(p) for p in probs]

    def predict(self, text: str, min_labels: int = 1) -> dict[str, Any]:
        probs = self.predict_proba(text)
        emotions = [
            {"label": label, "confidence": round(prob, 4)}
            for label, prob in zip(self.labels, probs)
            if prob >= self.threshold
        ]
        emotions.sort(key=lambda item: item["confidence"], reverse=True)
        if not emotions and min_labels:
            best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
            emotions = [{"label": self.labels[best_idx], "confidence": round(probs[best_idx], 4)}]
        return {
            "emotions": emotions,
            "threshold": self.threshold,
            "probabilities": {label: round(prob, 4) for label, prob in zip(self.labels, probs)},
        }
