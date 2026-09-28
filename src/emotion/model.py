from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class RobertaMultiLabelClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 13,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls))
        result = {"logits": logits}
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            result["loss"] = loss_fn(logits, labels)
        return result

    def save_pretrained(self, path: str) -> None:
        from pathlib import Path

        from src.utils.io import save_json

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory / "encoder")
        torch.save(self.classifier.state_dict(), directory / "classifier.pt")
        save_json(
            {
                "num_labels": self.num_labels,
                "dropout": self.dropout.p,
                "base_model": self.config._name_or_path,
            },
            directory / "model_config.json",
        )

    @classmethod
    def from_pretrained(cls, path: str, map_location: str | torch.device = "cpu") -> "RobertaMultiLabelClassifier":
        from pathlib import Path

        from src.utils.io import load_json

        directory = Path(path)
        meta = load_json(directory / "model_config.json")
        model = cls(
            model_name=str(directory / "encoder"),
            num_labels=int(meta["num_labels"]),
            dropout=float(meta.get("dropout", 0.1)),
        )
        state = torch.load(directory / "classifier.pt", map_location=map_location)
        model.classifier.load_state_dict(state)
        return model
