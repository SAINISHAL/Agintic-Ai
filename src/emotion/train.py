from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.emotion.dataset import EmotionMultiLabelDataset
from src.emotion.metrics import compute_multilabel_metrics, tune_threshold
from src.emotion.model import RobertaMultiLabelClassifier
from src.utils.device import select_device
from src.utils.io import ensure_dir, save_json
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def collect_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_probs = []
    all_labels = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            losses.append(float(outputs["loss"].item()))
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["labels"].cpu().numpy())
    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    return y_true, y_prob, float(np.mean(losses) if losses else 0.0)


def train_emotion_model(
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(int(config.get("seed", 42)))
    device = select_device()
    logger = setup_logger("emotion.train", output_dir / "train.log")
    labels = config["emotions"]
    max_length = int(config.get("max_length", 384))
    base_model = config.get("base_model", "roberta-base")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_ds = EmotionMultiLabelDataset(train_examples, tokenizer, max_length)
    val_ds = EmotionMultiLabelDataset(val_examples, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 8)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(config.get("eval_batch_size", 16)), shuffle=False)

    model = RobertaMultiLabelClassifier(
        model_name=base_model,
        num_labels=len(labels),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 2e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    epochs = int(config.get("epochs", 4))
    total_steps = max(1, len(train_loader) * epochs)
    warmup = int(total_steps * float(config.get("warmup_ratio", 0.06)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    best_micro = -1.0
    history = []
    best_dir = output_dir / "best"
    ensure_dir(best_dir)

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running.append(float(loss.item()))

        y_true, y_prob, val_loss = collect_probs(model, val_loader, device)
        provisional = compute_multilabel_metrics(y_true, y_prob, labels, float(config.get("threshold", 0.4)))
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running) if running else 0.0),
            "val_loss": val_loss,
            "val_micro_f1": provisional["micro_f1"],
            "val_macro_f1": provisional["macro_f1"],
        }
        history.append(epoch_row)
        logger.info(epoch_row)

        if provisional["micro_f1"] > best_micro:
            best_micro = provisional["micro_f1"]
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir / "tokenizer")
            save_json({"epoch": epoch, **epoch_row}, best_dir / "best_epoch.json")

    model = RobertaMultiLabelClassifier.from_pretrained(best_dir, map_location=device).to(device)
    y_true, y_prob, val_loss = collect_probs(model, val_loader, device)
    search = config.get("threshold_search", {})
    tuned = tune_threshold(
        y_true,
        y_prob,
        labels,
        min_t=float(search.get("min", 0.15)),
        max_t=float(search.get("max", 0.85)),
        step=float(search.get("step", 0.05)),
        selection_metric=str(search.get("selection_metric", "micro_f1")),
    )
    chosen_threshold = tuned["best_threshold"]
    save_json(
        {
            "labels": labels,
            "threshold": chosen_threshold,
            "max_length": max_length,
            "base_model": base_model,
            "threshold_tuning": {
                "selection_metric": tuned["selection_metric"],
                "best_score": tuned["best_score"],
                "curve": tuned["curve"],
            },
        },
        best_dir / "label_mapping.json",
    )
    save_json({"history": history, "threshold_tuning": tuned, "val_loss": val_loss}, output_dir / "training_metrics.json")
    logger.info("Selected validation threshold: %s", chosen_threshold)
    return {
        "best_dir": str(best_dir),
        "threshold": chosen_threshold,
        "history": history,
        "threshold_tuning": tuned,
    }
